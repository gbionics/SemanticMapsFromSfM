import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import time
import tqdm

from gsplat.rendering import rasterization_2dgs
from utils.utils import rgb_to_sh
from utils.gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from utils.traj import generate_interpolated_path
from nerfview import CameraState, RenderTabState

try:
    from sklearn.cluster import HDBSCAN  # type: ignore
except Exception:
    HDBSCAN = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None
 
try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None

def get_random_colors(num_colors: int):
    """Generate random RGB colors in [0,1], with index 0 reserved for background (black)."""
    import numpy as _np

    colors = [_np.asarray([0.0, 0.0, 0.0])]
    for _ in range(num_colors):
        colors.append(_np.random.rand(3).astype(_np.float32))
    return _np.stack(colors, axis=0)


def load_splats_from_ckpt(ckpt_path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    splats = ckpt["splats"]
    # convert to tensors on device
    splats_t = {k: torch.as_tensor(v, device=device) for k, v in splats.items()}
    return splats_t


def compute_combined_features(
    means: torch.Tensor,
    feats: torch.Tensor,
    spatial_weight: float = 1.0,
) -> np.ndarray:
    """Combine 3D positions and feature vectors into a single numpy array for clustering."""
    means_np = means.detach().cpu().numpy()
    feats_np = feats.detach().cpu().numpy()
    # normalize each modality
    means_norm = (means_np - means_np.mean(0)) / (means_np.std(0) + 1e-9)
    feats_norm = (feats_np - feats_np.mean(0)) / (feats_np.std(0) + 1e-9)
    combined = np.concatenate([spatial_weight * means_norm, feats_norm], axis=1)
    return combined


def cluster_gaussians(
    means: torch.Tensor,
    features: torch.Tensor,
    method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    spatial_weight: float = 1.0,
    min_cluster_size: int = 15,
    min_samples: int = 30,
    spatial_partition: bool = True,
    voxel_size: Optional[float] = None,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster gaussians using combined spatial+feature vectors.

    Returns:
        labels (N,) : int labels (-1 indicates noise)
        centroids (K, D) : cluster centroids in feature space (excluding noise)
    """
    N = len(means)
    print(f"Clustering {N} gaussians using method='{method}' (spatial_weight={spatial_weight})")
    tic = time.time()
    # If the scene is very large, try spatial partitioning (voxel grid) first
    if N > 200_000 and spatial_partition:
        means_np = means.detach().cpu().numpy()
        feats_np = features.detach().cpu().numpy()

        # choose a voxel size heuristically if not provided
        if voxel_size is None:
            extent = means_np.max(0) - means_np.min(0)
            # aim for ~50 divisions along the largest axis
            voxel_size = float(max(extent) / max(1.0, 50.0)) if max(extent) > 0 else 1.0

        coords = np.floor((means_np - means_np.min(0)) / voxel_size).astype(np.int64)
        voxels, inv = np.unique(coords, axis=0, return_inverse=True)
        n_pre = voxels.shape[0]
        print(f"Spatial partitioning: {n_pre} voxels (voxel_size={voxel_size:.4f})")

        # compute per-voxel mean positions and features
        pre_means = np.zeros((n_pre, 3), dtype=means_np.dtype)
        pre_feats = np.zeros((n_pre, feats_np.shape[1]), dtype=feats_np.dtype)
        counts = np.zeros((n_pre,), dtype=np.int64)
        for i in range(n_pre):
            mask = inv == i
            counts[i] = int(mask.sum())
            if counts[i] > 0:
                pre_means[i] = means_np[mask].mean(axis=0)
                pre_feats[i] = feats_np[mask].mean(axis=0)

        # if only a few voxels found, fall back to original approach
        if n_pre <= 1:
            print("Partitioning produced a single voxel, falling back to non-partitioned clustering")
        else:
            # combine spatial and feature modalities for voxel centers
            combined_centers = np.concatenate([
                (pre_means - pre_means.mean(0)) / (pre_means.std(0) + 1e-9) * spatial_weight,
                (pre_feats - pre_feats.mean(0)) / (pre_feats.std(0) + 1e-9),
            ], axis=1)

            if method == "hdbscan" and HDBSCAN is not None:
                clim = HDBSCAN(min_cluster_size=max(2, min_cluster_size // 10), min_samples=max(2, min_samples // 2))
                center_labels = clim.fit_predict(combined_centers)
            else:
                n_clusters_cent = n_clusters or max(2, n_pre // 50)
                if KMeans is None:
                    raise RuntimeError("Neither HDBSCAN nor KMeans is available for clustering.")
                km2 = KMeans(n_clusters=n_clusters_cent, random_state=42)
                center_labels = km2.fit_predict(combined_centers)

            # map voxel-level labels back to original gaussians
            labels = center_labels[inv]
            toc = time.time()
            print(f"Spatial-partition clustering finished in {toc-tic:.2f}s (voxels={n_pre})")
            # continue to compute centroids below
    
    # Fast two-stage path for very large N: pre-cluster features with MiniBatchKMeans (fallback)
    if N > 200_000 and (('labels' not in locals() or labels is None) and MiniBatchKMeans is not None):
        # choose number of preclusters to reduce problem size
        n_pre = min(max(2000, N // 200), 20000)
        print(f"Large scene detected (N={N}), running MiniBatchKMeans pre-clustering into {n_pre} groups")
        feats_np = features.detach().cpu().numpy()
        mbkm = MiniBatchKMeans(n_clusters=n_pre, batch_size=10_000, random_state=42)
        pre_labels = mbkm.fit_predict(feats_np)
        centers = mbkm.cluster_centers_  # [n_pre, D]

        # compute per-precluster mean positions
        means_np = means.detach().cpu().numpy()
        pre_means = np.zeros((n_pre, 3), dtype=means_np.dtype)
        counts = np.zeros((n_pre,), dtype=np.int64)
        for i in range(n_pre):
            mask = pre_labels == i
            counts[i] = mask.sum()
            if counts[i] > 0:
                pre_means[i] = means_np[mask].mean(axis=0)

        # run HDBSCAN/KMeans on precluster centers + positions
        combined_centers = np.concatenate([
            (pre_means - pre_means.mean(0)) / (pre_means.std(0) + 1e-9) * spatial_weight,
            (centers - centers.mean(0)) / (centers.std(0) + 1e-9),
        ], axis=1)

        if method == "hdbscan" and HDBSCAN is not None:
            clim = HDBSCAN(min_cluster_size=max(2, min_cluster_size // 10), min_samples=max(2, min_samples // 2))
            center_labels = clim.fit_predict(combined_centers)
        else:
            n_clusters_cent = n_clusters or max(2, n_pre // 50)
            if KMeans is None:
                raise RuntimeError("Neither HDBSCAN nor KMeans is available for clustering.")
            km2 = KMeans(n_clusters=n_clusters_cent, random_state=42)
            center_labels = km2.fit_predict(combined_centers)

        # map back to original gaussians
        labels = center_labels[pre_labels]
        toc = time.time()
        print(f"Two-stage clustering finished in {toc-tic:.2f}s (preclusters={n_pre})")
    elif 'labels' in locals() and labels is not None:
        # Labels were already produced by spatial partitioning above; skip global clustering.
        toc = time.time()
        print(f"Using spatial-partition labels; skipping global clustering (elapsed {toc-tic:.2f}s)")
    else:
        combined = compute_combined_features(means, features, spatial_weight=spatial_weight)

        labels = None
        if method == "hdbscan" and HDBSCAN is not None:
            clim = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clim.fit_predict(combined)
        elif method == "dbscan":
            if DBSCAN is None:
                raise RuntimeError("DBSCAN from scikit-learn is not available.")
            # Cluster using cosine similarity on the learned features. We normalize features per-row
            # and use metric='cosine' so DBSCAN treats close-by vectors in cosine space as neighbors.
            print(f"Running DBSCAN on features (eps={dbscan_eps}, min_samples={dbscan_min_samples})")
            feats_np = features.detach().cpu().numpy()
            # normalize rows to unit length to make cosine explicit (DBSCAN metric='cosine' works either way)
            norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12
            feats_norm = feats_np / norms
            db = DBSCAN(metric="cosine", eps=float(dbscan_eps), min_samples=int(dbscan_min_samples), n_jobs=-1)
            labels = db.fit_predict(feats_norm)
        else:
            # fallback to kmeans
            if n_clusters is None:
                n_clusters = max(2, int(len(combined) / 2000))
            if KMeans is None:
                raise RuntimeError("Neither HDBSCAN nor KMeans is available for clustering.")
            print(f"Falling back to KMeans with n_clusters={n_clusters}")
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(combined)
        toc = time.time()
        print(f"Clustering finished in {toc-tic:.2f}s")

    # compute centroids in feature space (features argument)
    labels_np = np.asarray(labels)
    unique = np.unique(labels_np)
    unique = unique[unique != -1]
    feats_np = features.detach().cpu().numpy()
    centroids = []
    # compute centroids with a progress indicator if many clusters
    if unique.size > 0:
        iterator = unique
        if unique.size > 20:
            iterator = tqdm.tqdm(unique, desc="computing centroids")
        for u in iterator:
            mask = labels_np == u
            centroids.append(feats_np[mask].mean(0))
    if len(centroids) == 0:
        centroids_arr = np.empty((0, feats_np.shape[1]), dtype=feats_np.dtype)
    else:
        centroids_arr = np.stack(centroids, axis=0)

    return labels_np, centroids_arr


def assign_cluster_colors(labels: np.ndarray, palette: Optional[np.ndarray] = None) -> np.ndarray:
    N = labels.shape[0]
    if palette is None:
        # get random colors (includes background at index 0)
        palette = get_random_colors(int(max(1, labels.max() + 1)))
    colors = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        lab = labels[i]
        if lab == -1:
            colors[i] = np.array([0.0, 0.0, 0.0])
        else:
            colors[i] = palette[lab % len(palette)][:3]
    return colors


def save_clustered_ckpt(original_ckpt: str, out_path: str, cluster_colors_rgb: np.ndarray):
    ckpt = torch.load(original_ckpt, map_location="cpu")
    splats = ckpt["splats"]
    # If checkpoint uses SH bands, convert RGB to sh0 + shN zeros for remaining bands
    if "sh0" in splats and "shN" in splats:
        sh0 = rgb_to_sh(torch.from_numpy(cluster_colors_rgb).float() / 1.0)
        splats["sh0"] = sh0.cpu().numpy()
        # leave shN unchanged or zero
        splats["shN"] = np.zeros_like(splats["shN"])
    else:
        # set 'colors' parameter (post-activation). store as logits
        colors_logit = np.log(cluster_colors_rgb.clip(1e-6, 1 - 1e-6) / (1 - cluster_colors_rgb.clip(1e-6, 1 - 1e-6)))
        splats["colors"] = colors_logit

    torch.save(ckpt, out_path)


def render_previews(
    splats: dict,
    out_dir: str,
    num_views: int = 8,
    width: int = 800,
    height: int = 600,
):
    os.makedirs(out_dir, exist_ok=True)
    # choose device: use CUDA if available so the CUDA-backed rasterizer receives CUDA tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    means = torch.as_tensor(splats["means"]).to(device).float()
    quats = torch.as_tensor(splats["quats"]).to(device).float()
    scales = torch.exp(torch.as_tensor(splats["scales"]).to(device).float())
    opacities = torch.sigmoid(torch.as_tensor(splats["opacities"]).to(device).float())

    # colors: prefer 'colors' then sh0+shN
    if "colors" in splats:
        colors = torch.as_tensor(splats["colors"]).to(device).float()
    elif "sh0" in splats:
        # reconstruct per-gaussian RGB from sh0 (only band 0)
        sh0 = torch.as_tensor(splats["sh0"]).to(device).float()
        colors = torch.sigmoid(sh0[:, 0, :])
    else:
        # fallback: random
        colors = torch.rand((means.shape[0], 3), device=device, dtype=means.dtype)

    # create simple circular camera poses around scene centroid
    centroid = means.mean(0)
    radius = (means - centroid).abs().max().item() * 4.0 + 1.0
    focal = 500.0
    # ensure K uses same dtype/device as tensors
    K = torch.tensor([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]], dtype=means.dtype, device=means.device)

    for i in tqdm.tqdm(range(num_views), desc="rendering previews"):
        theta = 2 * np.pi * (i / num_views)
        # create cam_pos and up with the same dtype/device as means to avoid type promotion
        cam_offset = torch.tensor([radius * np.cos(theta), radius * np.sin(theta), radius * 0.1], dtype=means.dtype, device=means.device)
        cam_pos = centroid.to(dtype=means.dtype) + cam_offset
        # build c2w
        up = torch.tensor([0.0, 0.0, 1.0], dtype=means.dtype, device=means.device)
        forward = (cam_pos - centroid)
        forward = forward / torch.linalg.norm(forward)
        # use torch.linalg.cross to avoid deprecated torch.cross signature
        right = torch.linalg.cross(up, forward)
        right = right / torch.linalg.norm(right)
        upv = torch.linalg.cross(forward, right)
        c2w = torch.eye(4, dtype=means.dtype, device=means.device)
        c2w[:3, 0] = right
        c2w[:3, 1] = upv
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos

        render_colors, *_ = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors.unsqueeze(0),
            viewmats=torch.linalg.inv(c2w.unsqueeze(0)),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            packed=False,
            absgrad=False,
            sparse_grad=False,
        )
        img = (render_colors[0, ..., :3].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        from imageio import imwrite

        imwrite(os.path.join(out_dir, f"preview_{i:02d}.png"), img)


def _build_splats_tensors(splats: dict, device: str = "cpu"):
    """Return tensors required by rasterization_2dgs: means, quats, scales, opacities, colors"""
    means = torch.as_tensor(splats["means"]).to(device).float()
    quats = torch.as_tensor(splats["quats"]).to(device).float()
    scales = torch.as_tensor(splats["scales"]).to(device).float()
    opacities = torch.as_tensor(splats["opacities"]).to(device).float()
    # handle colors/sh
    if "colors" in splats:
        colors = torch.as_tensor(splats["colors"]).to(device).float()
    elif "sh0" in splats:
        sh0 = torch.as_tensor(splats["sh0"]).to(device).float()
        colors = torch.sigmoid(sh0[:, 0, :])
    else:
        colors = torch.rand((means.shape[0], 3), device=device)
    return means, quats, scales, opacities, colors


def make_viewer_render_fn(splats: dict, device: str = "cpu"):
    """Create a render function suitable for `GsplatViewer` that renders the clustered splats."""
    means, quats, scales, opacities, colors = _build_splats_tensors(splats, device=device)

    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        # mirror Runner._viewer_render_fn behavior, but render clustered colors
        if isinstance(render_tab_state, GsplatRenderTabState):
            if render_tab_state.preview_render:
                width = render_tab_state.render_width
                height = render_tab_state.render_height
            else:
                width = render_tab_state.viewer_width
                height = render_tab_state.viewer_height
        else:
            width, height = 800, 600

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)

        # rasterize using post-activation colors (sh_degree=None)
        render_colors, render_alphas, render_normals, normals_from_depth, render_distort, render_median, info = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors.unsqueeze(0),
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=None,
            packed=False,
            absgrad=False,
            sparse_grad=False,
        )

        # return numpy image as Viewer expects
        img = render_colors[0, ..., :3].clamp(0, 1).detach().cpu().numpy()
        return img

    return render_fn


def render_trajectory_video(splats: dict, out_path: str, num_frames: int = 120, width: int = 800, height: int = 600):
    """Render a sequence of views around the scene centroid and save as mp4."""
    # Build a CPU-only camera path (path stored as numpy arrays) then render using CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build CPU tensors to compute centroid and path
    means_cpu, quats_cpu, scales_cpu, opacities_cpu, colors_cpu = _build_splats_tensors(splats, device="cpu")
    centroid = means_cpu.mean(0)
    radius = (means_cpu - centroid).abs().max().item() * 4.0 + 1.0

    # create simple circular path of camera-to-world matrices (kept on CPU so we can store .numpy())
    path = []
    for i in range(num_frames):
        theta = 2 * np.pi * (i / num_frames)
        cam_offset = torch.tensor([radius * np.cos(theta), radius * np.sin(theta), radius * 0.1], dtype=means_cpu.dtype, device=means_cpu.device)
        cam_pos = centroid.to(dtype=means_cpu.dtype) + cam_offset
        up = torch.tensor([0.0, 0.0, 1.0], dtype=means_cpu.dtype, device=means_cpu.device)
        forward = (cam_pos - centroid)
        forward = forward / torch.linalg.norm(forward)
        right = torch.linalg.cross(up, forward)
        right = right / torch.linalg.norm(right)
        upv = torch.linalg.cross(forward, right)
        c2w = torch.eye(4, dtype=means_cpu.dtype, device=means_cpu.device)
        c2w[:3, 0] = right
        c2w[:3, 1] = upv
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        path.append(c2w.cpu().numpy())

    # Now build tensors on the preferred device for rendering
    means, quats, scales, opacities, colors = _build_splats_tensors(splats, device=device)

    # Focal and K
    focal = 500.0
    # K on the same device/dtype as rendering tensors
    K = torch.tensor([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]], device=means.device, dtype=means.dtype)

    # Render frames and write video
    import imageio
    writer = imageio.get_writer(out_path, fps=30)
    for c2w in tqdm.tqdm(path, desc="rendering trajectory"):
        # convert cpu numpy path to rendering device/dtype
        c2w_t = torch.from_numpy(c2w).to(device=means.device, dtype=means.dtype)
        render_colors, *_ = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors.unsqueeze(0),
            viewmats=torch.linalg.inv(c2w_t.unsqueeze(0)),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            packed=False,
            absgrad=False,
            sparse_grad=False,
        )
        img = (render_colors[0, ..., :3].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        writer.append_data(img)
    writer.close()


def segment_feature_field(ckpt_path: str, out_dir: Optional[str] = None):
    # prefer CUDA for rendering if available; clustering will move tensors to CPU as needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats = load_splats_from_ckpt(ckpt_path, device=device)

    # determine feature tensor to use
    if "seg_features" in splats:
        feats = splats["seg_features"].float()
    elif "features" in splats:
        feats = splats["features"].float()
    else:
        # fallback to colors (sigmoid if logits)
        if "colors" in splats:
            c = torch.as_tensor(splats["colors"]).float()
            feats = torch.sigmoid(c)
        elif "sh0" in splats:
            sh0 = torch.as_tensor(splats["sh0"]).float()
            feats = torch.sigmoid(sh0[:, 0, :])
        else:
            raise RuntimeError("No suitable per-gaussian features found in checkpoint.")

    means = splats["means"].float()

    labels, centroids = cluster_gaussians(means, feats, method=("hdbscan" if HDBSCAN is not None else "kmeans"))
    # report cluster summary
    unique, counts = np.unique(labels, return_counts=True)
    # exclude noise label -1 from cluster count summary
    valid = unique != -1
    n_clusters = int(np.sum(valid))
    print(f"Found {n_clusters} clusters (including {int(np.sum(unique == -1))} noise gaussians)")
    if n_clusters > 0:
        cluster_summary = sorted([(int(u), int(c)) for u, c in zip(unique[valid], counts[valid])], key=lambda x: -x[1])
        print("Top clusters (id,count):", cluster_summary[:10])

    colors = assign_cluster_colors(labels)

    # Determine output path: allow `--out` to be either a directory or a file path.
    if out_dir is None:
        out_path = str(Path(ckpt_path).with_suffix("")) + "_clustered.pt"
    else:
        out_p = Path(out_dir)
        # If path exists and is a directory, write a file inside it.
        if out_p.exists() and out_p.is_dir():
            out_path = str(out_p / (Path(ckpt_path).stem + "_clustered.pt"))
        else:
            # If given path has a .pt suffix, treat it as the full file path (create parent dir).
            if out_p.suffix == ".pt":
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_path = str(out_p)
            else:
                # Treat as a directory path (may not exist yet) and create it.
                out_p.mkdir(parents=True, exist_ok=True)
                out_path = str(out_p / (Path(ckpt_path).stem + "_clustered.pt"))

    save_clustered_ckpt(ckpt_path, out_path, colors)

    preview_dir = Path(out_path).parent / "cluster_previews"
    print("Rendering preview images...")
    render_previews(splats, str(preview_dir))

    # Start an interactive GsplatViewer wired to the clustered splats
    try:
        import viser

        server = viser.ViserServer(port=8080, verbose=False)
        render_fn = make_viewer_render_fn(splats, device=device)
        viewer = GsplatViewer(server=server, render_fn=render_fn, output_dir=Path(preview_dir), mode="rendering")
        print("Viewer running at http://localhost:8080 — press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Viewer stopped by user.")
    except Exception as e:
        print("Interactive viewer unavailable:", e)

    print(f"Clustered checkpoint saved to: {out_path}")
    print(f"Preview images saved to: {preview_dir}")


if __name__ == "__main__":
    import argparse

    dense_dir = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/RobotLab/dense/ckpts'
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to 2DGS checkpoint (.pt)", default=os.path.join(dense_dir, 'ckpt_29999.pt'))
    p.add_argument("--out", help="Output clustered checkpoint path", default=dense_dir)
    args = p.parse_args()
    segment_feature_field(args.ckpt, args.out)
