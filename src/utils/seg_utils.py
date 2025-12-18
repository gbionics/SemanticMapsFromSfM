import random
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
from PIL import Image
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin
from typing import NamedTuple, Dict, Optional
import open3d as o3d
from gsplat.rendering import rasterization

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, 
           backpropagate_seg=False, mask3D=None, seg_features=None) -> Dict:
    """
    Render the scene using gsplat rasterization.
    
    This function replaces the gaussian_renderer.render() function to use gsplat library.
    Returns a dictionary with rendering outputs including feature maps for segmentation.
    
    Args:
        viewpoint_camera: Camera object with intrinsics and extrinsics
        pc: Gaussian model with splats
        pipe: Pipeline configuration
        bg_color: Background color
        scaling_modifier: Scaling modifier for splats
        override_color: Optional override for colors
        backpropagate_seg: Whether to backpropagate segmentation
        mask3D: Optional 3D mask for selective rendering
        seg_features: Optional segmentation features
    
    Returns:
        Dictionary with 'render' and 'render_feature_map' keys
    """
    means3D = pc.get_xyz  # [N, 3]
    quats = pc.get_rotation  # [N, 4]
    scales = torch.exp(pc.get_scaling)  # [N, 3]
    opacities = torch.sigmoid(pc.get_opacity)  # [N,]
    
    # Get colors from SH coefficients or precomputed
    if override_color is None:
        shs = pc.get_features  # [N, 3, (max_sh_degree+1)^2]
        from src.utils.sh_utils import eval_sh
        shs_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(colors + 0.5, 0.0)
    else:
        colors = override_color
    
    # Handle segmentation features
    if seg_features is None:
        seg_features = pc.get_seg_features if hasattr(pc, 'get_seg_features') else None
    
    # Apply mask if provided
    if mask3D is not None:
        means3D = means3D[mask3D]
        quats = quats[mask3D]
        scales = scales[mask3D]
        opacities = opacities[mask3D]
        colors = colors[mask3D]
        if seg_features is not None:
            seg_features = seg_features[mask3D]
    
    # Build camera matrices
    import math
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # Convert to gsplat format
    viewmat = viewpoint_camera.world_view_transform  # [4, 4]
    K = torch.tensor([
        [viewpoint_camera.image_width / (2 * tanfovx), 0, viewpoint_camera.image_width / 2],
        [0, viewpoint_camera.image_height / (2 * tanfovy), viewpoint_camera.image_height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=means3D.device)
    
    # Rasterize with gsplat
    renders, radii, info = rasterization(
        means=means3D.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors.unsqueeze(0),
        viewmats=torch.linalg.inv(viewpoint_camera.world_view_transform).unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
    )
    
    # Render segmentation features if provided
    rendered_features = None
    if seg_features is not None:
        # Normalize segmentation features and render
        seg_features_normalized = torch.tanh(seg_features)
        renders_seg, _, _ = rasterization(
            means=means3D.unsqueeze(0),
            quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0),
            opacities=opacities.unsqueeze(0),
            colors=seg_features_normalized.unsqueeze(0),
            viewmats=torch.linalg.inv(viewpoint_camera.world_view_transform).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
        )
        rendered_features = renders_seg[0].permute(2, 0, 1)
    
    return {
        "render": renders[0].permute(2, 0, 1),
        "render_feature_map": rendered_features if rendered_features is not None else renders[0, ..., :3].permute(2, 0, 1),
        "viewspace_points": means3D,
        "visibility_filter": radii[0] > 0,
        "radii": radii[0],
    }


def render_mask(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, 
                override_color=None, mask=None) -> Dict:
    """
    Render mask for a specific set of splats using gsplat rasterization.
    
    This function replaces the gaussian_renderer.render_mask() function to use gsplat library.
    
    Args:
        viewpoint_camera: Camera object with intrinsics and extrinsics
        pc: Gaussian model with splats
        pipe: Pipeline configuration
        bg_color: Background color
        scaling_modifier: Scaling modifier for splats
        override_color: Optional override for colors
        mask: 3D mask tensor indicating which splats to render
    
    Returns:
        Dictionary with 'render' and 'render_mask' keys
    """
    means3D = pc.get_xyz  # [N, 3]
    quats = pc.get_rotation  # [N, 4]
    scales = torch.exp(pc.get_scaling)  # [N, 3]
    opacities = torch.sigmoid(pc.get_opacity)  # [N,]
    
    # Get colors from SH coefficients or precomputed
    if override_color is None:
        shs = pc.get_features  # [N, 3, (max_sh_degree+1)^2]
        from src.utils.sh_utils import eval_sh
        shs_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(colors + 0.5, 0.0)
    else:
        colors = override_color
    
    # Prepare mask as colors for rendering
    if mask is not None:
        mask_colors = mask.float().unsqueeze(-1).expand(-1, 3)  # Convert to RGB
    else:
        mask_colors = colors
    
    # Build camera matrices
    import math
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    K = torch.tensor([
        [viewpoint_camera.image_width / (2 * tanfovx), 0, viewpoint_camera.image_width / 2],
        [0, viewpoint_camera.image_height / (2 * tanfovy), viewpoint_camera.image_height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=means3D.device)
    
    # Rasterize mask with gsplat
    renders, radii, info = rasterization(
        means=means3D.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=mask_colors.unsqueeze(0),
        viewmats=torch.linalg.inv(viewpoint_camera.world_view_transform).unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
    )
    
    # Extract mask (first channel)
    rendered_mask = renders[0, ..., 0:1].permute(2, 0, 1)
    
    return {
        "render": renders[0].permute(2, 0, 1),
        "render_mask": rendered_mask,
    }

def downsample_voxel_grid(point_cloud: BasicPointCloud, voxel_size: float) -> BasicPointCloud:
    """Downsamples the point cloud using voxel grid filtering."""
    # Convert BasicPointCloud to Open3D format
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
    o3d_cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors)
    o3d_cloud.normals = o3d.utility.Vector3dVector(point_cloud.normals)
    
    # Apply voxel downsampling
    downsampled_cloud = o3d_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to BasicPointCloud
    return BasicPointCloud(points=np.asarray(downsampled_cloud.points),
                           colors=np.asarray(downsampled_cloud.colors),
                           normals=np.asarray(downsampled_cloud.normals))

def get_random_colors(num_colors):
    '''
    Generate random colors for visualization
    
    Args:
        num_colors (int): number of colors to generate
        
    Returns:
        colors (np.ndarray): (num_colors, 3) array of colors, in RGB, [0, 1]
    '''
    colors = []
    colors.append(np.asarray([0, 0, 0]))
    for _ in range(num_colors):
        colors.append(np.random.rand(3))
    colors = np.array(colors)
    return colors

def get_objmask(feature_map, feature_prompt, thresh, min_pixnum=400, filter=True):
    img_shape = feature_map.shape[1:]
    features_reshaped = feature_map.view(feature_map.shape[0], -1).T
    logits = torch.sum((features_reshaped - feature_prompt[None, ...]) ** 2, dim=-1)
    logits = logits.reshape(img_shape)
    segmask = logits < thresh
    if torch.sum(segmask) < min_pixnum:
        segmask[...] = False
    elif filter:
        kernel = np.ones((4, 4), np.uint8)
        import cv2
        segmask = cv2.erode((segmask.cpu().numpy().astype(np.uint8)), kernel, iterations=1)
        segmask = cv2.dilate(segmask, kernel, iterations=1)
        segmask = torch.from_numpy(segmask)
    return segmask == 1, logits

def get_segmentation_map(features, centroids, metric="cosine", thr=0.85):
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T
    if metric == "cosine":
        (max, max_indices) = torch.max(torch.matmul(features_reshaped, centroids.T), dim=1)
        max_indices += 1
        max_indices[max < thr] = 0
        pred_seg = max_indices.reshape(H, W)
    elif metric == "euclidean":
        eu_dist = torch.cdist(features_reshaped, centroids) #10*torch.cdist(features_reshaped[:, :3], centroids[:, :3]) + torch.cdist(features_reshaped[:, 3:], centroids[:, 3:])
        (min, min_indices) = torch.min(eu_dist, dim=1)
        min_indices += 1
        min_indices[min > thr] = 0
        pred_seg = min_indices.reshape(H, W)
    else:
        pred_seg = torch.zeros((H, W))
        for id, c in enumerate(centroids):
            mask, _ = get_objmask(features, c, thr)
            pred_seg[mask] = id+1
    return pred_seg

def compute_init_segmentation(features, min_cluster_size=15, min_samples=30, device="cuda"):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=-1)
    hdb.fit(features)
    return torch.Tensor(hdb.labels_).to(device)

def cluster_scene(features, centroids, labels):
    n = features.shape[0] - 1
    batch_size = 64
    n_batches = n // batch_size + 1
    min_indices = torch.zeros(features.shape[0], dtype=torch.int)
    min_vals = torch.zeros(features.shape[0])

    for i in range(n_batches):
        j = i * batch_size
        k = min(n, (i + 1) * batch_size)
        dist = 10.0 * torch.cdist(features[j:k, :3], centroids[:, :3]) + torch.cdist(features[j:k, 3:], centroids[:, 3:])
        (vals, indices) = torch.min(dist, dim=1)
        min_indices[j:k] = indices[:]
        min_vals[j:k] = vals[:]
    
    res = labels[min_indices]
    return res

def compute_id_area_per_view(views, gaussians, pipeline, background, centroids):
    id_area_per_view = {}
    segmap_list = {}
    
    for idx, view in enumerate(views):
        rendering_pkg = render(view, gaussians, pipeline, background)
        
        rendering_features = rendering_pkg["render_feature_map"]

        pred_segmap = get_segmentation_map(rendering_features, centroids, metric="cosine", thr=-0.25)
        segmap_list[idx] = pred_segmap

        id_unique_list = torch.unique(pred_segmap)[1:]

        area_per_id = [torch.sum(pred_segmap == id) for id in id_unique_list]
        for k, id in enumerate(id_unique_list):
            if id.item() not in id_area_per_view.keys():
                id_area_per_view[id.item()] = {}
            id_area_per_view[id.item()][idx] = area_per_id[k].item()
    
    # sort from biggest to smallest area
    for id in id_area_per_view:
        id_dict = id_area_per_view[id]
        id_area_per_view[id] = {k: v for k, v in sorted(id_dict.items(), key=lambda item: item[1], reverse=True)}
    return id_area_per_view, segmap_list

def extract_image_segment(img, mask):
    img_numpy = np.array(img)
    segment_numpy = np.zeros_like(img_numpy)
    
    # extract segment of interest
    segment_numpy[mask] = img_numpy[mask]
    segment = Image.fromarray(segment_numpy)
    
    transparency_mask_numpy = np.zeros_like(mask, dtype=np.uint8)
    transparency_mask_numpy[mask] = 255
    transparency_mask = Image.fromarray(transparency_mask_numpy, mode='L')
    res = Image.new("RGB", img.size, (0, 0, 0))
    
    # segmented image with a transparent background
    res.paste(segment, mask=transparency_mask)
    return res

def get_csr_matrix(similarity, neighbor_indices_tensor, min_cos_value):
    indices = []
    data = []
    
    for i in range(similarity.shape[0]):
        cos_per_neighbor = similarity[i]
        neighbor_indices = neighbor_indices_tensor[i]
        for j, val in enumerate(cos_per_neighbor):
            if val >= min_cos_value:
                k = neighbor_indices[j]
                indices.append([i, k])
                data.append(val)
    
    return np.asarray(indices), np.asarray(data)

def refine_segmentation(classes, points, features, id_unique_list, n_per_id, min_cluster_size=30):
    for n_p, id in zip(n_per_id, id_unique_list):
        if n_p < min_cluster_size:
            classes[classes == id] = -1
    p_lab = points[classes != -1]
    p_unlab = points[classes == -1]
    
    # Guard: if no labeled points remain, return as-is
    if p_lab.shape[0] == 0 or p_unlab.shape[0] == 0:
        print(f"Skipping refinement (labeled={p_lab.shape[0]}, unlabeled={p_unlab.shape[0]})")
        return classes
    
    batch_size = 128
    n_batches = max(1, p_unlab.shape[0] // batch_size + (1 if p_unlab.shape[0] % batch_size != 0 else 0))
    neig_list = []
    for i in range(n_batches):
        j = i * batch_size
        k = min((i + 1) * batch_size, p_unlab.shape[0])
        dists = torch.cdist(p_unlab[j:k, :], p_lab)
        # topk returns (values, indices) with shapes [batch, 6]
        _, neighbor_indices = dists.topk(min(6, p_lab.shape[0]), largest=False, dim=1)
        neig_list.append(neighbor_indices)
    neighbor_indices_tensor = torch.cat(neig_list, dim=0)  # [num_unlab, 6]
    
    # Gather features for labeled points and their neighbors
    labeled_features = features[classes != -1]  # [num_lab, D]
    neighbor_preds = labeled_features[neighbor_indices_tensor]  # [num_unlab, 6, D]
    
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    unlab_preds = features[classes == -1]  # [num_unlab, D]
    # Expand unlab_preds to [num_unlab, 6, D] and compute similarity
    data = cos(unlab_preds.unsqueeze(1).expand(-1, neighbor_indices_tensor.shape[1], -1), neighbor_preds)
    
    id_k_neig = torch.argmin(data, dim=1).cpu()  # [num_unlab]
    id_p_lab = torch.zeros_like(id_k_neig)
    for i, k in enumerate(id_k_neig):
        id_p_lab[i] = neighbor_indices_tensor[i, k].cpu()
    
    aux = labeled_features[id_p_lab]  # Get cluster IDs from labeled set
    # Get actual class labels
    labeled_classes = classes[classes != -1][id_p_lab]
    classes[classes == -1] = labeled_classes
    return classes

'''def refine_segmentation(labels, xyz, id_unique_list, n_per_id, min_cluster_size=30):
    for i, id in enumerate(id_unique_list):
        n_p = n_per_id[i]
        if n_p < min_cluster_size:
            labels[labels == id] = -1
    p_lab = xyz[labels != -1]
    p_unlab = xyz[labels == -1]
    indices = pairwise_distances_argmin(p_unlab, p_lab)
    new_labels = labels[labels != -1][indices]
    labels[labels == -1] = new_labels
    return labels'''

def cluster_segments(xyz, features, classes, k=5, thresh_cos=0.7):
    xyz = xyz.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(xyz)
    _, neighbor_indices_tensor = nbrs.kneighbors(xyz)
    neighbor_preds = features[torch.from_numpy(neighbor_indices_tensor).cuda()]
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    similarity = cos(features.unsqueeze(1).expand(-1, k, -1, -1), neighbor_preds)
    indices, values = get_csr_matrix(similarity.cpu().numpy(), neighbor_indices_tensor, min_cos_value=thresh_cos)
    n_p = xyz.shape[0]
    graph = sp.sparse.csr_matrix((values.squeeze(), (indices[:, 0], indices[:, 1])), shape=(n_p, n_p))
    _, labels = sp.sparse.csgraph.connected_components(graph, directed=False)
    id_unique_list = torch.unique(classes)
    res = torch.zeros_like(classes).long()
    for i, id in enumerate(id_unique_list):
        res[classes == id] = labels[i]
    return res

def get_unique_id_list(labels):
    id_unique_list, n_per_id = np.unique(labels, return_counts=True)
    if id_unique_list[0] == -1:
        id_unique_list = id_unique_list[1:]
        n_per_id = n_per_id[1:]
    return id_unique_list, n_per_id

def compute_centroids(features, labels, id_unique_list):
    dim_features = features.shape[-1]
    n = id_unique_list.shape[0]
    f_mean_per_cluster = torch.zeros((n, dim_features)).cuda()
    for i in range(n):
        mask = labels == id_unique_list[i]
        f_mean_per_cluster[i, ...] = torch.mean(features[mask, :], dim=0, keepdim=True)
    return f_mean_per_cluster


def cluster_point_cloud(xyz, features, k=10, thresh_cos=0.5, min_cluster_size=15):
    xyz_numpy = xyz.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(xyz_numpy)
    _, neighbor_indices_tensor = nbrs.kneighbors(xyz_numpy)
    
    # Convert indices to torch tensor on the correct device
    neighbor_indices_torch = torch.from_numpy(neighbor_indices_tensor).long().to(features.device)
    
    # Properly gather neighbor features: [N, k] indices -> [N, k, D]
    neighbor_preds = features[neighbor_indices_torch]  # [N, k, D]
    
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    similarity = cos(features.unsqueeze(1).expand(-1, k, -1), neighbor_preds)
    
    indices, values = get_csr_matrix(similarity.cpu().numpy(), neighbor_indices_tensor, min_cos_value=thresh_cos)
    
    n_p = xyz_numpy.shape[0]
    graph = sp.sparse.csr_matrix((values.squeeze(), (indices[:, 0], indices[:, 1])), shape=(n_p, n_p))
    _, labels = sp.sparse.csgraph.connected_components(graph, directed=False)
    
    id_unique_list, n_per_id = get_unique_id_list(labels)
    print(f"Clustering: {len(id_unique_list)} clusters found (thresh_cos={thresh_cos})")
    
    # Skip refine_segmentation if no valid clusters found
    if len(id_unique_list) == 0:
        print("No valid clusters found! Returning empty results.")
        labels = torch.from_numpy(labels).to(xyz.device)
        id_unique_list = torch.tensor([], dtype=torch.long, device=xyz.device)
        f_mean_per_cluster = torch.zeros((0, features.shape[1]), device=xyz.device)
        f_mean_per_cluster_xyz = torch.zeros((0, 3), device=xyz.device)
        return labels, f_mean_per_cluster, f_mean_per_cluster_xyz, id_unique_list
    
    labels = refine_segmentation(labels, xyz, features, id_unique_list, n_per_id, min_cluster_size=min_cluster_size)
    labels = torch.from_numpy(labels).to(xyz.device)
    id_unique_list = torch.unique(labels)
    if id_unique_list[0] == -1:
        id_unique_list = id_unique_list[1:].to(xyz.device)
    f_mean_per_cluster = compute_centroids(features, labels, id_unique_list)
    f_mean_per_cluster_xyz = compute_centroids(xyz, labels, id_unique_list)
    '''labels = cluster_segments(f_mean_per_cluster_xyz, f_mean_per_cluster.unsqueeze(1), labels, k=11, thresh_cos=0.815)
    id_unique_list, n_per_id = torch.unique(labels, return_counts=True)
    if id_unique_list[0] == -1:
        id_unique_list = id_unique_list[1:].cuda()
        n_per_id = n_per_id[1:].cuda()
    f_mean_per_cluster = compute_centroids(features, labels, id_unique_list)
    f_mean_per_cluster_xyz = compute_centroids(xyz, labels, id_unique_list)'''
    return labels, f_mean_per_cluster, f_mean_per_cluster_xyz, id_unique_list

def compute_features_per_rendering(labels, features, centroids):
    unique_list = torch.unique(labels)
    unique_list = unique_list[unique_list != -1]
    for i, id in enumerate(unique_list):
        features[labels == id, ...] = centroids[i, ...]
    return features

def compute_area_per_view(views, gaussians, pipeline, background, labels, id_unique_list):
    id_area_per_view = {}
    masks_per_id = {}
    mask3d = torch.zeros_like(gaussians._opacity, device="cuda")
    for id in id_unique_list:
        mask3d[...] = 0.0
        mask3d[labels == id] = 1.0
        id_area_per_view[id.item()] = {}
        masks_per_id[id.item()] = {}
        for j, view in enumerate(views):
            rendering_pkg = render_mask(view, gaussians, pipeline, background, mask=mask3d)
            logits = rendering_pkg["render_mask"].permute(1, 2, 0).squeeze()
            binary = logits > 0.4
            area = torch.count_nonzero(binary).item()
            if area > 1500:
                id_area_per_view[id.item()][j] = area
                masks_per_id[id.item()][j] = binary.cpu().numpy()
                '''import matplotlib.pyplot as plt
                plt.imshow(logits.cpu().numpy())
                plt.show()'''
        print(id)
    # sort from biggest to smallest area
    for id in id_area_per_view:
        id_dict = id_area_per_view[id]
        id_area_per_view[id] = {k: v for k, v in sorted(id_dict.items(), key=lambda item: item[1], reverse=True)}
    return id_area_per_view, masks_per_id