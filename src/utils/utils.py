import random
from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def initialize_seg_features(xyz: np.ndarray, seg_features_dim: int) -> np.ndarray:
    """Initialize segmentation features for Gaussians.
    
    Args:
        xyz: Point cloud positions of shape (N, 3)
        seg_features_dim: Dimension of segmentation features
        
    Returns:
        Segmentation features of shape (N, seg_features_dim)
    """
    res = np.random.randn(xyz.shape[0], seg_features_dim).astype(np.float32)
    return res


def hierarchical_contrastive_loss(
    segmask: torch.Tensor,
    features: torch.Tensor,
    id_unique_list: torch.Tensor,
    n_i_list: torch.Tensor,
    adj_mtx: Optional[torch.Tensor] = None,
    lap_mtx: Optional[torch.Tensor] = None,
    dim_features: int = 16,
    lambda_1_val: float = 1e-4,
    lambda_2_val: float = 1e-2,
    epsilon: float = 100,
) -> torch.Tensor:
    """Compute hierarchical contrastive clustering loss for segmentation features.
    
    This implements the same loss as in train_lerf.py, optionally using adjacency and 
    Laplacian matrices for graph-based regularization.
    
    Args:
        segmask: Tensor of shape (H, W) with segment IDs
        features: Tensor of shape (H, W, D) with feature vectors
        id_unique_list: Tensor of shape (n_p,) with unique segment IDs
        n_i_list: Tensor of shape (n_p,) with number of pixels per segment
        adj_mtx: Adjacency matrix of shape (n_p, n_p) for graph loss (optional)
        lap_mtx: Laplacian matrix of shape (n_p, n_p) for graph loss (optional)
        dim_features: Dimensionality of features
        lambda_1_val: Weight for clustering loss
        lambda_2_val: Weight for graph loss
        epsilon: Epsilon parameter for smoothing
        
    Returns:
        Total loss value
    """
    device = features.device
    n_p = id_unique_list.shape[0]  # Number of unique segments
    
    if n_p <= 1:
        return torch.tensor(0.0, device=device, dtype=features.dtype)
    
    # Compute mean features per cluster
    f_mean_per_cluster = torch.zeros((n_p, dim_features), device=device)
    phi_per_cluster = torch.zeros((n_p, 1), device=device)
    
    for i in range(n_p):
        mask = segmask == id_unique_list[i]
        if mask.sum() > 0:
            f_mean_per_cluster[i] = torch.mean(features[mask], dim=0, keepdim=True)
            phi_per_cluster[i] = (
                torch.norm(features[mask] - f_mean_per_cluster[i], dim=1, keepdim=True).sum()
                / (n_i_list[i] * torch.log(n_i_list[i] + epsilon))
            )
    
    # Clip phi values
    phi_per_cluster = torch.clamp(phi_per_cluster * 10, min=0.1, max=1.0)
    phi_per_cluster = phi_per_cluster.detach()
    
    # Compute contrastive loss per cluster (clustering component)
    loss_per_cluster = torch.zeros(n_p, device=device)
    
    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        phi = phi_per_cluster[i]
        mask = segmask == id_unique_list[i]
        f_ij = features[mask]  # shape (ni, dim_features)
        
        if f_ij.shape[0] > 0:
            # Numerator: similarity with own cluster mean
            num = torch.exp(torch.matmul(f_ij, f_mean.squeeze(-1)) / (phi + 1e-6))
            
            # Denominator: sum of similarities with all cluster means
            all_means = f_mean_per_cluster.t()  # [dim_features, n_p]
            all_phis = phi_per_cluster.t()  # [1, n_p]
            den = torch.sum(
                torch.exp(torch.matmul(f_ij, all_means) / (all_phis + 1e-6)),
                dim=1,
            )
            
            loss_per_cluster[i] = torch.sum(-torch.log(num / (den + 1e-6)))
    
    # Clustering loss component
    clustering_loss = lambda_1_val * torch.mean(loss_per_cluster)
    
    # Graph-based loss component (if adjacency and Laplacian matrices provided)
    if adj_mtx is not None and lap_mtx is not None:
        graph_loss = _graph_contrastive_loss(
            f_mean_per_cluster, phi_per_cluster, adj_mtx, lap_mtx, lambda_val=lambda_2_val
        )
        return clustering_loss + graph_loss
    else:
        # Return just clustering loss, weighted by lambda_2 as well
        return clustering_loss + lambda_2_val * torch.mean(loss_per_cluster)


def _graph_contrastive_loss(
    features: torch.Tensor,
    phi: torch.Tensor,
    A: torch.Tensor,
    L: torch.Tensor,
    lambda_val: float = 1e-2,
) -> torch.Tensor:
    """Compute graph-based contrastive loss using adjacency and Laplacian matrices.
    
    Args:
        features: Cluster mean features of shape (n_p, dim_features)
        phi: Temperature parameters of shape (n_p, 1)
        A: Adjacency matrix of shape (n_p, n_p)
        L: Laplacian matrix of shape (n_p, n_p)
        lambda_val: Weight for the graph loss
        
    Returns:
        Graph loss value
    """
    sample_size = A.shape[0]
    
    # Compute similarity matrix
    S = torch.exp(torch.matmul(features, features.transpose(-1, -2)))
    loss_per_sample = torch.zeros(sample_size, device=features.device)
    
    for i in range(sample_size):
        # Connected neighbors (edges in graph)
        B_intra = L[i, :] < 0
        if torch.count_nonzero(B_intra) < 1:
            continue
        
        t = phi[i]
        S_i = S[i] / (t + 1e-6)
        
        # Loss for connected components (should be similar)
        S_intra = torch.sum(-L[i][B_intra] * S_i[B_intra])
        
        # Loss for non-connected components (should be dissimilar)
        B_inter = L[i, :] == 0
        S_inter = torch.sum(S_i[B_inter])
        
        loss_per_sample[i] = torch.log(S_intra / (S_inter + 1e-6))
    
    return lambda_val * torch.mean(loss_per_sample)
