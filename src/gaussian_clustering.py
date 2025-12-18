import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.utils.seg_utils import (
    cluster_point_cloud,
    compute_centroids,
)


class GaussianClusterer:
    def __init__(self, model_dir: str, result_dir: str, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.result_dir = Path(result_dir)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.cluster_dir = self.result_dir / "clustering"
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

    def load_gaussians_from_runner(self, runner) -> Dict[str, torch.Tensor]:
        splats = runner.splats

        means = splats["means"].detach()
        features = None
        # segmentation features used by some models
        if "seg_features" in splats:
            features = splats["seg_features"].detach()
        # fallback to features/sh coefficients
        elif "features" in splats:
            features = splats["features"].detach()
        elif "sh0" in splats:
            sh0 = splats["sh0"].detach()
            shN = splats.get("shN", sh0).detach()
            features = torch.cat([sh0, shN], dim=-1)

        opacities = torch.sigmoid(splats["opacities"]).detach() if "opacities" in splats else None
        scales = torch.exp(splats["scales"]).detach() if "scales" in splats else None

        if features is None:
            raise RuntimeError("Could not find segmentation/features in runner.splats.\n"
                               "Expected 'seg_features', 'features' or 'sh0' entries.")

        # Ensure features are 2D [N, D]
        if features.ndim > 2:
            N = features.shape[0]
            features = features.reshape(N, -1)
        elif features.ndim == 1:
            features = features.unsqueeze(-1)

        return {
            "means": means.to(self.device),
            "features": features.to(self.device),
            "opacities": opacities.to(self.device) if opacities is not None else None,
            "scales": scales.to(self.device) if scales is not None else None,
        }

    def cluster_gaussians(self, gaussians: Dict[str, torch.Tensor], k: int = 10, thresh_cos: float = 0.5,
                          min_cluster_size: int = 15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = gaussians["means"]
        features = gaussians["features"]

        # Ensure features are 2D and normalized
        if features.ndim != 2:
            N = features.shape[0]
            features = features.reshape(N, -1)
        
        # normalize features along feature dimension
        features = torch.nn.functional.normalize(features, dim=-1)

        labels, centroids_feat, centroids_xyz, id_unique_list = cluster_point_cloud(
            xyz=means, features=features, k=k, thresh_cos=thresh_cos, min_cluster_size=min_cluster_size
        )

        return labels, centroids_feat, centroids_xyz, id_unique_list

    def save_results(self, labels: torch.Tensor, centroids_features: torch.Tensor,
                     centroids_xyz: torch.Tensor, id_unique_list: torch.Tensor) -> Path:
        out = {
            "labels": labels.cpu().numpy(),
            "centroids_features": centroids_features.cpu().numpy(),
            "centroids_xyz": centroids_xyz.cpu().numpy(),
            "id_unique_list": id_unique_list.cpu().numpy(),
        }
        out_file = self.cluster_dir / "clustering_results.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(out, f)

        # brief text summary
        summary = self.cluster_dir / "clustering_summary.txt"
        with open(summary, "w") as f:
            f.write(f"Total gaussians: {out['labels'].shape[0]}\n")
            f.write(f"Num clusters: {len(out['id_unique_list'])}\n")
            f.write(f"Cluster ids: {out['id_unique_list'].tolist()}\n")

        return out_file


def cluster_gaussian_splats(runner, model_dir: str, result_dir: str, k: int = 10, thresh_cos: float = 0.8,
                            min_cluster_size: int = 15) -> Dict:
    clusterer = GaussianClusterer(model_dir, result_dir)
    gaussians = clusterer.load_gaussians_from_runner(runner)
    labels, centroids_feat, centroids_xyz, id_unique_list = clusterer.cluster_gaussians(
        gaussians, k=k, thresh_cos=thresh_cos, min_cluster_size=min_cluster_size
    )
    out_file = clusterer.save_results(labels, centroids_feat, centroids_xyz, id_unique_list)

    return {
        "labels": labels,
        "centroids_features": centroids_feat,
        "centroids_xyz": centroids_xyz,
        "unique_ids": id_unique_list,
        "saved_to": str(out_file),
    }
