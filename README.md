**Project Overview**
- **What:** A gsplat-based 2D Gaussian Splatting training and rendering toolkit extended with an optional segmentation feature field and PCA-based feature visualization.
- **Primary scripts:** File: [src/splatting_2DGS.py](src/splatting_2DGS.py) implements training, evaluation and trajectory rendering. Supporting utilities live under File: [src/utils](src/utils).

**Requirements**
- **OS:** Linux (development tested).
- **Python:** 3.9+ (match your conda env). Recommended: create a dedicated env (example below).
- **Core packages:** `torch` (CUDA build matching your GPU), `imageio`, `tyro`, `tqdm`, `viser`, `gsplat` (CUDA extension). See the project code for the exact imports in File: [src/splatting_2DGS.py](src/splatting_2DGS.py).

**Quick setup**
- Create a conda env and install dependencies (example):

  `conda create -n splatting python=3.9 -y`
  `conda activate splatting`
  `pip install -r requirements.txt`  # if you maintain one; otherwise install packages listed in Requirements

- Install/compile `gsplat` CUDA extension according to its README; ensure the compiled package is installed into the env used to run scripts.

**Data layout**
- The repo expects a COLMAP-style dataset layout under a data folder. Default dataset path is set in the `Config` of File: [src/splatting_2DGS.py](src/splatting_2DGS.py).
- Optional segmentation masks (used when `--seg-opt` is enabled) should be placed in a single `masks_dir` with filenames matching COLMAP image names and these conventions (train_lerf/train_scannet format):
  - `{image_name}.npy` : segmentation mask (H, W) of integer labels
  - `{image_name}_am.npy` : adjacency matrix (optional)
  - `{image_name}_lm.npy` : Laplacian matrix (optional)
  - `{image_name}_maps.npy` : hierarchical level maps (optional)

**Running training**
- Example (defaults live in File: [src/splatting_2DGS.py](src/splatting_2DGS.py) `Config`):

  `python src/splatting_2DGS.py --seg-opt --seg-features-dim=16 --contrastive-lambda-1 1e-4 --contrastive-lambda-2 1e-2`

- Important config knobs are in the `Config` dataclass at the top of File: [src/splatting_2DGS.py](src/splatting_2DGS.py). You can override them on the CLI (the script uses `tyro` for CLI parsing).

**Evaluation & Rendering**
- To evaluate a checkpoint and render a trajectory use `--ckpt` pointing to a saved checkpoint in `{result_dir}/ckpts`.
- The trajectory renderer writes a video to `{result_dir}/videos/traj_<step>.mp4` and the evaluation images to `{result_dir}/renders/`.

**Segmentation feature field**
- When `--seg-opt` is enabled the code creates a per-Gaussian learnable `seg_features` field and renders it alongside RGB/depth. Training uses a hierarchical contrastive loss similar to train_lerf/train_scannet when mask auxiliary matrices are available.
- Data format expectations: see the Data layout section above and the dataset loader in File: [src/utils/parser.py](src/utils/parser.py).

**Visualization**
- Trajectory videos include three panels side-by-side: rendered color, depth, and PCA-based coloring of the segmentation feature field (if available). PCA coloring projects per-pixel D-dimensional features to RGB using the top-3 principal components.

**Configuration tips**
- If your dataset images were downsampled differently from COLMAP, adjust `--data_factor` or use the `Parser` logic in File: [src/utils/parser.py](src/utils/parser.py).
- To disable viewer and run headless use `--disable-viewer`.

**Troubleshooting**
- Warning about `TORCH_CUDA_ARCH_LIST`:
  - Message: `TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.`
  - Fix (optional): set `export TORCH_CUDA_ARCH_LIST="<your-archs>"` before building CUDA extensions to restrict architectures.
- Rasterizer shape assertion failures:
  - If you encounter assertions from the `gsplat` rasterizer about `colors` shapes, ensure the `colors` argument passed to `rasterization_2dgs` matches the expected layout in `gsplat.rendering.rasterization_2dgs`:
    - When `sh_degree` is set, `colors` should be SH coefficients with shape `[..., N, K, 3]` where `K >= (sh_degree+1)**2`.
    - When `sh_degree` is `None`, `colors` should be post-activation with shape `[..., N, D]` (e.g., `[..., N, 3]` for RGB)
  - The repository includes fixes that render segmentation features as multi-channel post-activation colors to avoid incompatible SH coefficient shapes. If you modify that code, follow the above shape rules.

**Development notes**
- Key files:
  - File: [src/splatting_2DGS.py](src/splatting_2DGS.py) — main training/eval/render script
  - File: [src/utils/parser.py](src/utils/parser.py) — COLMAP parser and dataset loader (includes mask loading)
  - File: [src/utils/utils.py](src/utils/utils.py) — helper utilities, segmentation feature initialization and contrastive loss
  - Folder: [2DGSeg-main](2DGSeg-main) — external components used for segmentation/loss references

**Contributing**
- For new features or bug fixes: open a branch, include a focused unit test or a short repro, and submit a PR with a clear description.

**License**
- See `LICENSE` files in subfolders for component licenses (e.g., 2DGSeg submodule). This repository does not add a global license file by default.

If you want, I can:
- Add a `requirements.txt` listing the precise package versions used for testing.
- Add quick-run scripts (`scripts/train.sh`, `scripts/eval.sh`) that wrap the common CLI invocations.

## Setting up the CONDA environment
'''
conda create -n splatting python=3.11
conda activate splatting
pip install torch torchvision
pip install open3d trimesh scikit-image opencv-python plyfile tqdm
pip install gsplat imageio tyro viser pyyaml opencv-python pycolmap torchmetrics tensorboard scikit-learn matplotlib nerfview splines 
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
pip install imageio-ffmpeg
pip install sam2 segment-anything
'''
