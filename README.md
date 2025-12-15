# Installation

## Installing COLMAP

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

# Issues

- Need to find a way to safely install COLMAP. Currently on some machine the automated call to colmap works, in other tosses errors realted to lacking solvers ('Can't use SPARSE_SCHUR sparse_linear_algebra_library_type = SUITE_SPARSE, because support was not enabled when Ceres Solver was built.') or lack of a channel for rendering the visual. 