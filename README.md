# Installation

## Installing COLMAP

## Setting up the CONDA environment
'''
conda create -n gsplat_modelling python=3.11 -y

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

conda install -c conda-forge \                                                                           
    numpy \
    scipy \
    opencv \
    scikit-learn \
    scikit-image \
    matplotlib \
    imageio \
    imageio-ffmpeg \
    tqdm \
    tensorboard \
    pyyaml \
    plyfile \
    -y
conda install -c conda-forge numpy scipy opencv scikit-learn scikit-image matplotlib imageio imageio-ffmpeg tqdm tensorboard pyyaml plyfile    -y

pip install \                                                                                            
    tyro \
    viser \
    gsplat \
    nerfview \
    torchmetrics \
    trimesh \
    open3d \
    splines \
    pycolmap \
    segment-anything \
    sam2

pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
'''

Old version:
conda create -n splatting python=3.11
conda activate splatting
pip install torch torchvision
pip install open3d trimesh scikit-image opencv-python plyfile tqdm
pip install gsplat imageio tyro viser pyyaml opencv-python pycolmap torchmetrics tensorboard scikit-learn matplotlib nerfview splines 
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
pip install imageio-ffmpeg
pip install sam2 segment-anything

# Viewing clustered Gaussian splats

After training and clustering, a clustered results file is written to `<result_dir>/clustering/clustering_results.pkl` and a clustered checkpoint can be produced from your checkpoint. You can visualize clustered splats in two ways:

- From the pipeline (`master.py`) — the script will attempt to locate the latest checkpoint under `<result_dir>/ckpts` and automatically launch an interactive viewer after clustering completes.

- Manually using the viewer helper:

```bash
python src/view_clustered_splats.py /path/to/ckpt_29999.pt /path/to/dense/clustering/clustering_results.pkl --viewer
```

This will save a clustered checkpoint (`*_clustered.pt`) next to the original checkpoint, render preview images into a `cluster_previews` folder, and (with `--viewer`) launch the `GsplatViewer` at http://localhost:8080.

If you prefer only previews without the web viewer, omit `--viewer`.

Note: the interactive viewer requires `viser` and other rendering dependencies available in your environment.

# Issues

- Need to find a way to safely install COLMAP. Currently on some machine the automated call to colmap works, in other tosses errors realted to lacking solvers ('Can't use SPARSE_SCHUR sparse_linear_algebra_library_type = SUITE_SPARSE, because support was not enabled when Ceres Solver was built.') or lack of a channel for rendering the visual. 