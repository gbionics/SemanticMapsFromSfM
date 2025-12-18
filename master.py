import os
import tyro
from src.preprocessing import resize_images, generate_segmentation_masks, prepare_dirs, visualize_masks
from src.reconstruct_with_COLMAP import run_colmap
from src.splatting_2DGS import Config, Runner
from src.splatting_3DGS import Config as Config3DGS

root_dir = os.getcwd()
path_to_raw_images = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/RobotLab/Images_raw'
model_name = 'RobotLab'

preprocess = False
image_size_max = 640

generate_sfm = False

generate_nvs = False

# Splatting method: "2DGS" or "3DGS"
splatting_method = "2DGS"

# Main file for the reconstruction and rendering automated pipeline

# 1) Preprocessing images and segmentation masks
model_dir = os.path.join(root_dir, 'Dataset', model_name)
    
if preprocess:
    images_dir, sparse_dir, masks_dir, visual_dir, dense_dir = prepare_dirs(model_dir)      
    resize_images(path_to_raw_images, images_dir, image_size_max)
    generate_segmentation_masks(images_dir, masks_dir, n_levels=1, sam2=False)
    visualize_masks(masks_dir, visual_dir, images_dir, sample=50)
else:
    images_dir = os.path.join(model_dir, 'images')
    sparse_dir = os.path.join(model_dir, 'sparse')
    dense_dir = os.path.join(model_dir, 'dense')
    masks_dir = os.path.join(model_dir, 'masks')
    visual_dir = os.path.join(model_dir, 'visualize')
        
# 2) Generate the COLMAP SfM model
if generate_sfm:  
    print('Generating the SfM reconstruction via Colmap')
    log_file_name = os.path.join(model_dir, 'colmap_reconstruction_log.txt')
    with open(log_file_name, 'w',
          buffering=1) as log_file:  
        run_colmap(images_dir, model_dir, log_file)

# 3) Initialize the splatting reconstruction

if splatting_method == "2DGS":
    from src.splatting_2DGS import Config, Runner
elif splatting_method == "3DGS":
    from src.splatting_3DGS import Config, Runner
else:
    raise ValueError(f"Unknown splatting method: {splatting_method}. Please choose '2DGS' or '3DGS'.")

cfg = tyro.cli(Config)
cfg.adjust_steps(cfg.steps_scaler)

cfg.data_dir = model_dir
cfg.result_dir = dense_dir
cfg.semantic_only_start = 20000

runner = Runner(cfg)

if generate_nvs:
    runner.train()
    
# 4) Clustering the Gaussian splats (feature-field clustering / 3D segmentation)
try:
    from src.gaussian_clustering import cluster_gaussian_splats

    print("Running Gaussian clustering on trained model...")
    clustering_results = cluster_gaussian_splats(
        runner=runner,
        model_dir=model_dir,
        result_dir=dense_dir,
        k=10,
        thresh_cos=0.8,
        min_cluster_size=15,
    )
    print(f"Clustering complete. Found {len(clustering_results['unique_ids'])} clusters. Results saved to {clustering_results['saved_to']}")
except Exception as e:
    print(f"Clustering failed: {e}")
    raise

# Attempt to automatically visualize clustering results by finding the latest checkpoint
try:
    ckpt_dir = os.path.join(dense_dir, 'ckpts')
    latest_ckpt = None
    if os.path.isdir(ckpt_dir):
        cands = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
        if len(cands) > 0:
            latest_ckpt = max(cands, key=os.path.getmtime)

    if latest_ckpt is None:
        print(f"No checkpoint (.pt) found in {ckpt_dir}; skipping automatic visualization.")
    else:
        try:
            from src.view_clustered_splats import main as _view_main
            print(f"Launching clustered splats viewer using checkpoint {latest_ckpt}...")
            # This will save a clustered checkpoint, render previews and (optionally) launch the web viewer.
            _view_main(str(latest_ckpt), clustering_results['saved_to'], out=None, previews=True, viewer=True)
        except Exception as e:
            print(f"Failed to launch clustered viewer: {e}")
except Exception as e:
    print(f"Automatic visualization step failed: {e}")

