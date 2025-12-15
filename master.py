import os
import tyro
from src.preprocessing import resize_images, generate_segmentation_masks, prepare_dirs, visualize_masks
from src.reconstruct_with_COLMAP import run_colmap
from src.splatting_2DGS import Config, Runner

root_dir = os.getcwd()
path_to_raw_images = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/RobotLab/Images_raw'
model_name = 'RobotLab'

preprocess = True
image_size_max = 640

generate_sfm = False

generate_2DGS = True

# Main file for the reconstruction and rendering automated pipeline

# 1) Preprocessing images and segmentation masks
model_dir = os.path.join(root_dir, 'Dataset', model_name)
    
if preprocess:
    images_dir, sparse_dir, masks_dir, visual_dir = prepare_dirs(model_dir)      
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

# 3) Initialize the 2DGS reconstruction

if generate_2DGS:

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    
    cfg.data_dir = model_dir
    cfg.result_dir = dense_dir
    cfg.semantic_only_start = 20000
    
    runner = Runner(cfg)

    runner.train()