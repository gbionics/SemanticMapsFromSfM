import os
from src.preprocessing import resize_images, generate_segmentation_masks, prepare_dirs
from src.reconstruct_with_COLMAP import run_colmap


root_dir = os.getcwd()
path_to_raw_images = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/RobotLab/Images_raw'
# path_to_raw_images = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/CRIS_Workshop/images_raw'
model_name = 'RobotWorkshop'

preprocess = False
image_size_max = 640

generate_sfm = False
generate_3dgs = True

# Main file for the reconstruction and rendering automated pipeline

# 1) Preprocessing images and segmentation masks
model_dir = os.path.join(root_dir, 'Dataset', model_name)
    
if preprocess:
    images_dir, sparse_dir, masks_dir = prepare_dirs(model_dir)      
    resize_images(path_to_raw_images, images_dir, image_size_max)
    generate_segmentation_masks(images_dir, masks_dir, n_levels=1, sam2=False)

else:
    images_dir = os.path.join(model_dir, 'images')
    sparse_dir = os.path.join(model_dir, 'sparse')
    dense_dir = os.path.join(model_dir, 'dense')
    masks_dir = os.path.join(model_dir, 'masks')
        
# 2) Generate the COLMAP SfM model

if generate_sfm:
    print('Generating the SfM reconstruction via Colmap')
    log_file_name = os.path.join(model_dir, 'colmap_reconstruction_log.txt')
    with open(log_file_name, 'w',
          buffering=1) as log_file:  
        run_colmap(images_dir, model_dir, log_file)
        

if generate_3dgs:
    print('pippo')    