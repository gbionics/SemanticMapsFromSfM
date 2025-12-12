# This code generates a COLMAP reconstruction from a set of images, with the idea to use them to reconstruct a 3DGS model enhanced with semantic information.

import os 
import numpy 
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image
from src.preprocessing import prepare_dirs, resize_images, generate_segmentation_masks


# 2) Run COLMAP reconstruction to generate a sparse model

def run_colmap(images_dir, rec_dir, log_file):
    # Useful information from the COLMAP CODE:
    # void OptionManager::ModifyForExtremeQuality() {
    #      // Most of the options are set to extreme quality by default.
    #      sift_extraction->estimate_affine_shape = true;
    #      sift_extraction->domain_size_pooling = true;
    #      sift_matching->guided_matching = true;
    #      mapper->ba_local_max_num_iterations = 40;
    #      mapper->ba_local_max_refinements = 3;
    #      mapper->ba_global_max_num_iterations = 100;
    #    }

    # 1. Feature extraction.
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting feature extraction.\n'
    log_file.write(log_message)
    command = 'QT_QPA_PLATFORM=offscreen colmap feature_extractor \
    --image_path /home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/CRIS_Workshop_mini/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.estimate_affine_shape=true \
    --SiftExtraction.domain_size_pooling=true \
    --database_path /home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Dataset/CRIS_Workshop_mini/sparse/database.db'

    command = 'colmap feature_extractor \
               --image_path {:s} \
               --ImageReader.camera_model PINHOLE\
               --SiftExtraction.estimate_affine_shape=true \
               --SiftExtraction.domain_size_pooling=true \
               --database_path {:s}/database.db'.format(images_dir, rec_dir)
    print(command)
    os.system(command)

    # 2. Feature matching.
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting feature matching.\n'
    log_file.write(log_message)
    command = 'export QT_QPA_PLATFORM=offscreen'
    os.system(command)
    command = 'colmap exhaustive_matcher \
               --SiftMatching.guided_matching=true \
               --database_path {:s}/database.db'.format(rec_dir)
    # This might be needed to limit memory usage: --SiftMatching.max_num_matches 10000 \
    print(command)
    os.system(command)

    # 3. Sparse reconstruction.

    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting sparse reconstruction.\n'
    log_file.write(log_message)
    Path(rec_dir + '/sparse/').mkdir(parents=True, exist_ok=True)
    command = 'colmap mapper \
               --database_path {:s}/database.db \
               --image_path {:s} \
               --output_path {:s}'.format(rec_dir, images_dir, rec_dir)
    # --ba_local_max_num_iterations=40 \
    # --ba_local_max_refinements=3 \
    # --ba_global_max_num_iterations=100'.format(data_root_path, data_root_path, data_root_path)
    print(command)
    os.system(command)

    # 4. Image undistortion (WARNING: This works on the first segment, which is not guaranteed to be a large one).
    """now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting image undistortion.\n'
    log_file.write(log_message)
    command = 'colmap image_undistorter \
              --image_path {:s}/images_all \
              --input_path {:s}/sparse/0 \
              --output_path {:s}/dense \
              --output_type COLMAP \
              --max_image_size 9000'.format(data_root_path, data_root_path, data_root_path)  # It seems that images with a width larger than this one are rescaled before being operated on.
    print(command)
    system(command)

    # 5. Patch stereo matching.
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting patch stereo matching.\n'
    log_file.write(log_message)
    command = 'colmap patch_match_stereo \
           --workspace_path {:s}/dense \
           --workspace_format COLMAP \
           --PatchMatchStereo.geom_consistency true'.format(data_root_path)
    print(command)
    system(command)

    # 6. Stereo fusion.
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    log_message = '\t' + dt_string + ' - Starting stereo fusion.\n'
    log_file.write(log_message)
    command = 'colmap stereo_fusion \
               --workspace_path {:s}/dense \
               --workspace_format COLMAP \
               --input_type geometric \
               --output_path {:s}/dense/fused.ply'.format(data_root_path, data_root_path)
    print(command)
    system(command)"""

# 3) Generate the segmenation masks for each image


if __name__ == "__main__":
    import argparse

    root_dir = os.getcwd()
    project_name = 'CRIS_Workshop_mini'
    rec_path = os.path.join(root_dir, 'Dataset', project_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help='input directory with the raw, original images', default= rec_path + '/images_raw')
    parser.add_argument("--model_dir", help='output directory where to store the scaled images', default= rec_path + '/COLMAP')
    parser.add_argument("--size", help='desired maximum length of the scaled images', default=640)
    args = parser.parse_args()

    images_dir, sparse_dir, masks_dir = prepare_dirs(args.model_dir)      
    resize_images(args.input_dir, images_dir, args.size)

    log_file_name = os.path.join(args.model_dir, 'colmap_reconstruction_log.txt')
    with open(log_file_name, 'w',
          buffering=1) as log_file:  # The buffering parameter is set to that data is written immediately.

        run_colmap(images_dir, sparse_dir, log_file)

    # preprocess(images_dir, rec_path, 'segmentation_masks', skip_pca=False)

