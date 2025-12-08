# This code generates a COLMAP reconstruction from a set of images, with the idea to use them to reconstruct a 3DGS model enhanced with semantic information.

import os 
import numpy 
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image


# 0) Ensure that the correct folders are available

def prepare_dirs(model_dir):
    images_dir = os.path.join(model_dir, 'images')
    sparse_dir = os.path.join(model_dir, 'sparse')
    dense_dir = os.path.join(model_dir, 'dense')
    masks_dir = os.path.join(model_dir, 'masks')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    return images_dir, sparse_dir, masks_dir

# 1) Resize all images before to make the reconstruction and NVS work easier

def resize_images(input_dir, output_dir, max_side=640):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image file extensions
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                # Determine scale factor
                w, h = img.size
                scale = max_side / max(w, h)

                # Compute new size
                new_size = (int(w * scale), int(h * scale))

                # Resize with high-quality resampling
                resized_img = img.resize(new_size, Image.LANCZOS)

                # Save image
                resized_img.save(output_path)

            print(f"Saved: {output_path}")

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
    command = 'colmap feature_extractor \
               --image_path {:s} \
               --ImageReader.camera_model SIMPLE_RADIAL\
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

