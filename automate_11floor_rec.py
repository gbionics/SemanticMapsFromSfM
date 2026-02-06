import os
import sys

from src.preprocessing import prepare_dirs, resize_images
from src.reconstruct_with_COLMAP import run_colmap

# Start by getting a list of all the datasets available

DATA_ROOT = "/home/mtoso/Documents/Datasets/"
SAVE_ROOT = "/home/mtoso/Documents/SemanticMapsFromSfM/Dataset/"

sparsify_images = False
root_dir = os.getcwd()
image_size_max = 640

if sparsify_images:
    scenes = os.listdir(DATA_ROOT)
    print("Available scenes:")
    for i, scene in enumerate(scenes):
        print(f"{i}: {scene}")  

    # For each scene, prepare the reconstruction directory and copy the images there, sampling one every 10 images sorted by name
    for scene in scenes:
        print(f"Processing scene: {scene}")
        subscenes = os.listdir(os.path.join(DATA_ROOT, scene))
        for subscene in subscenes:
            print(f"  Processing subscene: {subscene}")
            
            # create a reconstruction directory for the subscene
            recon_dir = os.path.join(SAVE_ROOT, f"{scene}_{subscene}")
            os.makedirs(recon_dir, exist_ok=True)

            # Create a raw_images directory and copy the images there, sampling one every 10 images sorted by name
            raw_images_dir = os.path.join(recon_dir, "raw_images")
            os.makedirs(raw_images_dir, exist_ok=True)
            subscene_path = os.path.join(DATA_ROOT, scene, subscene)
            images = sorted([f for f in os.listdir(subscene_path) if f.endswith('.jpg') or f.endswith('.png')])
            sampled_images = images[::10]  # Sample one every 10 images
            for img in sampled_images:
                src = os.path.join(subscene_path, img)
                dst = os.path.join(raw_images_dir, img)
                os.symlink(src, dst)  # Create a symbolic link to save space

scenes = os.listdir(SAVE_ROOT)
print("Available scenes:")
for i, scene in enumerate(scenes):
    print(f"Processing scene: {scene}")
    try:
        model_dir = os.path.join(SAVE_ROOT, scene)
        images_dir, sparse_dir, masks_dir, visual_dir, dense_dir = prepare_dirs(model_dir)
        
        path_to_raw_images = os.path.join(model_dir, "raw_images")

        resize_images(path_to_raw_images, images_dir, image_size_max)
        # generate_segmentation_masks(images_dir, masks_dir, n_levels=1, sam2=False)
        # visualize_masks(masks_dir, visual_dir, images_dir, sample=50)

        print('Generating the SfM reconstruction via Colmap')
        log_file_name = os.path.join(model_dir, 'colmap_reconstruction_log.txt')
        with open(log_file_name, 'w',
            buffering=1) as log_file:  
            run_colmap(images_dir, model_dir, log_file)
    except:
        print(f"Error processing scene: {scene}")
        continue    