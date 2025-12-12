import os
from PIL import Image
from src.utils.segmentor import Segmentor
from tqdm import tqdm
import numpy as np


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

    progress_bar = tqdm(range(len(os.listdir(input_dir))), desc="Pre-processing progress... Rescaling the images")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                progress_bar.update(1)
                continue

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
            progress_bar.update(1)

            # print(f"Saved: {output_path}")
            
def generate_segmentation_masks(input_dir, output_path, n_levels=1, sam2=False):
    # read the images from the input directory
    images_dirlist = sorted(os.listdir(input_dir))
    images = [Image.open(os.path.join(input_dir, image_name)) for image_name in images_dirlist]
    sam = Segmentor("cuda", sam2)

    progress_bar = tqdm(range(len(images)), desc="Pre-processing progress... Extracting segmentation masks")
    
    for img, dir in zip(images, images_dirlist):
        
        orig_w, orig_h = img.size
        if orig_w > 1600:
            scale = orig_h / 1080
            resolution = (int(orig_w / scale), int(orig_h / scale))
            img = img.resize(resolution)
        img_numpy = np.array(img.convert("RGB"))
        img_name = dir.split('.')[0]
        
        if os.path.exists(os.path.join(output_path, img_name + ".npy")): 
            progress_bar.update(1)
            continue
        
        l_maps, segmap, adj_mtx, laplacian = sam.proccess_image(img_numpy, n_levels=n_levels)

        np.save(os.path.join(output_path, img_name + ".npy"), segmap)
        np.save(os.path.join(output_path, img_name + "_am.npy"), adj_mtx)
        np.save(os.path.join(output_path, img_name + "_lm.npy"), laplacian)
        np.save(os.path.join(output_path, img_name + "_maps.npy"), l_maps)
        progress_bar.update(1)
    
    progress_bar.close()

                            
if __name__ == "__main__":
    import argparse

    root_dir = os.getcwd()
    path_to_raw_images = os.path.join(root_dir, 'Dataset/CRIS_Workshop_mini/images_raw')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help='input directory with the raw, original images', default= path_to_raw_images)
    parser.add_argument("--model_name", help='output directory where to store the scaled images', default= 'CRIS_Workshop_mini')
    # parser.add_argument("--model_dir", help='output directory where to store the scaled images', default= rec_path + '/COLMAP')
    parser.add_argument("--size", help='desired maximum length of the scaled images', default=640)
    args = parser.parse_args()

    rec_path = os.path.join(root_dir, 'Dataset', args.model_name)

    images_dir, sparse_dir, masks_dir = prepare_dirs(rec_path)      
    resize_images(path_to_raw_images, images_dir, args.size)
    generate_segmentation_masks(images_dir, masks_dir, n_levels=1, sam2=False)
    
    
