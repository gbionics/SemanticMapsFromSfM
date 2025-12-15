import os
from PIL import Image
from src.utils.segmentor import Segmentor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def visualize_masks(masks_dir: str, out_repo: str, images_dir: str = None, sample: int = 10, alpha: float = 0.6):
    """Generate and store segmentation mask overlays into a repository directory.

    This function requires `out_repo` (a path to a repository or output directory) and will save
    overlay images there instead of displaying them interactively.

    Args:
        masks_dir: directory containing mask files (.npy). Expects files like "frame_00000015.npy" (segmap)
        out_repo: directory where overlay images will be stored (required).
        images_dir: optional directory containing corresponding RGB images (same stem names). If provided,
                    the mask will be overlaid on the image; otherwise the color mask alone is stored.
        sample: number of masks to process (if <=0, process all). If greater than number of files, all are used.
        alpha: blend factor for overlaying mask on image (0..1).
    """
    # out_repo is required
    if out_repo is None:
        raise ValueError("out_repo is required and must be a valid directory path to store overlays")

    # collect mask files (exclude auxiliary files ending with _am, _lm, _maps)
    files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy') and not (f.endswith('_am.npy') or f.endswith('_lm.npy') or f.endswith('_maps.npy'))])
    if len(files) == 0:
        print(f"No mask .npy files found in {masks_dir}")
        return

    if sample is None or sample <= 0:
        sel = files
    else:
        sel = files[:min(sample, len(files))]

    # Ensure output repository directory exists
    out_dir = os.path.join(out_repo, 'mask_overlays')
    os.makedirs(out_dir, exist_ok=True)

    for fname in sel:
        path = os.path.join(masks_dir, fname)
        mask = np.load(path)

        # handle unexpected shapes: if mask has channels, try to pick the first channel
        if mask.ndim == 3 and mask.shape[2] in (1,):
            mask = mask[..., 0]
        if mask.ndim != 2:
            # try to reduce to 2D by argmax along channel axis
            if mask.ndim == 3:
                mask = mask.argmax(axis=2)
            else:
                raise RuntimeError(f"Unsupported mask shape: {mask.shape} for file {fname}")

        h, w = mask.shape
        uniques = np.unique(mask)

        # build a color palette deterministic from label value
        palette = {}
        rng = np.random.RandomState(0)
        for lab in uniques:
            if lab == 0:
                palette[int(lab)] = np.array([0, 0, 0], dtype=np.uint8)  # background black
            else:
                # deterministic color by hashing label
                rng.seed(int(lab) + 1)
                palette[int(lab)] = (rng.randint(30, 230, size=3)).astype(np.uint8)

        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for lab, col in palette.items():
            color_mask[mask == lab] = col

        if images_dir is not None:
            # try to find matching image by same stem
            stem = os.path.splitext(fname)[0]
            # try common image extensions
            found_img = None
            for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'):
                candidate = os.path.join(images_dir, stem + ext)
                if os.path.exists(candidate):
                    found_img = candidate
                    break
            if found_img is None:
                # fallback: try any file starting with stem
                for candidate in os.listdir(images_dir):
                    if candidate.startswith(stem):
                        found_img = os.path.join(images_dir, candidate)
                        break

            if found_img is not None:
                img = np.array(Image.open(found_img).convert('RGB'))
                # resize mask overlay if image size differs
                if img.shape[0] != h or img.shape[1] != w:
                    color_mask_pil = Image.fromarray(color_mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
                    color_mask = np.array(color_mask_pil)
                overlay = (img * (1.0 - alpha) + color_mask.astype(np.float32) * alpha).astype(np.uint8)
                display_img = overlay
            else:
                display_img = color_mask
        else:
            display_img = color_mask

        plt.figure(figsize=(10, 8))
        plt.imshow(display_img)
        plt.axis('off')
        plt.title(fname)

        out_path = os.path.join(out_dir, stem + '_overlay.png')
        plt.imsave(out_path, display_img)
        plt.close()

    print(f"Saved {len(sel)} overlays to: {out_dir}")



# 0) Ensure that the correct folders are available

def prepare_dirs(model_dir):
    images_dir = os.path.join(model_dir, 'images')
    sparse_dir = os.path.join(model_dir, 'sparse')
    dense_dir = os.path.join(model_dir, 'dense')
    masks_dir = os.path.join(model_dir, 'masks')
    visual_dir = os.path.join(model_dir, 'visualize')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    return images_dir, sparse_dir, masks_dir, visual_dir

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
    
    
