import numpy as np
from PIL import Image, ImageFilter
import clip, torch, torchvision
from copy import deepcopy
from open_vocab_segmentor.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork

def extract_image_segment(img, mask):
    img_numpy = np.array(img)
    segment_numpy = np.zeros_like(img_numpy)
    
    # extract segment of interest
    segment_numpy[mask] = img_numpy[mask]
    segment = Image.fromarray(segment_numpy)
    
    transparency_mask_numpy = np.zeros_like(mask, dtype=np.uint8)
    transparency_mask_numpy[mask] = 255
    transparency_mask = Image.fromarray(transparency_mask_numpy, mode='L')
    res = Image.new("RGB", img.size, (0, 0, 0))
    
    # segmented image with a transparent background
    res.paste(segment, mask=transparency_mask)
    
    return res

def get_segment_list(img, masks):
    res = [torchvision.transforms.functional.pil_to_tensor(extract_image_segment(img, mask['segmentation']).crop(box_xywh_to_xyxy(mask['bbox'])).resize((224, 224))) for mask in masks]
    return res

def generate_clip_embeddings(img, segmap, device="cuda"):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    segment_list = get_segment_list(img, segmap)

    if not segment_list:
        return np.empty((0, 512))
    
    # preprocess each segment in the image
    preprocessed_images = [preprocess(segment).to(device) for segment in segment_list]
    stacked_images = torch.stack(preprocessed_images)
    
    # get a clip-embedding per each segment in the image
    image_features = clip_model.encode_image(stacked_images)
    image_features_numpy = image_features.cpu().detach().numpy()
    
    return image_features_numpy

def box_xywh_to_xyxy(box_xywh):
    box_xyxy = deepcopy(box_xywh)
    box_xyxy[2] = box_xyxy[2] + box_xyxy[0]
    box_xyxy[3] = box_xyxy[3] + box_xyxy[1]
    return box_xyxy

def generate_crop_boxes(masks, img_size):
    for mask in masks:
        box_xyxy = deepcopy(mask["bbox"])
        width, height = mask["bbox"][2:]
        x, y = mask["bbox"][:2]
        padding = abs(width - height) // 2 + 0
        is_width_larger = width >= height
        if is_width_larger:
            padding = padding if ((y-padding >= 0) and (y+height+padding < img_size[1])) else min(y, img_size[1]-(y+height+1))
            box_xyxy[1], box_xyxy[3]  = y-padding, y+height+padding
            padding = 0 if ((x-0 >= 0) and (x+width+0 < img_size[0])) else min(x, img_size[0]-(x+width+1))
            box_xyxy[0], box_xyxy[2] = x-padding, x+width+padding
        else:
            padding = padding if ((x-padding >= 0) and (x+width+padding < img_size[0])) else min(x, img_size[0]-(x+width+1))
            box_xyxy[0], box_xyxy[2] = x-padding, x+width+padding
            padding = 0 if ((y-0 >= 0) and (y+height+0 < img_size[1])) else min(y, img_size[1]-(y+height+1))
            box_xyxy[1], box_xyxy[3]  = y-padding, y+height+padding
        mask["bbox_xyxy"] = box_xyxy
    return masks

def compute_espresso_embeddings(img_features):
    pca_features = torch.zeros_like(img_features)
    
    for i, ftr in enumerate(img_features):
        ftr = ftr[..., None]
        A = torch.softmax((1 / np.sqrt(512.0)) * (ftr @ ftr.T), dim=1)
        U, s, Vt = torch.linalg.svd(A)
        A_k = (s * U.T)
        pca_features[i, ...] = (A_k @ ftr).squeeze()
    
    return pca_features.float().cpu().numpy()


class CLIPGenerator(object):
    def __init__(self, device):
        self.device = device
        self.load_clip_generator()
    
    def load_clip_generator(self):
        self.clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    
    def proccess_image(self, image, masks, pca_features=True):
        if not masks:
            return np.empty((0, 512))
        
        patches = get_segment_list(image, masks)
        
        # preprocess each segment in the image
        stacked_images = torch.stack(patches, dim=0) / 255.
        stacked_images = stacked_images.to(self.device)

        # get a clip-embedding per each segment in the image
        image_features = self.clip_model.encode_image(stacked_images)
        image_features_quality = self.clip_model.compute_quality(image_features)

        image_features_pca = compute_espresso_embeddings(image_features) if pca_features else None

        image_features_numpy = image_features.float().cpu().numpy()
        image_features_quality_numpy = image_features_quality.cpu().numpy()
        
        return image_features_numpy, image_features_quality_numpy, image_features_pca