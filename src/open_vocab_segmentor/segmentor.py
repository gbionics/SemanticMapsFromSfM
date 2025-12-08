import os
import numpy as np
from operator import itemgetter
import warnings

class Segmentor(object):
    def __init__(self, device, sam2=False, points_per_side=32, box_nms_thresh=0.95, multimask_output=True) -> None:
        self.device = device
        self.load_mask_generator(sam2=sam2, points_per_side=points_per_side, box_nms_thresh=box_nms_thresh, multimask_output=multimask_output)
    
    def load_mask_generator(self, 
                            sam2=True, 
                            points_per_side=32,
                            pred_iou_thresh=0.7,
                            stability_score_thresh=0.925,
                            stability_score_offset=1.0,
                            box_nms_thresh=0.7,
                            min_mask_region_area=100,
                            multimask_output=True):
        if sam2:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_checkpoint = os.path.join("/home/mcastillo", "Workarea", "sam2", "checkpoints", "sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=points_per_side,
                points_per_batch=64,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                crop_n_layers=1,
                box_nms_thresh=box_nms_thresh,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_mask_region_area,
                use_m2m=False,
                multimask_output=multimask_output,
                )
        else:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            sam_checkpoint = os.path.join("/data01/kriahidehkordi", "segment-anything", "checkpoints", "sam_vit_h_4b8939.pth")
            model_type = "vit_h"
            sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam_model.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                sam_model, 
                points_per_side=points_per_side,
                points_per_batch=64,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=1.0,
                crop_n_layers=1,
                box_nms_thresh=box_nms_thresh,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_mask_region_area
                )
    
    def generate_segmentation_map(self, image):
        masks = self.mask_generator.generate(image)

        masks = sorted(masks, key=itemgetter('area'), reverse=True)
        
        # create a segmentation map
        height, width = image.shape[0], image.shape[1]
        res = np.zeros((height, width))
        
        for i, mask in enumerate(masks):
            is_untaken = np.count_nonzero(res[mask['segmentation']]) < 100
            if is_untaken:
                res[mask['segmentation']] = i+1
        
        return res
    
    def proccess_image(self, image):
        masks = self.mask_generator.generate(image)

        height, width = image.shape[0], image.shape[1]
        res = np.zeros((len(masks), height, width), dtype=np.bool_)

        for i, mask in enumerate(masks):
            mask['id'] = i
            res[i][mask['segmentation']] = True
        
        return masks, res