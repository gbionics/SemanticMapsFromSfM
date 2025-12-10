import os
import numpy as np
from operator import itemgetter
import warnings
from src.common import path_to_sam2_checkpoint, path_to_sam_checkpoint


class Segmentor(object):
    def __init__(self, device, sam2=True, points_per_side=32, box_nms_thresh=0.95, multimask_output=True) -> None:
        self.device = device
        self.load_mask_generator(sam2=sam2, points_per_side=points_per_side, box_nms_thresh=box_nms_thresh, multimask_output=multimask_output)
    
    def load_mask_generator(self,
                            sam2=True, 
                            points_per_side=64,
                            pred_iou_thresh=0.7,
                            stability_score_thresh=0.935,
                            stability_score_offset=1.0,
                            box_nms_thresh=0.7,
                            min_mask_region_area=100,
                            multimask_output=True):
        if sam2:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_checkpoint = path_to_sam2_checkpoint
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
            sam_checkpoint = path_to_sam_checkpoint # os.path.join("/home/mcastillo", "Workspace", "SAM_checkpoints", "sam_vit_h_4b8939.pth")
            model_type = "vit_h"
            sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam_model.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                sam_model, 
                points_per_side=points_per_side,
                points_per_batch=64,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                crop_n_layers=1,
                box_nms_thresh=box_nms_thresh,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_mask_region_area
                )
    
    def generate_segmentation_map(self, image_size, maps):
        height, width = image_size
        res = np.zeros((height, width), dtype=np.uint16)

        for i in range(maps.shape[0]):
            map = maps[i]
            id_unique_list = np.unique(map)
            if id_unique_list[0] == 0:
                id_unique_list = id_unique_list[1:]
            for id in id_unique_list:
                is_valid_mask = True
                aux = np.copy(res)
                mask = map == id
                aux[mask] = id
                j_list = np.unique(res[mask])
                for j in j_list:
                    if j == 0:
                        continue
                    per_j_area = np.count_nonzero(aux == j)
                    if per_j_area <= 100:
                        is_valid_mask = False
                if is_valid_mask:
                    res[mask] = id
        
        id_unique_list = np.unique(res)
        if id_unique_list[0] == 0:
            id_unique_list = id_unique_list[1:]
        
        return res, id_unique_list

    def generate_adjacency_matrix(self, id_unique_list, segmap, binary_masks, n_levels):
        n_p = id_unique_list.shape[0]
        adj_mtx = np.zeros((n_p, n_p))
        overlap_sum = np.sum(binary_masks, axis=0)

        for id in id_unique_list:
            mask = segmap == id
            num_overlap, area_overlap = np.unique(overlap_sum[mask], return_counts=True)
            max_idx = np.argmax(area_overlap)
            mask = np.logical_and(mask, overlap_sum == num_overlap[max_idx])
            y, x = np.nonzero(mask)
            u, v = y[0], x[0]
            hot_vec = np.nonzero(binary_masks[:, u, v])[0]
            for k in range(len(hot_vec)-1):
                i = hot_vec[k]
                j = hot_vec[k+1]
                adj_mtx[i, j] = 1.0
                adj_mtx[j, i] = 1.0
        
        w_prev = 1.0

        for lvl in range(n_levels-1):
            degree = np.count_nonzero(adj_mtx, axis=-1)
            w = 1.0 / (2.0 + lvl)
            aux = adj_mtx == w_prev
            for i in range(n_p):
                d = degree[i]
                if d > 1:
                    e_list = np.nonzero(aux[i, :])[0]
                    for j in range(d-1):
                        id1 = e_list[j]
                        for k in range(j+1, d):
                            id2 = e_list[k]
                            if adj_mtx[id1, id2] == 0:
                                adj_mtx[id1, id2] = w
                                adj_mtx[id2, id1] = w
            w_prev = w
        
        adj_mtx += np.eye(n_p)
        return adj_mtx
    
    def proccess_image(self, image, n_levels):
        masks = self.mask_generator.generate(image)

        masks = sorted(masks, key=itemgetter('area'), reverse=True)

        # create a segmentation maps
        height, width = image.shape[0], image.shape[1]
        maps = np.zeros((n_levels, height, width), dtype=np.int16)

        masks_res = []
        
        for i, mask in enumerate(masks):
            k = 0
            is_untaken = np.count_nonzero(maps[k][mask['segmentation']]) / mask['area'] < 0.2 # if True then save the mask on map_0
            while not is_untaken and k < n_levels-1:
                k += 1
                is_untaken = np.count_nonzero(maps[k][mask['segmentation']]) / mask['area'] < 0.2
            if is_untaken:
                maps[k][mask['segmentation']] = i + 1
                mask['lvl'] = k
                mask['id'] = i + 1
                masks_res.append(mask)
            else:
                warnings.warn("This image has more than f{n_levels} levels of segmentation")
        
        '''import matplotlib.pyplot as plt
        plt.imshow(maps[0])
        plt.show()
        plt.imshow(maps[1])
        plt.show()'''

        res, id_unique_list = self.generate_segmentation_map((height, width), maps)

        binary_masks = np.zeros((len(id_unique_list), height, width), dtype=np.bool_)

        for j, id in enumerate(id_unique_list):
            lvl = np.nonzero(maps == id)[0][0]
            mask = maps[lvl] == id
            binary_masks[j][mask] = True
        
        adj_mtx = self.generate_adjacency_matrix(id_unique_list, res, binary_masks, n_levels)

        n_p = adj_mtx.shape[0]
        degree_matrix = np.eye(n_p) * np.count_nonzero(adj_mtx, axis=-1)
        laplacian_matrix = degree_matrix - adj_mtx
        return maps, res, adj_mtx!=0, laplacian_matrix