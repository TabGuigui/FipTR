import mmcv
import time
import torch.nn.functional as F
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from .metrics import IntersectionOverUnion, PanopticMetric
from .visualizer import visualize_motion, visualize_det
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir="val_vis",
                    show_score_thr=0.3):

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    # evaluate motion in (short, long) ranges
    EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
    num_motion_class = 2

    motion_panoptic_metrics = {}
    motion_iou_metrics = {}
    for key in EVALUATION_RANGES.keys():
        motion_panoptic_metrics[key] = PanopticMetric(n_classes=num_motion_class, temporally_consistent=True)
        motion_iou_metrics[key] = IntersectionOverUnion(num_motion_class)

    motion_eval_count = 0

    semantic_colours = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)

    out_dir = "val_vis"
    duration = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if "has_invalid_frame" in data:
                has_invalid_frame = data['has_invalid_frame'][0]
            else:
                has_invalid_frame = False
            
            start = time.time()
            result = model(return_loss=False, rescale=True, **data)
            end = time.time()
            duration.append(end - start)
            time_sum = sum(duration)
            print("FPS : %f" % (1.0 / (time_sum / len(duration))))

            if type(result) == list:
                model_name = "fistr"
            else:
                model_name = "beverse"
            
            if not has_invalid_frame or not has_invalid_frame.item():
                motion_eval_count += 1
            
                if model_name == "fistr": # fistr
                    motion_segmentation = result[0]['pts_bbox']["segmentation"].unsqueeze(0)
                    motion_instance = result[0]['pts_bbox']["instance"].unsqueeze(0)
                    if "gt_segmentation" in data:
                        motion_targets = {
                        "segmentation": data["gt_segmentation"][0],
                        "instance": data["gt_instance"][0],
                            }
                    else:
                        motion_targets = {
                        "segmentation": data["motion_segmentation"][0],
                        "instance": data["motion_instance"][0],
                            }
                    
                    visualize_motion(motion_targets, {"segmentation": motion_segmentation, "instance": motion_instance}, 
                    save_path = out_dir, model="fistr", index=i)

                    visualize_det(img_metas = data["img_metas"][0].data[0][0],
                          bbox_results= result[0], 
                          vis_thresh = 0.25, 
                          save_path = out_dir, index=i)
                
                else: # beverse
                    motion_segmentation = result['motion_segmentation'] # bs 5 1 200 200 bs*t*1*200*200
                    motion_instance = result["motion_instance"] # bs 5 200 200 bs*t*200*200
                    motion_targets = {
                            'motion_segmentation': data['motion_segmentation'][0], # bs 5 200 200
                            'motion_instance': data['motion_instance'][0], # bs 5 200 200
                            'instance_centerness': data['instance_centerness'][0], # bs 5 1 200
                            'instance_offset': data['instance_offset'][0], # bs 5 2 200
                            'instance_flow': data['instance_flow'][0], # bs 5 2 200 200
                            'future_egomotion': data['future_egomotions'][0], # bs 7 6
                        }
                    motion_targets, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(motion_targets) 

                    visualize_motion(motion_targets, result["motion_predictions"], 
                                     save_path = out_dir, model="beverse", index = i)
                
                for key, grid in EVALUATION_RANGES.items():
                    limits = slice(grid[0], grid[1])

                    motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                            ).cpu(), motion_targets['segmentation'][..., limits, limits].contiguous())
                    motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous().cpu(),
                            motion_targets["instance"][..., limits, limits].contiguous())

            
            results.extend(result)

        batch_size = 1
        for _ in range(batch_size):
            prog_bar.update()
    
    print('\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))
    
    for key, grid in EVALUATION_RANGES.items():
        results_str = 'grid = {}: '.format(key)

        iou_scores = motion_iou_metrics[key].compute()
        panoptic_scores = motion_panoptic_metrics[key].compute()

        results_str += 'iou = {:.3f}, '.format(
            iou_scores[1].item() * 100)

        for panoptic_key, value in panoptic_scores.items():
                        results_str += '{} = {:.3f}, '.format(
                            panoptic_key, value[1].item() * 100)
    return results