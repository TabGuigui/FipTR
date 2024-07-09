import mmcv
import cv2
import torch.nn.functional as F
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from PIL import Image
from scipy.optimize import linear_sum_assignment

from ..fiptr.utils.instance import predict_instance_segmentation_and_trajectories as predict_beverse
from ..fiptr.visualize.motion_visualisation import plot_instance_map
from .metrics import IntersectionOverUnion, PanopticMetric

import time
def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out
def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
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
        motion_panoptic_metrics[key] = PanopticMetric(
            n_classes=num_motion_class, temporally_consistent=True)
        motion_iou_metrics[key] = IntersectionOverUnion(
            num_motion_class)

    motion_eval_count = 0

    semantic_colours = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)

    rotate_flag = 0
    recorder = {"segmentation": [], "instance": []}
    duration = []
    for i, data in enumerate(data_loader):
        if i not in [8, 32, 48, 66]:
        # if i not in [66]:
            continue
        with torch.no_grad():
            # has_invalid_frame = data['has_invalid_frame'][0] # 需要判断有没有非法图像
            if "has_invalid_frame" in data:
                has_invalid_frame = data['has_invalid_frame'][0]
            elif "gt_occ_has_invalid_frame" in data:
                has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
            else:
                has_invalid_frame = False
            start = time.time()
            result = model(return_loss=False, rescale=True, **data)
            end = time.time()
            duration.append(end - start)
            time_sum = sum(duration)
            print("FPS : %f" % (1.0 / (time_sum / len(duration))))
            if not has_invalid_frame or not has_invalid_frame.item():
                motion_eval_count += 1
                if type(result) == list: # bevpanformer
                    if result[0]['pts_bbox']["segmentation"].dim() == 3:
                        motion_segmentation = result[0]['pts_bbox']["segmentation"].unsqueeze(0)
                        motion_instance = result[0]['pts_bbox']["instance"].unsqueeze(0)
                    else:
                        motion_segmentation = result[0]['pts_bbox']["segmentation"].unsqueeze(0).unsqueeze(0)
                        motion_instance = result[0]['pts_bbox']["instance"].unsqueeze(0).unsqueeze(0)
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
                    rotate_flag = 0
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
                    rotate_flag = 1
                    motion_targets, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(motion_targets) 
                    # sementation: bs t 1 200 200, instance: bs t 200 200
                imgs = []
                if motion_segmentation.shape[1] == 5:
                    
                    for j in range(5):
                        final_img = show_motion({key: value[: ,j].unsqueeze(1) for key, value in motion_targets.items() if key in ["segmentation", "instance"]}, motion_segmentation[:, j].unsqueeze(1), motion_instance[:, j].unsqueeze(1)
                                    , semantic_colours, rotate_flag = rotate_flag, i = j)
                        imgs.append(final_img)
                    fig = plt.figure(figsize=(10,2),dpi=300)
                    for j in range(5):
                        ax = plt.subplot(1,5,j + 1)
                        plt.imshow(imgs[j])
                        plt.axis('off')
                        # ax.set_ylabel("gt", fontsize=16)
                        plt.draw()
                    plt.savefig("try.jpg")
                    # beverse
                    if "motion_predictions" in result:
                        show_motion_traj(motion_targets, result["motion_predictions"], rotate_flag=True)
                    else:
                        show_motion_traj(motion_targets, {"segmentation": motion_segmentation, "instance": motion_instance}, rotate_flag=False)

                else:
                    show_motion(motion_targets, motion_segmentation, motion_instance, semantic_colours, rotate_flag = rotate_flag, i="singleframe")
                for key, grid in EVALUATION_RANGES.items():
                    limits = slice(grid[0], grid[1])

                    motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                            ).cpu(), motion_targets['segmentation'][..., limits, limits].contiguous())
                    motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous().cpu(),
                            motion_targets["instance"][..., limits, limits].contiguous())

            results.extend(result)
            # motion_labels = model.module.pts_bbox_head.prepare_future_labels(
            #             motion_target)
              
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
        print(results_str)
    return results
        

def show_motion_traj(motion_targets, motion_preds, rotate_flag=True):
    if rotate_flag:
        segmentation_binary = motion_targets['segmentation']
        segmentation = segmentation_binary.new_zeros(
                segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)

        motion_labels = dict()
        motion_labels['segmentation'] = segmentation.float() * 10
        motion_labels["instance"] = motion_targets["instance"]
        motion_labels['instance_center'] = motion_targets['centerness']
        motion_labels['instance_offset'] = motion_targets['offset']
        motion_labels['instance_flow'] = motion_targets['flow']
        gt_image = plot_motion_prediction(motion_labels, "beverse")
        pred_image = plot_motion_prediction(motion_preds, "beverse")
    else:
        gt_image = plot_motion_prediction(motion_targets, "bevpanformer")
        pred_image = plot_motion_prediction(motion_preds, "bevpanformer")

    

    final_image = np.concatenate([pred_image, gt_image])
    final_image = flip_rotate_image(final_image)
    fig = plt.figure(figsize=(6, 6),dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.draw()
    plt.savefig("motion.jpg")

def plot_motion_prediction(motion_preds, title):
    if title == "beverse" :
        consistent_instance_seg, matched_centers = predict_beverse(motion_preds, compute_matched_centers=True)
    elif title == "bevpanformer":
        consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(motion_preds, compute_matched_centers=True)
    unique_ids = torch.unique(
        consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(
        consistent_instance_seg[0, 0].cpu().numpy(), instance_map)

    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, tuple(path[t].astype(np.int)), tuple(path[t + 1].astype(np.int)), color, 4)

    # # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]
    return vis_image

def predict_instance_segmentation_and_trajectories(
    output, compute_matched_centers=False, 
):
    preds = output['segmentation'].detach()

    batch_size, seq_len = preds.shape[:2]
    pred_inst = output["instance"]

    consistent_instance_seg = pred_inst
    if compute_matched_centers:
        assert batch_size == 1
        # Generate trajectories
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device),
        ))

        for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

        return consistent_instance_seg, matched_centers
    return consistent_instance_seg


def make_instance_id_temporally_consistent(pred_inst, future_flow, matching_threshold=3.0):
    """
    Parameters
    ----------
        pred_inst: torch.Tensor (1, seq_len, h, w)
        future_flow: torch.Tensor(1, seq_len, 2, h, w)
        matching_threshold: distance threshold for a match to be valid.

    Returns
    -------
    consistent_instance_seg: torch.Tensor(1, seq_len, h, w)

    1. time t. Loop over all detected instances. Use flow to compute new centers at time t+1.
    2. Store those centers
    3. time t+1. Re-identify instances by comparing position of actual centers, and flow-warped centers.
        Make the labels at t+1 consistent with the matching
    4. Repeat
    """
    assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

    # Initialise instance segmentations with prediction corresponding to the present
    consistent_instance_seg = [pred_inst[0, 0]]
    largest_instance_id = consistent_instance_seg[0].max().item()

    _, seq_len, h, w = pred_inst.shape
    device = pred_inst.device
    for t in range(seq_len - 1):
        # Compute predicted future instance means
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device), torch.arange(
                w, dtype=torch.float, device=device),
        ))

        # Add future flow
        grid = grid + future_flow[0, t]
        warped_centers = []
        # Go through all ids, except the background
        t_instance_ids = torch.unique(
            consistent_instance_seg[-1])[1:].cpu().numpy()

        if len(t_instance_ids) == 0:
            # No instance so nothing to update
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in t_instance_ids:
            instance_mask = (consistent_instance_seg[-1] == instance_id)

            warped_centers.append(grid[:, instance_mask].mean(dim=1))
        warped_centers = torch.stack(warped_centers)

        # Compute actual future instance means
        centers = []
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device), torch.arange(
                w, dtype=torch.float, device=device),
        ))
        n_instances = int(pred_inst[0, t + 1].max().item())

        if n_instances == 0:
            # No instance, so nothing to update.
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in range(1, n_instances + 1):
            instance_mask = (pred_inst[0, t + 1] == instance_id)
            centers.append(grid[:, instance_mask].mean(dim=1))
        centers = torch.stack(centers)

        # Compute distance matrix between warped centers and actual centers
        distances = torch.norm(centers.unsqueeze(
            0) - warped_centers.unsqueeze(1), dim=-1).cpu().numpy()
        # outputs (row, col) with row: index in frame t, col: index in frame t+1
        # the missing ids in col must be added (correspond to new instances)
        ids_t, ids_t_one = linear_sum_assignment(distances)
        matching_distances = distances[ids_t, ids_t_one]
        # Offset by one as id=0 is the background
        ids_t += 1
        ids_t_one += 1

        # swap ids_t with real ids. as those ids correspond to the position in the distance matrix.
        id_mapping = dict(
            zip(np.arange(1, len(t_instance_ids) + 1), t_instance_ids))
        ids_t = np.vectorize(id_mapping.__getitem__, otypes=[np.int64])(ids_t)

        # Filter low quality match
        ids_t = ids_t[matching_distances < matching_threshold]
        ids_t_one = ids_t_one[matching_distances < matching_threshold]

        # Elements that are in t+1, but weren't matched
        remaining_ids = set(torch.unique(
            pred_inst[0, t + 1]).cpu().numpy()).difference(set(ids_t_one))
        # remove background
        remaining_ids.remove(0)
        #  Set remaining_ids to a new unique id
        for remaining_id in list(remaining_ids):
            largest_instance_id += 1
            ids_t = np.append(ids_t, largest_instance_id)
            ids_t_one = np.append(ids_t_one, remaining_id)

        consistent_instance_seg.append(update_instance_ids(
            pred_inst[0, t + 1], old_ids=ids_t_one, new_ids=ids_t))

    consistent_instance_seg = torch.stack(consistent_instance_seg).unsqueeze(0)
    return consistent_instance_seg

def get_instance_segmentation_and_centers(
    center_predictions: torch.Tensor,
    offset_predictions: torch.Tensor,
    foreground_mask: torch.Tensor,
    conf_threshold: float = 0.1,
    nms_kernel_size: float = 3,
    max_n_instance_centers: int = 100,
):
    width, height = center_predictions.shape[-2:]
    center_predictions = center_predictions.view(1, width, height)
    offset_predictions = offset_predictions.view(2, width, height)
    foreground_mask = foreground_mask.view(1, width, height)

    centers = find_instance_centers(
        center_predictions, conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size)
    if not len(centers):
        return torch.zeros(center_predictions.shape, dtype=torch.int64, device=center_predictions.device), \
            torch.zeros((0, 2), device=centers.device)

    if len(centers) > max_n_instance_centers:
        # print(f'There are a lot of detected instance centers: {centers.shape}')
        centers = centers[:max_n_instance_centers].clone()

    # 每个像素位置 + 预测的 offset，分配给最近的物体
    instance_ids = group_pixels(centers, offset_predictions)
    instance_seg = (instance_ids * foreground_mask.float()).long()

    # Make the indices of instance_seg consecutive
    instance_seg = make_instance_seg_consecutive(instance_seg)

    return instance_seg.long(), centers

def find_instance_centers(center_prediction: torch.Tensor, conf_threshold: float = 0.1, nms_kernel_size: float = 3):
    assert len(center_prediction.shape) == 3
    center_prediction = F.threshold(
        center_prediction, threshold=conf_threshold, value=-1)

    nms_padding = (nms_kernel_size - 1) // 2
    maxpooled_center_prediction = F.max_pool2d(
        center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
    )

    # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
    center_prediction[center_prediction != maxpooled_center_prediction] = -1
    return torch.nonzero(center_prediction > 0)[:, 1:]

def group_pixels(centers: torch.Tensor, offset_predictions: torch.Tensor) -> torch.Tensor:
    width, height = offset_predictions.shape[-2:]
    x_grid = (
        torch.arange(width, dtype=offset_predictions.dtype,
                     device=offset_predictions.device)
        .view(1, width, 1)
        .repeat(1, 1, height)
    )
    y_grid = (
        torch.arange(height, dtype=offset_predictions.dtype,
                     device=offset_predictions.device)
        .view(1, 1, height)
        .repeat(1, width, 1)
    )
    pixel_grid = torch.cat((x_grid, y_grid), dim=0)
    center_locations = (pixel_grid + offset_predictions).view(2,
                                                              width * height, 1).permute(2, 1, 0)
    centers = centers.view(-1, 1, 2)

    distances = torch.norm(centers - center_locations, dim=-1)

    instance_id = torch.argmin(distances, dim=0).reshape(1, width, height) + 1
    return instance_id

def make_instance_seg_consecutive(instance_seg):
    # Make the indices of instance_seg consecutive
    unique_ids = torch.unique(instance_seg)
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg

def update_instance_ids(instance_seg, old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """
    indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id

    return indices[instance_seg].long()

def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }


def show_motion(motion_targets, segmentation_pred,instance_pred, semantic_colours, rotate_flag=True, i = 0):

    semantic_seg = motion_targets['segmentation'].squeeze(2).cpu().numpy()
    semantic_plot = semantic_colours[semantic_seg[0, 0]]
    semantic_plot = make_contour(semantic_plot)
            
    semantic_pred = segmentation_pred.squeeze(2).cpu().numpy()
    semantic_pred = semantic_colours[semantic_pred[0, 0]]
    semantic_pred = make_contour(semantic_pred)

    # instance show
    pred = instance_pred
    gt = motion_targets['instance']
    unique_ids = torch.unique(
        pred).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))

    instance_pred = plot_instance_map(
        pred.cpu().numpy(), instance_map)
    
    unique_ids = torch.unique(
        gt).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_gt = plot_instance_map(
        gt.cpu().numpy(), instance_map)

    if rotate_flag:
        semantic_plot = flip_rotate_image(semantic_plot)
        semantic_pred = flip_rotate_image(semantic_pred)
        instance_gt = flip_rotate_image(instance_gt)
        instance_pred = flip_rotate_image(instance_pred)
        title_name = "beverse pred"
    else:
        semantic_plot = flip_rotate_image(semantic_plot)
        semantic_pred = flip_rotate_image(semantic_pred)
        instance_gt = flip_rotate_image(instance_gt)
        instance_pred = flip_rotate_image(instance_pred)
        title_name = "bevpanformer pred"
    
    semantic = np.hstack([semantic_plot, semantic_pred])
    instance = np.hstack([instance_gt, instance_pred])
    final_img = np.vstack([semantic, instance])
    fig = plt.figure(figsize=(6, 6),dpi=300)
    ax = plt.subplot(2,2,1)
    plt.imshow(semantic_plot)
    plt.axis('off')
    plt.title("ground truth")
    # ax.set_ylabel("gt", fontsize=16)
    plt.draw()

    ax = plt.subplot(2,2,2)
    plt.imshow(semantic_pred)
    plt.axis('off')
    plt.title(title_name)
    # ax.set_ylabel("pred", fontsize=16)
    plt.draw()

    ax = plt.subplot(2,2,3)
    plt.imshow(instance_gt)
    plt.axis('off')
    plt.title("ground truth")
    # ax.set_ylabel("pred", fontsize=16)
    plt.draw()

    ax = plt.subplot(2,2,4)
    plt.imshow(instance_pred)
    plt.axis('off')
    plt.title(title_name)
    # ax.set_ylabel("pred", fontsize=16)
    plt.draw()

    plt.savefig("try_{}.jpg".format(i))

    return final_img


INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])