import numpy as np
import torch
import imageio
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from ..fiptr.utils.instance import predict_instance_segmentation_and_trajectories as predict_instance_segmentation_and_trajectories_beverse
from ..fiptr.visualize.motion_visualisation import plot_instance_map

def visualize_motion(motion_targets, motion_preds, save_path, model = "fistr", index = None):
    if model == "beverse":
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
        gt_image = plot_motion(motion_labels, "beverse")
        pred_image = plot_motion(motion_preds, "beverse")
    elif model == "fistr":
        gt_image = plot_motion(motion_targets, "fistr")
        pred_image = plot_motion(motion_preds, "fistr")

    final_image = np.concatenate([pred_image, gt_image])
    final_image = flip_rotate_image(final_image)
    
    if save_path == None:
        save_path = "val_vis"
    os.makedirs(save_path, exist_ok=True)
    if index != None:
        imageio.imwrite('{}/motion_{}.png'.format(save_path, index), final_image)
    else:
        imageio.imwrite('{}/motion.png'.format(save_path), final_image)
    
def visualize_det(img_metas, bbox_results, vis_thresh, save_path, index = None):

    img_infos = img_metas['img_info'][-1]

    # prediction
    bbox_results = bbox_results["pts_bbox"]
    pred_lidar_boxes = bbox_results["boxes_3d"]
    pred_labels = bbox_results['labels_3d']
    pred_scores_3d = bbox_results["scores_3d"]
    pred_score_mask = pred_scores_3d > vis_thresh
    pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
    pred_labels = pred_labels[pred_score_mask]

    gt_bbox_color = (61, 102, 255)
    pred_bbox_color = (241, 101, 72)

    pred_imgs = {}
    for cam_type, img_info in img_infos.items():
        img_filename = img_info['data_path']
        img = imageio.imread(img_filename)

        cam2lidar_rt = np.eye(4)
        cam2lidar_rt[:3, :3] = img_info['sensor2lidar_rotation']
        cam2lidar_rt[:3, -1] = img_info['sensor2lidar_translation']
        lidar2cam_rt = np.linalg.inv(cam2lidar_rt)

        lidar2ego_rt = np.eye(4)
        lidar2ego_rt[:3, :3] = img_metas['lidar2ego_rots']
        lidar2ego_rt[:3, -1] = img_metas['lidar2ego_trans']
        ego2lidar_rt = np.linalg.inv(lidar2ego_rt)

        ego2cam_rt = lidar2cam_rt @ ego2lidar_rt
        intrinsic = img_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0],
                :intrinsic.shape[1]] = intrinsic
        lidar2img = (viewpad @ lidar2cam_rt)
        ego2img = (viewpad @ ego2cam_rt)

        if len(pred_lidar_boxes.tensor) == 0:
            img_with_pred = img
        else:
            img_with_pred = draw_lidar_bbox3d_on_img(
                        pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color,  thickness=2)

            img_with_pred = cv2.putText(img_with_pred, cam_type, (80, 100),cv2.FONT_HERSHEY_SIMPLEX , 3 , (255,255,255), thickness = 4)
        imageio.imwrite(
                '{}/det_pred_{}.png'.format(save_path, cam_type), img)
        pred_imgs[cam_type] = img_with_pred

    val_w = 6.4
    val_h = val_w / 16 * 9
    fig = plt.figure(figsize=(3 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w)
    gs = mpl.gridspec.GridSpec(2, 3, width_ratios=width_ratios)
    plt.subplots_adjust(wspace=0, hspace=0)
    vis_orders = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]

    for img_index, vis_cam_type in enumerate(vis_orders):
        vis_pred_img = pred_imgs[vis_cam_type]

        # prediction
        ax = plt.subplot(gs[(img_index // 3), img_index % 3])
        plt.imshow(vis_pred_img)
        plt.axis('off')
        plt.draw()
    
    plt.savefig(save_path + '/det_{}.png'.format(index), bbox_inches='tight', pad_inches=0)
        



def plot_motion(motion_preds, model):
    if model == "beverse" :
        consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories_beverse(motion_preds, compute_matched_centers=True)
    elif model == "fistr":
        consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(motion_preds, compute_matched_centers=True)
    
    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(consistent_instance_seg[0, 0].cpu().numpy(), instance_map)

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

def predict_instance_segmentation_and_trajectories(output, compute_matched_centers=False, ):
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

def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for instance_id, global_instance_id in instance_map.items()}

def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)

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