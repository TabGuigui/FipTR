from enum import unique
from unittest import result
import torch
import numpy as np
import cv2
import pdb

from ..dense_heads.map_head import calculate_birds_eye_view_parameters
from mmdet.datasets.builder import PIPELINES
from ..utils.instance import convert_instance_mask_to_center_and_offset_label, convert_instance_mask_to_center_and_offset_label_with_warper
from ..utils.instance import convert_instance_mask_to_center_and_offset_label_with_warper_forfistr
from ..utils.warper import FeatureWarper

import pdb


@PIPELINES.register_module()
class ConvertMotionLabels(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True):
        self.grid_conf = grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible

        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']

        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])

        self.warper = FeatureWarper(grid_conf=grid_conf)

    def __call__(self, results):
        # annotation_token ==> instance_id
        instance_map = {}

        # convert LiDAR bounding boxes to motion labels
        num_frame = len(results['gt_bboxes_3d'])
        all_gt_bboxes_3d = results['gt_bboxes_3d']
        all_gt_labels_3d = results['gt_labels_3d']
        all_instance_tokens = results['instance_tokens']
        all_vis_tokens = results['gt_vis_tokens']
        # 4x4 transformation matrix (if exist)
        bev_transform = results.get('aug_transform', None)

        segmentations = []
        instances = []

        # 对于 invalid frame: 所有 label 均为 255
        # 对于 valid frame: seg & instance 背景是 0，其它背景为255

        for frame_index in range(num_frame):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            vis_tokens = all_vis_tokens[frame_index]

            if gt_bboxes_3d is None:
                # for invalid samples
                segmentation = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
                instance = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
            else:
                # for valid samples
                segmentation = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                    gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    instance_tokens = instance_tokens[vehicle_mask]
                    vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    visible_mask = (vis_tokens != 1)
                    gt_bboxes_3d = gt_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    instance_tokens = instance_tokens[visible_mask]

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    bbox_corners = gt_bboxes_3d.corners[:, [
                        0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)

                    for index, instance_token in enumerate(instance_tokens):
                        if instance_token not in instance_map:
                            instance_map[instance_token] = len(
                                instance_map) + 1

                        # instance_id start from 1
                        instance_id = instance_map[instance_token]
                        poly_region = bbox_corners[index]
                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], instance_id)

            segmentations.append(segmentation)
            instances.append(instance)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long() # 5 200 200

        # generate heatmap & offset from segmentation & instance
        future_egomotions = results['future_egomotions'][- num_frame:]
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label_with_warper(
            instance_img=instances,
            future_egomotion=future_egomotions,
            num_instances=len(instance_map),
            ignore_index=self.ignore_index,
            subtract_egomotion=True,
            warper=self.warper,
            bev_transform=bev_transform,
        )

        invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
        instance_centerness[invalid_mask] = self.ignore_index

        # only keep detection labels for the current frame
        results['gt_bboxes_3d'] = all_gt_bboxes_3d[0]
        results['gt_labels_3d'] = all_gt_labels_3d[0]
        results['instance_tokens'] = all_instance_tokens[0]
        results['gt_valid_flag'] = results['gt_valid_flag'][0]

        results.update({
            'motion_segmentation': segmentations,
            'motion_instance': instances,
            'instance_centerness': instance_centerness,
            'instance_offset': instance_offset,
            'instance_flow': instance_flow,
        })

        return results



@PIPELINES.register_module()
class ConvertMotionLabelsFistr(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True, train_mode=True, pcd_range = None):
        self.grid_conf = grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible

        nusc_classes = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
            'motorcycle', 'bicycle', 'pedestrian', 'barrier', 'traffic_cone'
        ]
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']

        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])

        self.warper = FeatureWarper(grid_conf=grid_conf)
        self.train_mode = train_mode

        if pcd_range != None:
            self.filter_bev_range = True
            pcd_range = np.array(pcd_range, dtype=np.float32)
            self.bev_range = pcd_range[[0, 1, 3, 4]]
        else:
            self.filter_bev_range = False

    def __call__(self, results):
        # annotation_token ==> instance_id
        instance_map = {}
        gt_masks = {}

        # convert LiDAR bounding boxes to motion labels
        num_frame = len(results['gt_bboxes_3d'])
        all_gt_bboxes_3d = results['gt_bboxes_3d']
        all_gt_labels_3d = results['gt_labels_3d']
        all_instance_tokens = results['instance_tokens']
        all_vis_tokens = results['gt_vis_tokens']
        # 4x4 transformation matrix (if exist)
        bev_transform = results.get('aug_transform', None)

        segmentations = []
        instances = []

        # 对于 invalid frame: 所有 label 均为 255
        # 对于 valid frame: seg & instance 背景是 0，其它背景为255

        for frame_index in range(num_frame):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            vis_tokens = all_vis_tokens[frame_index]

            if gt_bboxes_3d is None:
                # for invalid samples
                segmentation = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
                instance = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
            else:
                # for valid samples
                segmentation = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                
                if self.filter_bev_range:
                    mask = gt_bboxes_3d.in_range_bev(self.bev_range)
                    gt_bboxes_3d = gt_bboxes_3d[mask]
                    gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
                    instance_tokens = instance_tokens[mask.numpy().astype(np.bool)]
                    vis_tokens = vis_tokens[mask.numpy().astype(np.bool)]
                    # limit rad to [-pi, pi]
                    gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                    gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    instance_tokens = instance_tokens[vehicle_mask]
                    vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    visible_mask = (vis_tokens != 1)
                    gt_bboxes_3d = gt_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    instance_tokens = instance_tokens[visible_mask]

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    bbox_corners = gt_bboxes_3d.corners[:, [
                        0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)

                    for index, instance_token in enumerate(instance_tokens):
                        if instance_token not in instance_map:
                            if frame_index == 0:
                                instance_map[instance_token] = len(instance_map) + 1
                                # 为每一个instance定义一个mask
                                gt_masks[instance_token] = np.zeros((num_frame, self.bev_dimension[1], self.bev_dimension[0]), dtype=np.float32) 
                            else:
                                instance_map[instance_token] = len(instance_map) + 1
                        
                        # instance_id start from 1
                        instance_id = instance_map[instance_token]
                        poly_region = bbox_corners[index]
                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], instance_id)
                        if frame_index == 0:
                            cv2.fillPoly(gt_masks[instance_token][frame_index], [poly_region], 1.0)

            segmentations.append(segmentation)
            instances.append(instance)

            if frame_index == 0:
                results['gt_bboxes_3d'] = gt_bboxes_3d
                results['gt_labels_3d'] = gt_labels_3d
                results['instance_tokens'] = instance_tokens

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()

        # generate heatmap & offset from segmentation & instance
        future_egomotions = results['future_egomotions'][- num_frame:]
        instance_centerness, instance_offset, _, _, _ = convert_instance_mask_to_center_and_offset_label_with_warper_forfistr(
            instance_img=instances,
            future_egomotion=future_egomotions,
            num_instances=len(instance_map),
            ignore_index=self.ignore_index,
            subtract_egomotion=True,
            warper=self.warper,
            bev_transform=bev_transform,
        )

        warped_semantic_seg = self.warper.cumulative_warp_features_reverse(
            segmentations.float().unsqueeze(0).unsqueeze(2), # 1 5 1 200 200
            future_egomotions.unsqueeze(0), # 1 7 6
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()[0, :, 0]

        warped_instance_seg = self.warper.cumulative_warp_features_reverse(
            instances.float().unsqueeze(0).unsqueeze(2), # 1 5 1 200 200
            future_egomotions.unsqueeze(0), # 1 7 6
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()[0, :, 0]

        # warp替换
        instances = warped_instance_seg
        segmentations = warped_semantic_seg

        instance_flow, instance_backward_flow = self.center_offset_flow(
            instances, 
            len(instance_map), 
            )
        
        # debug

        seq = segmentations.shape[0]
        for instance_token in gt_masks.keys():
            instance_id = instance_map[instance_token]
            for t in range(1, seq):
                mask = (instances[t] == instance_id).to(int)
                if mask.sum() > 0:
                    gt_masks[instance_token][t] = mask
                else:
                    gt_masks[instance_token][t] = gt_masks[instance_token][t-1]
        
        if gt_masks == {}:
            gt_masks = torch.from_numpy(np.zeros((0, num_frame, self.bev_dimension[1], self.bev_dimension[0]), dtype=np.float32))
        else:
            gt_masks = torch.from_numpy(np.stack(list(gt_masks.values()), axis = 0))

        results.update({
            'motion_segmentation': segmentations,
            'motion_instance': instances,
            'gt_masks': gt_masks,
            'instance_centerness': instance_centerness,
            'instance_offset': instance_offset,
            "instance_flow": instance_flow, # 5 * 2 * 200 * 200
            "gt_backward_flow": instance_backward_flow # 5 * 2 * 200 * 200
        })

        return results

    def center_offset_flow(self, instance_img, num_instances): # TODO 没有用ignore，静态地方就预测静态
        seq_len, h, w = instance_img.shape
        # future flow
        future_displacement_label = torch.zeros(seq_len, 2, h, w)
        # backward flow
        backward_flow = torch.zeros(seq_len, 2, h, w)

        # x is vertical displacement, y is horizontal displacement
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float),
                            torch.arange(w, dtype=torch.float))

        # iterate over all instances across this sequence
        for instance_id in range(1, num_instances+1):
            instance_id = int(instance_id)
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                instance_mask = (instance_img[t] == instance_id)
                if instance_mask.sum() == 0: 
                    if prev_mask is not None: # 用前一帧的的结果 如果前一帧出现了，补
                        instance_mask = prev_mask
                    # this instance is not in this frame
                    else:
                        prev_xc = None
                        prev_yc = None
                        prev_mask = None
                        continue
                # the Bird-Eye-View center of the instance
                xc = x[instance_mask].mean()
                yc = y[instance_mask].mean()

                if prev_xc is not None and instance_mask.sum() > 0:
                    delta_x = xc - prev_xc
                    delta_y = yc - prev_yc
                    future_displacement_label[t-1, 0, prev_mask] = delta_x
                    future_displacement_label[t-1, 1, prev_mask] = delta_y
                    backward_flow[t-1, 0, instance_mask] = -1 * delta_x
                    backward_flow[t-1, 1, instance_mask] = -1 * delta_y
                        
                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

        
        return future_displacement_label[:seq_len -1], backward_flow[:seq_len-1]


@PIPELINES.register_module()
class ConvertMotionLabelsFistr255(ConvertMotionLabelsFistr):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def center_offset_flow(self, instance_img, num_instances): # TODO 没有用ignore，静态地方就预测静态
        seq_len, h, w = instance_img.shape
        # future flow
        future_displacement_label = self.ignore_index * torch.ones(seq_len, 2, h, w)
        # backward flow
        backward_flow = self.ignore_index*torch.ones(seq_len, 2, h, w)

        # x is vertical displacement, y is horizontal displacement
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float),
                            torch.arange(w, dtype=torch.float))

        # iterate over all instances across this sequence
        for instance_id in range(1, num_instances+1):
            instance_id = int(instance_id)
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                instance_mask = (instance_img[t] == instance_id)
                if instance_mask.sum() == 0: 
                    if prev_mask is not None: # 用前一帧的的结果 如果前一帧出现了，补
                        instance_mask = prev_mask
                    # this instance is not in this frame
                    else:
                        prev_xc = None
                        prev_yc = None
                        prev_mask = None
                        continue
                # the Bird-Eye-View center of the instance
                xc = x[instance_mask].mean()
                yc = y[instance_mask].mean()

                if prev_xc is not None and instance_mask.sum() > 0:
                    delta_x = xc - prev_xc
                    delta_y = yc - prev_yc
                    future_displacement_label[t-1, 0, prev_mask] = delta_x
                    future_displacement_label[t-1, 1, prev_mask] = delta_y
                    backward_flow[t-1, 0, instance_mask] = -1 * delta_x
                    backward_flow[t-1, 1, instance_mask] = -1 * delta_y
                        
                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

        
        return future_displacement_label[:seq_len -1], backward_flow[:seq_len-1]
    