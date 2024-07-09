import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np

from mmdet3d.core import xywhr2xyxyr
from mmcv.ops import nms_bev

@BBOX_CODERS.register_module()
class NMSSegFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 seg_threshold=0.2,
                 mask_threshold=0.4):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.seg_threshold = seg_threshold
        self.mask_threshold = mask_threshold

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds, seg_masks):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]


        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            

            # seg post

            seg_scores, labels = cls_scores.max(-1)
            det_indexs = seg_scores > self.seg_threshold
            # det_scores, det_indexs = cls_scores.view(-1).topk(max_num) # TODO 调整
            # det_indexs = det_indexs // self.num_classes
            seg_masks = seg_masks[det_indexs]
            if seg_masks.shape[0] > 0:
            # kept = det_scores > self.seg_threshold
            # seg_masks = seg_masks[kept]
                bg_masks = seg_masks.sigmoid().max(0).values > self.mask_threshold
                cur_masks = seg_masks.flatten(1)
                instance = get_ids_area(cur_masks)
                instance *= bg_masks
                segmentation = (instance>0).to(torch.int16)
            else:
                instance = torch.zeros((200, 200), dtype=torch.long, device=seg_masks.device)
                segmentation = torch.zeros((200, 200), dtype=torch.long, device=seg_masks.device)

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'instance': instance,
                'segmentation': segmentation
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # final dec layer 4 1 150 7
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # 4 1 150 10
        all_seg_masks = preds_dicts["all_seg_masks"][-1] # 4 1 150 200 200 
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_seg_masks[i]))
        return predictions_list


@BBOX_CODERS.register_module()
class NMSMultiSegFreeCoder(NMSSegFreeCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 refine=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.mask_threshold_t = self.mask_threshold
        self.mask_threshold_time = []
        for i in range(5):
            self.mask_threshold_time.append(self.mask_threshold_t)
            self.mask_threshold_t *= 1.1
        self.refine = refine
    def decode_single(self, cls_scores, bbox_preds, seg_masks):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]


        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]


            # seg_scores, _ = cls_scores.max(-1)
            # det_indexs = seg_scores > self.seg_threshold
            # seg_masks = seg_masks[det_indexs]
            # if seg_masks.shape[0] == 0:
            #     instance = torch.zeros(seg_masks.shape[1:], device = seg_masks.device, dtype=torch.long)
            #     seg_out = torch.zeros(seg_masks.shape[1:], device = seg_masks.device)
            # else:
            #     seg_masks_sigomid = seg_masks.sigmoid() # q t h w 
            #     seg_masks_sigomid = seg_scores[det_indexs][:, None, None, None] * seg_masks_sigomid
            #     pred_seg_scores = seg_masks_sigomid.max(0)[0]
            #     if self.refine:
            #         seg_out = []
            #         for i in range(pred_seg_scores.shape[0]):
            #             seg_out.append(pred_seg_scores[i]>self.mask_threshold_time[i])
            #         seg_out = torch.stack(seg_out).long()
            #     else:
            #         seg_out = (pred_seg_scores > self.mask_threshold).long() # t h w
            #     instance = predict_instance_segmentation_and_trajectories(seg_out, seg_masks_sigomid)
            
            seg_scores = cls_scores.max(-1)[0]
            seg_masks_sigomid = seg_masks.sigmoid() # q t h w 
            seg_masks_sigomid = seg_scores[:, None, None, None] * seg_masks_sigomid
            pred_seg_scores = seg_masks_sigomid.max(0)[0]
            if self.refine:
                seg_out = []
                for i in range(pred_seg_scores.shape[0]):
                    seg_out.append(pred_seg_scores[i]>self.mask_threshold_time[i])
                seg_out = torch.stack(seg_out).long()
            else:
                seg_out = (pred_seg_scores > self.mask_threshold).long() # t h w
            instance = predict_instance_segmentation_and_trajectories(seg_out, seg_masks_sigomid)

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'instance': instance,
                'segmentation': seg_out
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # final dec layer 4 1 150 7
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # 4 1 150 10
        all_seg_masks = preds_dicts["all_seg_masks"].permute(1, 2, 0, 3, 4) # 5 1 150 200 200 
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_seg_masks[:, :, i]))
        return predictions_list


@BBOX_CODERS.register_module()
class NMSUniFreeCoder(NMSMultiSegFreeCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def decode_single(self, cls_scores, bbox_preds, seg_masks):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]


        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]


            obj_scores, labels = cls_scores.max(-1)
            det_indexs = obj_scores > self.seg_threshold
            seg_masks = seg_masks[det_indexs]
            obj_scores = obj_scores[det_indexs]
            if seg_masks.shape[0] == 0:
                instance = torch.zeros(seg_masks.shape[1:], device = seg_masks.device, dtype=torch.long)
                seg_out = torch.zeros(seg_masks.shape[1:], device = seg_masks.device)
            else:
                obj_scores = obj_scores[:, None, None, None]
                seg_masks_sigmoid = seg_masks.sigmoid()
                # compose the mask branch and det branch
                pred_ins_sigmoid = seg_masks_sigmoid * obj_scores
                pred_seg_scores = pred_ins_sigmoid.max(0)[0]
                seg_out = (pred_seg_scores > self.mask_threshold).long()
                instance = predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'instance': instance,
                'segmentation': seg_out
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict


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

def make_instance_seg_consecutive(instance_seg):
    # Make the indices of instance_seg consecutive
    unique_ids = torch.unique(instance_seg)  # include background
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg

def predict_instance_segmentation_and_trajectories(
                                    foreground_masks,
                                    ins_sigmoid,
                                    vehicles_id=1,
                                    ):
    if foreground_masks.dim() == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)  # [t, h, w]
    foreground_masks = foreground_masks == vehicles_id  # [t, h, w]  Only these places have foreground id
    
    argmax_ins = ins_sigmoid.argmax(dim=0)  # long, [t, h, w], ins_id starts from 0
    argmax_ins = argmax_ins + 1 # [t, h, w], ins_id starts from 1
    instance_seg = (argmax_ins * foreground_masks.float()).long()  # bg is 0, fg starts with 1

    # Make the indices of instance_seg consecutive
    instance_seg = make_instance_seg_consecutive(instance_seg).long()

    return instance_seg


def get_ids_area(masks,  dedup=False):
    '''
    Args:
        masks: Predicted masks (num_queries, h*w)
    '''


    m_id = masks.transpose(0, 1).softmax(-1) # 每个位置score最大的
    if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
        m_id = torch.zeros((200, 200), dtype=torch.long, device=m_id.device)
    else:
        m_id = (m_id.argmax(-1) + 1).view(200, 200)

    return m_id




@BBOX_CODERS.register_module()
class DETRTrack3DCoder(BaseBBoxCoder):
    """Bbox coder for DETR3D.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=0.2,
                 num_classes=7,
                 with_nms=False,
                 iou_thres=0.3):
        
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.with_nms = with_nms
        self.nms_iou_thres = iou_thres

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, 
                      track_scores, obj_idxes, with_mask=True, img_metas=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        max_num = min(cls_scores.size(0), self.max_num)

        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.num_classes

        _, bbox_index = track_scores.topk(max_num)
        
        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores
        
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = track_scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.with_nms:
            boxes_for_nms = xywhr2xyxyr(img_metas[0]['box_type_3d'](final_box_preds[:, :], 9).bev)
            nms_mask = boxes_for_nms.new_zeros(boxes_for_nms.shape[0]) > 0
            # print(self.nms_iou_thres)
            try:
                selected = nms_bev(
                    boxes_for_nms,
                    final_scores,
                    thresh=self.nms_iou_thres)
                nms_mask[selected] = True
            except:
                print('Error', boxes_for_nms, final_scores)
                nms_mask = boxes_for_nms.new_ones(boxes_for_nms.shape[0]) > 0
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            if not with_mask:
                mask = torch.ones_like(mask) > 0
            if self.with_nms:
                mask &= nms_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
                'bbox_index': bbox_index,
                'mask': mask
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, with_mask=True, img_metas=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
                Note: before sigmoid!
            bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['cls_scores']
        all_bbox_preds = preds_dicts['bbox_preds']
        track_scores = preds_dicts['track_scores']
        obj_idxes = preds_dicts['obj_idxes']
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        # bs size = 1
        predictions_list.append(self.decode_single(
            all_cls_scores, all_bbox_preds,
            track_scores, obj_idxes, with_mask, img_metas))
        #for i in range(batch_size):
        #    predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list
