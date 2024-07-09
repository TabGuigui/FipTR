# ---------------------------------------------
#  Modified by Tab Gui
# unibev, temporal instance flow loss
# ---------------------------------------------


import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F 

from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.core import (multi_apply, multi_apply, reduce_mean, build_assigner)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import build_loss
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.utils import TORCH_VERSION, digit_version

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from .segmentation import BEV_Decoder_x1
from .map_head import BevFeatureSlicer
from .bev_encoder import BevEncode, CustomBottleneck
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

@HEADS.register_module()
class FIPTR_LSS_TIMESPECIFICMASKQUERY(DETRHead):

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 loss_mask=None,
                 loss_dice=None,
                 future_frame=0,
                 grid_conf=None,
                 motion_grid_conf=None,
                 out_channels=256,
                 bev_encode_block='BottleNeck',
                 bev_encode_depth=[2, 2, 2],
                 num_channels=None,
                 backbone_output_ids=None,
                 norm_cfg=dict(type='BN'),
                 bev_encoder_fpn_type='lssfpn',
                 out_with_activision=False,
                 future_decoder=None,
                 future_layer=None,
                 loss_flow=dict(type='SmoothL1Loss', reduction='mean', loss_weight=0.1),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.future_frame = future_frame
                

        super(FIPTR_LSS_TIMESPECIFICMASKQUERY, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        # bev module
        self.taskfeat_encoders = BevEncode(
                    numC_input=self.in_channels,
                    numC_output=out_channels,
                    num_channels=num_channels,
                    backbone_output_ids=backbone_output_ids,
                    num_layer=bev_encode_depth,
                    bev_encode_block=bev_encode_block,
                    norm_cfg=norm_cfg,
                    bev_encoder_fpn_type=bev_encoder_fpn_type,
                    out_with_activision=out_with_activision,
                )       

        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)  
        self.loss_flow = build_loss(loss_flow)          
        # for different assigner
        if self.train_cfg:
            det_assigner = self.train_cfg['det_assigner']
            mask_assigner = self.train_cfg["mask_assigner"]
            self.det_assigner = build_assigner(det_assigner)
            self.mask_assigner = build_assigner(mask_assigner)

        self.future_decoder = build_transformer_layer_sequence(
            future_decoder)
        
        self.ref_2d = self.future_decoder.get_reference_points(self.bev_h, self.bev_h, dim='2d', bs=1, 
                                                               dtype=self.query_embedding.weight.dtype)
        self.future_layer = future_layer 
        
    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

        dense_decoder = BEV_Decoder_x1(
            dim=self.embed_dims,
            blocks=[self.embed_dims, self.embed_dims],
        )
        self.bev_up_branches = _get_clones(dense_decoder, self.future_frame + 1)

        self.future_bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

        flow_conv = BEV_Decoder_x1(
            dim=self.embed_dims,
            blocks=[self.embed_dims//4, 2]
        )
        self.flow_conv_branches = _get_clones(flow_conv, self.future_frame)
    
        object2instance_mlp = MLP(self.embed_dims, self.embed_dims//2, self.embed_dims, num_layers=3)
        self.temporal_query = _get_clones(object2instance_mlp, self.future_frame + 1)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        self.future_decoder.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, bev_feat):
        
        # bev_embed = self.task_feat_cropper(bev_feat)
        bev_embed = self.taskfeat_encoders([bev_feat]) # 200 x 200

        dtype = bev_embed.dtype
        bs = bev_embed.shape[0]
        object_query_embeds = self.query_embedding.weight.to(dtype)

        # cur decoder
        bev_embed = bev_embed.flatten(2) # 1 256 2500
        outputs = self.transformer(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            )
        bev_embed, hs, init_reference, inter_references = outputs # nq bs c
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes) # layer bs nq 7
        outputs_coords = torch.stack(outputs_coords)

        # seg 
        # reshape to bev_h * bev_w
        bev_seg_embed = bev_embed.permute(1, 2, 0).view(bs, -1, self.bev_h, self.bev_w) # 1 256 200 200
        outputs_seg_masks = []
        predict_flows = []

        hs_ins = hs[-1]
        hs_current = self.temporal_query[0](hs_ins)
        # up to 200/200
        final_bev_embed = self.bev_up_branches[0](bev_seg_embed)
        seg_masks = torch.einsum('bqc,bchw->bqhw', hs_current, final_bev_embed)
        outputs_seg_masks.append(seg_masks)

        # future module
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=hs_current.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # 1 256 200 200
        sample_means = []
        for i in range(self.future_frame):
            future_bev_query = self.future_bev_embedding.weight.to(dtype) # 40000 256
            downsample_flow = self.flow_conv_branches[i](bev_seg_embed) # 1 2 200 200
            predict_flows.append(downsample_flow)
            
            for j in range(self.future_layer):
                trans_layer = self.future_decoder.layers[i * self.future_layer + j]
                future_bev_query, sample_mean = trans_layer( # bs nq c
                    future_bev_query, # 2500 256
                    bev_pos=bev_pos,
                    ref_2d=self.ref_2d,
                    bev_h=self.bev_h,
                    bev_w=self.bev_w,
                    prev_bev=bev_embed, # 2500, 1, 256
                    predict_flow=downsample_flow)
                if j != self.future_layer-1:
                    future_bev_query = future_bev_query.squeeze(0)
                else:
                    sample_mean=sample_mean.permute(0, 2, 1).view(bs, 2, self.bev_h, self.bev_w)
                    sample_means.append(sample_mean)
            # predict_flows.append(torch.mean(predict_flow, dim = 1))
            bev_seg_embed = future_bev_query.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w)
            final_bev_embed = self.bev_up_branches[i+1](bev_seg_embed)
            hs_current = self.temporal_query[i+1](hs_ins)
            seg_masks = torch.einsum('bqc,bchw->bqhw', hs_current, final_bev_embed)
            outputs_seg_masks.append(seg_masks)

            bev_embed = future_bev_query.permute(1, 0, 2)
        sample_means = torch.stack(sample_means, dim = 1)
        predict_flows = torch.stack(predict_flows, dim = 1)
        outputs_seg_masks = torch.stack(outputs_seg_masks) # future frame * bs * nq * 200 * 200
        outputs_seg_masks = outputs_seg_masks.permute(1, 2, 0, 3, 4)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_seg_masks': outputs_seg_masks,
            'predict_flows': predict_flows
        }
        # TODO 匹配策略应该是用当前帧的结果来分配

        return outs
    
    def _get_det_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.det_assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def _get_mask_target_single(self,
                           cls_score,
                           bbox_pred,
                           seg_masks,
                           gt_labels,
                           gt_bboxes,
                           gt_masks,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0) # num_query
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1] # attrs

        assign_result = self.mask_assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore, seg_masks, gt_masks)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds # query中哪些是pos
        neg_inds = sampling_result.neg_inds # query中哪些是neg

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # seg branch
        mask_targets = gt_masks.new_zeros((num_bboxes, self.future_frame + 1, gt_masks.shape[-2], gt_masks.shape[-1]))
        mask_weights = gt_masks.new_zeros((num_bboxes, self.future_frame + 1, gt_masks.shape[-2], gt_masks.shape[-1]))
        mask_weights[pos_inds] = 1.0
        mask_targets[pos_inds] = gt_masks[sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_weights,
                pos_inds, neg_inds)

    def get_mask_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    seg_masks_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list, # TODO 暂时不用mask做匹配，只需要返回mask list即可
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, mask_targets_list, mask_weight_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_mask_target_single, cls_scores_list, bbox_preds_list, seg_masks_list,
            gt_labels_list, gt_bboxes_list, gt_masks_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weight_list,
                num_total_pos, num_total_neg)
    
    def get_det_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_det_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    
    def loss_det_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_det_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_mask_single(self,
                    cls_scores,
                    bbox_preds,
                    seg_masks, # 5 1 150 200 200
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list, # 8 5 200 200
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        seg_masks_list = [seg_masks[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_mask_targets(cls_scores_list, bbox_preds_list, seg_masks_list,
                                           gt_bboxes_list, gt_labels_list, gt_masks_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0).to(seg_masks.device)
        mask_weights = torch.cat(mask_weights_list, 0).to(seg_masks.device)

    
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        
        # @Author Tab Gui
        h, w = seg_masks.shape[-2], seg_masks.shape[-1]
        seg_masks = seg_masks.reshape(-1, self.future_frame + 1, h, w)
        loss_masks = []
        loss_dices = []
        for i in range(self.future_frame + 1):
            loss_mask = self.loss_mask(
                seg_masks[:, i], mask_targets[:, i], mask_weights[:, i], avg_factor = num_total_pos * h * w)
            loss_dice = self.loss_dice(
                seg_masks[:, i], mask_targets[:, i], mask_weights[:, i, 0, 0], avg_factor = num_total_pos)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_masks.append(torch.nan_to_num(loss_mask))
                loss_dices.append(torch.nan_to_num(loss_dice))
            else:
                loss_masks.append(loss_mask)
                loss_dices.append(loss_dice)



        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_masks, loss_dices


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_masks_list,
             gt_flow,
             gt_bboxes_ignore=None,
             img_metas=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores'] # declayer * bs * n * 10
        all_bbox_preds = preds_dicts['all_bbox_preds'] # declayer * bs * n * 10
        all_seg_masks = preds_dicts["all_seg_masks"] #  future_frame * bs * n * 200 * 200

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)] # declayer * bs * (N * 9)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(1)] # 1 * bs * (N  * 5 * 200 * 200)
        all_gt_masks_list = [gt_mask.to(device) for gt_mask in gt_masks_list]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # losses_cls, losses_bbox, losses_mask, losses_dice = multi_apply(
        #     self.loss_single, all_cls_scores, all_bbox_preds, all_seg_masks,
        #     all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
        #     all_gt_bboxes_ignore_list)
        
        losses_cls, losses_bbox = multi_apply(
            self.loss_det_single, all_cls_scores[:num_dec_layers-1], all_bbox_preds[:num_dec_layers-1],
            all_gt_bboxes_list[:num_dec_layers-1], all_gt_labels_list[:num_dec_layers-1],
            all_gt_bboxes_ignore_list[:num_dec_layers-1])
        
        losses_cls_final, losses_bbox_final, losses_masks_final, losses_dices_final = self.loss_mask_single(
            all_cls_scores[num_dec_layers-1], all_bbox_preds[num_dec_layers-1], all_seg_masks,
            all_gt_bboxes_list[num_dec_layers-1], all_gt_labels_list[num_dec_layers-1], all_gt_masks_list,
            all_gt_bboxes_ignore_list[num_dec_layers-1])

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls_final
        loss_dict['loss_bbox'] = losses_bbox_final
        

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:],
                                           losses_bbox[:],):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        num_future = 0
        for i, (loss_mask_i, loss_dice_i) in enumerate( zip(losses_masks_final, losses_dices_final) ):
            loss_dict[f'f_{num_future}.loss_mask'] = loss_mask_i 
            loss_dict[f'f_{num_future}.loss_dice'] = loss_dice_i 
            num_future += 1

        # flow branch
        pred_flows = preds_dicts["predict_flows"]
        
        gt_flows = torch.flip(gt_flow, dims = [2])
        
        loss_flow = 0.0
        for i in range(num_future - 1):
            current_flow, current_gt = pred_flows[:, i], gt_flows[:, i]
            weight = (current_gt != 255).to(int)
            avg_num = weight.sum()
            if avg_num == 0:
                loss_flow += current_flow.abs().sum().float()* 0.0
            else:
                loss_flow += torch.nan_to_num(self.loss_flow(current_flow, current_gt, weight=weight, avg_factor=avg_num))
            
            neg_weight = (current_gt == 255).to(int)
            neg_avg_num = neg_weight.sum()
            current_gt[torch.where(current_gt == 255)] = 0
            loss_flow += self.loss_flow(current_flow, current_gt, weight=neg_weight, avg_factor=neg_avg_num)
        loss_dict[f'loss_flow'] = loss_flow

        return loss_dict
    

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            if "instance" in preds:
                instance = preds["instance"]
                segmentation = preds["segmentation"]
                ret_list.append([bboxes, scores, labels, segmentation, instance])
            else:
                ret_list.append([bboxes, scores, labels])
        return ret_list


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x