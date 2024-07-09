# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xingtai Gui
# ---------------------------------------------
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import torch
from mmcv.runner import force_fp32, auto_fp16

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FutureDecoder(TransformerLayerSequence):
    def __init__(self, *args, dataset_type='nuscenes',
                 **kwargs):

        super(FutureDecoder, self).__init__(*args, **kwargs)

        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        
    @auto_fp16()
    def forward(self,
                bev_query,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                prev_bev=None,
                predict_flow=None,
                **kwargs):
        output = bev_query
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev,
                predict_flow=predict_flow,
                **kwargs)

        return output
    
@TRANSFORMER_LAYER.register_module()
class FutureDecoderLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(FutureDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 4
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'ffn'])
    def forward(self,
                query,
                bev_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                ref_2d=None,
                bev_h=None,
                bev_w=None,
                prev_bev=None,
                **kwargs):
        query = query.unsqueeze(1) # nq bs c
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # nq bs c

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        for layer in self.operation_order:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    value = prev_bev, # nq bs c
                    identity = identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_padding_mask=query_key_padding_mask,
                    ref_2d=ref_2d,
                    spatial_shapes=torch.tensor(
                                [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                            **kwargs
                )
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        return query

@TRANSFORMER_LAYER.register_module()
class SampleMeanDecoderLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(SampleMeanDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 4
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'ffn'])
    def forward(self,
                query,
                bev_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                ref_2d=None,
                bev_h=None,
                bev_w=None,
                prev_bev=None,
                **kwargs):
        query = query.unsqueeze(1) # nq bs c
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # nq bs c

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        for layer in self.operation_order:
            if layer == 'self_attn':
                query, sample_mean = self.attentions[attn_index](
                    query,
                    value = prev_bev, # nq bs c
                    identity = identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_padding_mask=query_key_padding_mask,
                    ref_2d=ref_2d,
                    spatial_shapes=torch.tensor(
                                [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                            **kwargs
                )
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        return query,sample_mean