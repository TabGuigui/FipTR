# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xingtai Gui
# ---------------------------------------------
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init, constant_init

@ATTENTION.register_module()
class DeformableSelfAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=1,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_bev_queue = num_bev_queue # 只需要一帧
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
    def forward(self,
                query,
                value=None, # prev_bev
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_2d=None,
                predict_flow=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        
        if query_pos is not None:
            query = query + query_pos
            # bs n embed
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        if identity is None:
            identity = query
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs, num_value, 
                              self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, 1, self.num_levels, self.num_points, 2)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, 1, self.num_levels * self.num_points 
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, 1, self.num_levels, self.num_points
        )

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        reference_points = ref_2d
        assert reference_points.shape[-1] == 2

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
            
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)


        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

@ATTENTION.register_module()
class FlowGuidedSelfAttentionV2(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=1,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 bev_h = 200):
        super().__init__(init_cfg)
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_bev_queue = num_bev_queue # 只需要一帧
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims + 2, num_heads * num_levels * num_points * 2)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

        self.range = bev_h

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
    def forward(self,
                query,
                value=None, # prev_bev
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_2d=None,
                predict_flow=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        
        if query_pos is not None:
            query = query + query_pos
            # bs n embed
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        if identity is None:
            identity = query
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs, num_value, 
                              self.num_heads, -1)
        
        sample_query = torch.cat((query, predict_flow.flatten(2).permute(0, 2, 1)), dim = 2)
        sampling_offsets = self.sampling_offsets(sample_query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, 1, self.num_levels, self.num_points, 2)

        # rebuttal
        # sampling_offsets = predict_flow.flatten(2).permute(0, 2, 1)[:, :, None, None, None, None, :]
        # sampling_offsets = sampling_offsets.repeat(1, 1, self.num_heads, 1, self.num_levels, self.num_points, 1)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, 1, self.num_levels * self.num_points 
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, 1, self.num_levels, self.num_points
        )

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        ref_2d = ref_2d.to(predict_flow.device)
        reference_points = ref_2d
        assert reference_points.shape[-1] == 2

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
            
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)


        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        mean_offset = sampling_offsets.mean(axis = (2, 3, 4))

        return self.dropout(output) + identity, mean_offset