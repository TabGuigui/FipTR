import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_norm_layer
from mmdet3d.models import builder

import pdb
from collections import OrderedDict

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=dict(type='BN')):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        assert norm_cfg['type'] in ['BN', 'SyncBN']
        if norm_cfg['type'] == 'BN':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, numC_input, numC_output, num_layer=[2, 2, 2], num_channels=None,
                 backbone_output_ids=None,  norm_cfg=dict(type='BN'), out_with_activision=False,
                 bev_encode_block='BottleNeck', multiview_learning=False, feature_fuse_type='SUM',
                 bev_encoder_fpn_type='lssfpn'):
        super(BevEncode, self).__init__()

        # build downsample modules for multiview learning
        self.multiview_learning = multiview_learning
        if self.multiview_learning:
            downsample_conv_list = []
            for i in range(len(num_layer)-1):
                downsample_conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(numC_input, numC_input * 2**(i+1),
                                  kernel_size=3, stride=2**(i+1), padding=1, bias=False),
                        build_norm_layer(norm_cfg, numC_input *
                                         2**(i+1), postfix=0)[1],
                        nn.ReLU(inplace=True)))
            self.downsample_conv_list = nn.Sequential(*downsample_conv_list)
        self.feature_fuse_type = feature_fuse_type

        # build backbone
        assert len(num_layer) >= 3
        num_channels = [numC_input * 2 ** (i+1) for i in range(
            len(num_layer))] if num_channels is None else num_channels

        # default: [128, 256, 512]

        # 输出最后三层特征
        self.backbone_output_ids = range(len(
            num_layer) - 3, len(num_layer)) if backbone_output_ids is None else backbone_output_ids

        layers = []
        if bev_encode_block == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(curr_numC, num_channels[i]//4, stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        
        # [1/2, 1/4, 1/8]
        elif bev_encode_block == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(curr_numC, num_channels[i], stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                             for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False

        self.layers = nn.Sequential(*layers)

        # build neck
        self.bev_encoder_fpn_type = bev_encoder_fpn_type
        if self.bev_encoder_fpn_type == 'lssfpn':
            self.up1 = Up(num_channels[-1] + num_channels[-3],
                          numC_output * 2, scale_factor=4, norm_cfg=norm_cfg)
        elif self.bev_encoder_fpn_type == 'fpnv1':
            img_neck_cfg = dict(
                type='FPNv1',
                in_channels=num_channels[-3:],
                out_channels=numC_output*2,
                num_outs=1,
                start_level=0,
                out_ids=[0])
            self.up1 = builder.build_neck(img_neck_cfg)
        else:
            assert False
        assert norm_cfg['type'] in ['BN', 'SyncBN']

        if norm_cfg['type'] == 'BN':
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(numC_output * 2, numC_output,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(numC_output),
                nn.ReLU(inplace=True),
            )
        else:
            # 移除掉输出层的 linear conv, 使得输出为激活后的特征值
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(numC_output * 2, numC_output,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, numC_output, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

        self.out_with_activision = out_with_activision
        if not self.out_with_activision:
            self.up2.add_module('4', nn.Conv2d(
                numC_output, numC_output, kernel_size=1, padding=0))
            # self.up2.add_module('linear_output', nn.Conv2d(
            #     numC_output, numC_output, kernel_size=1, padding=0))

        self.fp16_enabled = False

    def forward(self, bev_feat_list):
        feats = []
        x_tmp = bev_feat_list[0]
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            # x_tmp = checkpoint.checkpoint(layer,x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
            if lid < (len(self.layers)-1) and self.multiview_learning:
                if self.feature_fuse_type == 'SUM':
                    bev_feat_from_img_view = bev_feat_list[lid + 1]
                    bev_feat_from_img_view = self.downsample_conv_list[lid](
                        bev_feat_from_img_view)
                    x_tmp = x_tmp + bev_feat_from_img_view
                else:
                    assert False

        if self.bev_encoder_fpn_type == 'lssfpn':
            res = self.up1(feats[-1], feats[-3])
        elif self.bev_encoder_fpn_type == 'fpnv1':
            res = self.up1(feats)
        else:
            assert False

        res = self.up2(res)

        return res

class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)

class CustomBottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
                    ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                                       nn.ReLU(inplace=True))),
                    # Second conv block
                    ('conv', bottleneck_conv),
                    ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
                    ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                                     nn.ReLU(inplace=True))),
                    # Regulariser
                    ('dropout', nn.Dropout2d(p=dropout)),
                ]
            )
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x