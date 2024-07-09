'''
author tabguigui

base exp : FipTR with BEVDet4D, backbone is swin-T 

59.0 36.6 53.3 33.6
'''
_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

find_unused_parameters = False
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
sync_bn = False

# Only with dynamic obstacles referring to FIERY
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# Image-view augmentation
data_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'rot_lim': (-5.4, 5.4),
    'H': 900, 'W': 1600,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'crop_h': (0.0, 0.0),
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
}
bev_aug_params = {
    'rot_range': [-0.3925, 0.3925],
    'scale_range': [0.95, 1.05],
    'trans_std': [0, 0, 0],
    'hflip': 0.5,
    'vflip': 0.5,
}
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}
motion_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}

receptive_field = 3
future_frames = 4
future_layer = 4

_dim_=256
bev_h_=200
bev_w_=200
_ffn_dim_ = _dim_*2
_pos_dim_ = _dim_//2
model = dict(
    type='FIPTR_LSS',
    img_backbone=dict(
        type='SwinTransformer',
        pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS', in_channels=384+768, inverse=True,),
    transformer=dict(
        type='TransformerLSS',
        grid_conf=grid_conf,
        input_dim=data_aug_conf['final_dim'],
        numC_input=512,
        numC_Trans=64,
    ),
    temporal_model=dict(
        type='Temporal3DConvModel',
        receptive_field=receptive_field,
        input_egopose=True,
        in_channels=64,
        input_shape=(128, 128),
        with_skip_connect=True,
        grid_conf=grid_conf
    ),
    pts_bbox_head=dict(
        type='FIPTR_LSS_TIMESPECIFICMASKQUERY',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=500,
        num_classes=7,
        in_channels=64,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        future_frame=future_frames,
        grid_conf=grid_conf,
        motion_grid_conf=motion_grid_conf,
        future_layer=future_layer,
        future_decoder=dict(
            type="FutureDecoder",
            num_layers=future_frames * future_layer,
            transformerlayers=dict(
                type='SampleMeanDecoderLayer',
                attn_cfgs=[
                        dict(
                            type='FlowGuidedSelfAttentionV2',
                            embed_dims=_dim_,
                            num_levels=1),],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            ),
        transformer=dict(
            type='PerceptionTransformer_LSS',
            # embed_dims=256,
            embed_dims=_dim_,
            decoder=dict(  # DetrTransformerDecoder
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=
                        [
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
                                     ),
        ),
        bbox_coder=dict( # 训练的时候用这个coder在2080上会爆显存
            type='NMSMultiSegFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=7,
            seg_threshold=0.35,
            mask_threshold=0.3),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_mask=dict(
            type='SegFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=10.0
        ),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0
        ),
        ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        det_assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        mask_assigner=dict( # TODO 为了增加mask的assigner
            type='HungarianAssigner3DStrongMask',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            mask_cost=dict(type='DiceCost', weight=1.0, pred_act=True, naive_dice=True),
            pc_range=point_cloud_range))),
)

train_pipeline = [
    # load image and apply image-view augmentation
    dict(type='LoadMultiViewImageFromFiles_MTL', using_ego=True, temporal_consist=True,
         is_train=True, data_aug_conf=data_aug_conf),
    # load 3D bounding boxes & bev-semantic-maps
    dict(type='LoadAnnotations3D_MTL', with_bbox_3d=True,
         with_label_3d=True, with_instance_tokens=True),
    # bev-augmentations
    dict(
        type='MTLGlobalRotScaleTrans',
        rot_range=bev_aug_params['rot_range'],
        scale_ratio_range=bev_aug_params['scale_range'],
        translation_std=bev_aug_params['trans_std'],
        update_img2lidar=True),
    dict(
        type='MTLRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=bev_aug_params['hflip'],
        flip_ratio_bev_vertical=bev_aug_params['vflip'],
        update_img2lidar=True),

    # convert motion labels
    dict(type='ConvertMotionLabelsFistr255', grid_conf=motion_grid_conf, only_vehicle=True, pcd_range=point_cloud_range),

    # bundle & collect
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',  'future_egomotions', 'aug_transform', 'img_is_valid',
               'motion_segmentation', 'motion_instance','instance_flow', 'has_invalid_frame', 'gt_masks', 'gt_backward_flow'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'lidar2ego_rots', 'lidar2ego_trans',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_info')),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_MTL',
         using_ego=True, data_aug_conf=data_aug_conf),
    dict(type='LoadAnnotations3D_MTL', with_bbox_3d=True,
         with_label_3d=True, with_instance_tokens=True),
    dict(type='ConvertMotionLabelsFistr', grid_conf=motion_grid_conf, only_vehicle=True, train_mode = False),
    # filter objects
    # dict(type='TemporalObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='TemporalObjectNameFilter', classes=class_names),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'future_egomotions', 'motion_segmentation',
                      'motion_instance', 'has_invalid_frame', 'img_is_valid', 'instance_flow', 'gt_masks', 'gt_backward_flow',  'instance_centerness', 'instance_offset'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                           'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'img_info', 'lidar2ego_rots', 'lidar2ego_trans',)),
        ],
    ),
]

dataset_type = 'MTLEgoNuScenesDataset'
data_root = 'data/nuscenes/'
data_info_path = 'data/nuscenes/'
input_modality = dict(
    use_camera=True,
    use_lidar=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    prototype='lift-splat-shoot',
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # train=dict(
        # type='CBGSDataset',
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_path + 'nuscenes_infos_temporal_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            receptive_field=receptive_field,
            future_frames=future_frames,
            grid_conf=grid_conf,
            modality=input_modality,
            box_type_3d='LiDAR'
),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        ann_file=data_info_path + 'nuscenes_infos_temporal_val.pkl',
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        ann_file=data_info_path + 'nuscenes_infos_temporal_val.pkl',
        modality=input_modality,),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
evaluation = dict(interval=999, pipeline=test_pipeline)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)