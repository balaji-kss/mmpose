_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=25, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        milestones=[15, 22], 
        gamma=0.1,
        by_epoch=True,
        verbose=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(192, 256), sigma=6.0)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        # init_cfg=dict(
        #     type='Pretrained',
        #     # checkpoint='https://download.openmmlab.com/mmpose/'
        #     # 'pretrain_models/hrnet_w32-36af842e.pth'
        #     checkpoint='/home/balaji/pose_estimation/mmpose/models/exp1/resume/epoch_50.pth'
        #     ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=42,
        deconv_out_channels=(32, 32),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(32, 32),
        conv_kernel_sizes=(3, 3),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoTumekeDataset'
data_mode = 'topdown'
data_root = '/data/balaji/coco'
# load_from = '/data/balaji/models/exp10/epoch_50.pth'
load_from = '/data/balaji/models/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth'
occl_dir = '/data/balaji/occl_patches'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding = 1.25),
    dict(type='RandomFlip', direction='horizontal', prob = 0.5),
    dict(type='RandomHalfBody', prob = 0.1),
    dict(
        type='RandomBBoxTransform', 
        shift_prob = 0.3,
        scale_prob = 0.8,
        rotate_prob = 0.6,
        shift_factor = 0.1, 
        scale_factor = (0.8, 1.2),
        rotate_factor = 40,
    ),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp = True),
    dict(type='Augmentation', occl_dir=occl_dir),
    dict(type='GenerateTarget', encoder=codec, use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding = 1.25),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp = True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=56,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/wholebody_kpt_jsons/train_tumeke_v3.json',#train_tumeke_sample_42.json', #train_tumeke_sample_tiny.json
        data_prefix=dict(img='images/train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/wholebody_kpt_jsons/val_tumeke_v3.json',
        bbox_file=None,
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoTumekeMetric',
    # ann_file=data_root + 'annotations/wholebody_kpt_jsons/test.json',
    ann_file=None,
    use_area=False)
test_evaluator = val_evaluator

# visualizer = dict(vis_backends=[
#     dict(type='LocalVisBackend'),
#     dict(type='TensorboardVisBackend'),
# ])
