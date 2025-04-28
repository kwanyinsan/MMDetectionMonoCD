# configs/mm_config.py

_base_ = [
    'mm_tools/coco_detection.py',
    'mm_tools/schedule_1x.py',
    'mm_tools/default_runtime.py',
    'mm_tools/centernet_tta.py'
]

dataset_type = 'CocoDataset'
data_root = 'mm_data/'

# model settings
model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='ResNet', depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    neck=dict(
        type='CTResNetNeck',
        in_channels=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True
    ),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100)
)

# pipelines (leave as-is from centernet_tta base)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3),
        mean=[0,0,0],
        std=[1,1,1],
        to_rgb=True,
        test_pad_mode=None
    ),
    dict(type='Resize', scale=(512,512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}, to_float32=True),
    dict(
        type='RandomCenterCropPad',
        ratios=None, border=None,
        mean=[0,0,0], std=[1,1,1],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id','img_path','ori_shape','img_shape','border')
    )
]

# -----------------------------------------------------------------------------
# Data loaders: point at mm_data, only 1 class
# -----------------------------------------------------------------------------
metainfo = {
    'classes': ('Car',),
    'palette': [[220, 20, 60]]
}

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/instances_train.json',
            data_prefix=dict(img='train/'),
            pipeline=train_pipeline,
            backend_args={{_base_.backend_args}},
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        backend_args={{_base_.backend_args}},
    )
)

test_dataloader = val_dataloader  # reuse val settings

# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox'
)
val_evaluator = test_evaluator

# optimizer settings remain unchanged
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

# epochs and LR schedule remain as in centernet_tta.py
max_epochs = 28
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0, end=max_epochs, by_epoch=True,
        # these steps are still in repeated‐dataset epochs
        milestones=[18, 24],
        gamma=0.1
    )
]
train_cfg = dict(max_epochs=max_epochs)

# NOTE: This auto_scale_lr is internal to the base schedule
auto_scale_lr = dict(base_batch_size=128)
