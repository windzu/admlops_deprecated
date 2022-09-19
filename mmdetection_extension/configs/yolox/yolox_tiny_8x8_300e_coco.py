_base_ = ["../_base_/schedules/schedule_1x.py", "../_base_/default_runtime.py"]


###########################################
########### datasets settings #############
###########################################
img_scale = (640, 640)  # height, width
data_root = "data/coco/"
dataset_type = "CocoDataset"

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.5, 1.5), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_train.json",
        img_prefix=data_root + "train/",
        pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)


test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val.json",
        img_prefix=data_root + "val/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val.json",
        img_prefix=data_root + "val/",
        pipeline=test_pipeline,
    ),
)

###########################################
############ models settings ##############
###########################################
model = dict(
    type="YOLOX",
    input_size=img_scale,
    # NOTE : yolox s is (15, 25)
    random_size_range=(10, 20),
    random_size_interval=10,
    # NOTE : yolox s is dict(deepen_factor=0.33, widen_factor=0.375)
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.375),
    # NOTE : yolox s is in_channels=[128, 256, 512], out_channels=128
    neck=dict(type="YOLOXPAFPN", in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    # NOTE : yolox s is in_channels=128, feat_channels=128
    bbox_head=dict(type="YOLOXHead", num_classes=80, in_channels=96, feat_channels=96),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)


##########
### 以下两个部分没改完


# canka


###########################################
########### schedules settings ############
###########################################
# optimizer






optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10

# learning policy
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
runner = dict(type="EpochBasedRunner", max_epochs=12)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)


###########################################
############ runtime settings #############
###########################################

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# optimizer
# default 8 gpu


# learning policy
lr_config = dict(
    _delete_=True,
    policy="YOLOX",
    warmup="exp",
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05,
)

runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", num_last_epochs=num_last_epochs, interval=interval, priority=48),
    dict(type="ExpMomentumEMAHook", resume_from=resume_from, momentum=0.0001, priority=49),
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best="auto",
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric="bbox",
)
log_config = dict(interval=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
