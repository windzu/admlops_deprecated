import os

mmlab_extension_path = os.environ["MMLAB_EXTENSION_PATH"]

_base_ = [
    mmlab_extension_path + "/mmdetection3d/configs/_base_/models/hv_pointpillars_secfpn_kitti.py",
    mmlab_extension_path + "/mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py",
    mmlab_extension_path + "/mmdetection3d/configs/_base_/schedules/cyclic_40e.py",
    mmlab_extension_path + "/mmdetection3d/configs/_base_/default_runtime.py",
]

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
model = dict(
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type="MaxIoUAssigner",
            iou_calculator=dict(type="BboxOverlapsNearest3D"),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
)
# dataset settings
dataset_type = "PointCloudDataset"
data_root = mmlab_extension_path + "/mmdetection3d/data/pointcloud/"  # change default data root path
class_names = ["truck"]  # class names add truck
# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "pointcloud_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(truck=5),
    ),
    classes=class_names,
    sample_groups=dict(truck=15),
)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectSample", db_sampler=db_sampler, use_ground_plane=True),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
eval_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]
data = dict(
    samples_per_gpu=2,  # change batch_size to match your GPU memory
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            type=dataset_type,
            pipeline=train_pipeline,
            pts_prefix="velodyne",
            data_root=data_root,
            ann_file=data_root + "pointcloud_infos_train.pkl",
            classes=class_names,
        )
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        pts_prefix="velodyne",
        data_root=data_root,
        ann_file=data_root + "pointcloud_infos_val.pkl",
        classes=class_names,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        pts_prefix="velodyne",
        data_root=data_root,
        ann_file=data_root + "pointcloud_infos_val.pkl",
        classes=class_names,
    ),
)

evaluation = dict(interval=1, pipeline=eval_pipeline)
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)


# finetune
## set load_from value and finetune from pretrained model
load_from = (
    mmlab_extension_path
    + "/mmdetection3d/checkpoints/pointpillars/"
    + "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
)
