










# [ LidarDataset ] prepare_train_data(before pipeline ): 
{
    'sample_idx': '000002', 
    'pts_filename': '../mmdetection3d/data/lidar/lidar/000002.bin', 
    'ann_info': {
        'gt_bboxes_3d': LiDARInstance3DBoxes(tensor([[ -6.8640, -18.4916,  -0.1188,  18.7951,   4.0105,   4.7831,   1.5960]])), 
        'gt_labels_3d': array([0]), 
        'gt_names': array(['truck'], dtype='<U5')
        }, 
    'img_fields': [], 
    'bbox3d_fields': [], 
    'pts_mask_fields': [], 
    'pts_seg_fields': [], 
    'bbox_fields': [], 
    'mask_fields': [], 
    'seg_fields': [], 
    'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
    'box_mode_3d': <Box3DMode.LIDAR: 0>
}

# [ LidarDataset ] prepare_train_data(after pipeline ): 
{
    'img_metas': DataContainer({
        'flip': True, 
        'pcd_horizontal_flip': True, 
        'pcd_vertical_flip': False, 
        'box_mode_3d': <Box3DMode.LIDAR: 0>, 
        'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'pcd_trans': array([0., 0., 0.]), 
        'sample_idx': '000002', 
        'pcd_scale_factor': 1.0480002838856066, 
        'pcd_rotation': tensor([
            [ 0.9292,  0.3695,  0.0000],
            [-0.3695,  0.9292,  0.0000],
            [ 0.0000,  0.0000,  1.0000]]), 
        'pcd_rotation_angle': 0.3785173555035628, 
        'pts_filename': '../mmdetection3d/data/lidar/lidar/000002.bin', 
        'transformation_3d_flow': ['HF', 'R', 'S', 'T']}), 
    'points': DataContainer(tensor([
        [ 1.7645e-02,  1.5410e-01,  0.0000e+00,  1.5625e-01],
        [ 1.5049e+01, -4.3457e+00, -1.3705e+00,  4.9219e-01],
        [ 1.3515e+00, -1.8593e+01, -1.3036e+00,  3.9844e-01],
        ...,
        [ 5.0071e-02,  1.6000e-01, -2.9264e-03,  1.3281e-01],
        [ 2.6243e-02,  1.4852e-01, -5.2668e-03,  1.9141e-01],
        [ 5.6291e-02,  1.1736e-01, -3.2452e-02,  6.0938e-01]])), 
    'gt_bboxes_3d': DataContainer(LiDARInstance3DBoxes(tensor([], size=(0, 7)))), 
    'gt_labels_3d': DataContainer(tensor([], dtype=torch.int64))
}


# after data pipeline
{
    'img_metas': [
        DataContainer({
            'flip': False, 
            'pcd_horizontal_flip': False, 
            'pcd_vertical_flip': False, 
            'box_mode_3d': <Box3DMode.LIDAR: 0>, 
            'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
            'pcd_trans': array([0., 0., 0.]), 
            'sample_idx': '000027', 
            'pcd_scale_factor': 1.0, 
            'pcd_rotation': tensor([
                [1., 0., 0.],
                [-0., 1., 0.],
                [0., 0., 1.]]), 
            'pcd_rotation_angle': 0.0, 
            'pts_filename': '../mmdetection3d/data/lidar/lidar/000027.bin', 
            'transformation_3d_flow': ['R', 'S', 'T']})
        ], 
    'points': [
        DataContainer(tensor([
            [ 0.1490, -0.0198, -0.0547,  0.4844],
            [ 0.1398,  0.0060, -0.0482,  0.2031],
            [ 0.1630, -0.0142, -0.0532,  0.4961],
            ...,
            [ 0.1420,  0.0072,  0.0225,  0.1953],
            [ 0.1767, -0.0142,  0.0313,  0.5078],
            [ 0.1563,  0.0150,  0.0305,  0.1289]]))
        ]
}





[
    {
        'name': 'truck', 
        'path': 'lidar_gt_database/000001_truck_0.bin', 
        'idx': '000001', 
        'gt_idx': 0, 
        'box3d_lidar': array([ -7.468524, -16.038107, -0.07485 , 17.757347,4.748038,4.854287,1.555469], dtype=float32), 
        'num_points_in_gt': 1533,
        'difficulty': 0, 
        'group_id': 0,
    }, 
    {
        'name': 'truck', 
        'path': 'lidar_gt_database/000002_truck_0.bin', 
        'idx': '000002', 
        'gt_idx': 0, 
        'box3d_lidar': array([ -6.86402 , -18.491571,  -0.118824, 18.795055,4.010542,4.783063,1.596034], dtype=float32), 
        'num_points_in_gt': 1257, 
        'difficulty': 0, 
        'group_id': 1,
    },
]

[
    {
        'sample_idx': '000024', 
        'pts_filename': '../mmdetection3d/data/lidar/lidar/000024.bin', 
        'ann_info': {
            'gt_bboxes_3d': LiDARInstance3DBoxes(tensor([
                [-13.6857,  18.4922,   1.0297,  19.1466,   5.4672,   4.6370,  -1.5707],
                [ -7.8207, -17.2875,  -1.1632,  17.1751,   4.2365,   4.5272,  -1.5513]])), 
            'gt_labels_3d': array([0, 0]), 
            'gt_names': array(['truck', 'truck'], dtype='<U5')}, 
        'img_fields': [], 
        'bbox3d_fields': ['gt_bboxes_3d'], 
        'pts_mask_fields': [], 
        'pts_seg_fields': [], 
        'bbox_fields': [], 
        'mask_fields': [], 
        'seg_fields': [], 
        'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'box_mode_3d': <Box3DMode.LIDAR: 0>, 
        'points': LiDARPoints(tensor([
            [ 0.1496, -0.0146, -0.0547,  0.3555],
            [ 0.1508,  0.0118, -0.0521,  0.2031],
            [ 0.1444, -0.0076, -0.0470,  0.4492],
            ...,
            [ 0.1535,  0.0132,  0.0244,  0.2109],
            [ 0.1653, -0.0074,  0.0292,  0.4844],
            [ 0.1869,  0.0246,  0.0366,  0.1172]])), 
        'gt_bboxes_3d': LiDARInstance3DBoxes(tensor([
            [-13.6857,  18.4922,   1.0297,  19.1466,   5.4672,   4.6370,  -1.5707],
            [ -7.8207, -17.2875,  -1.1632,  17.1751,   4.2365,   4.5272,  -1.5513]])), 
        'gt_labels_3d': array([0, 0])
    }
]

[
    {
        "sample_idx": info["point_cloud"]["idx"],
        "pts_filename": info["point_cloud"]["path"],
        "ann_info": {
            "gt_bboxes_3d": LiDARInstance3DBoxes([ N ]),
            "gt_labels_3d": [ N ], # 数自label，从0开始，未知的为-1
            "gt_names": [  N  ], # label对应的class name
        },
        "box_type_3d": LiDARInstance3DBoxes
        "box_mode_3d": Box3DMode.LIDAR
        "img_fields": [],
        "bbox3d_fields": [],
        "pts_mask_fields": [],
        "pts_seg_fields": [],
        "bbox_fields": [],
        "mask_fields": [],
        "seg_fields": [],
    }
]


{
    "sample_idx": "idx",  # 文件名
    "pts_filename": "point_cloud path",  # 点云文件绝对路径
    "ann_info": {
        "name": [],  # class name list
        "location": [],  # location list [x,y,z]
        "dimensions": [],  # dimensions list [l,w,h]
        "rotation_y": [],  # rotation_y list [yaw]
    },
}

# 网络的输出
[
    {
        "boxes_3d": LiDARInstance3DBoxes(tensor([], size=(0, 7))),
        "scores_3d": tensor([]),
        "labels_3d": tensor([], dtype=torch.int64),
    },
]

# eval format后的结果
[
    {
        "name": array([], dtype=float64),
        "location": array([], shape=(0, 3), dtype=float64),
        "dimensions": array([], shape=(0, 3), dtype=float64),
        "rotation_y": array([], dtype=float64),
        "score": array([], dtype=float64),
        "sample_idx": array([], dtype=int64),
    },
]
# dataset ground truth format
[
    {
        "location": array([[-7.64003, -14.238812, -0.088559]]),
        "dimensions": array([[4.512487, 17.733538, 4.799274]]),
        "rotation_y": array([0.0]),
        "name": array(["truck"], dtype="<U5"),
        "num_points_in_gt": array([2120], dtype=int32),
    },
    {
        "location": array([[-7.99298, 18.761018, 1.615026], [-16.4247, -29.110842, -0.596363]]),
        "dimensions": array([[18.629422, 4.33807, 4.879027], [19.052341, 6.128458, 4.585352]]),
        "rotation_y": array([-1.618538, 0.0]),
        "name": array(["truck", "truck"], dtype="<U5"),
        "num_points_in_gt": array([447, 463], dtype=int32),
    },
]

# lidar data sample

{
    'img_metas': DataContainer({
        'flip': True, 
        'pcd_horizontal_flip': True, 
        'pcd_vertical_flip': False, 
        'box_mode_3d': <Box3DMode.LIDAR: 0>, 
        'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'pcd_trans': array([0., 0., 0.]), 
        'sample_idx': '000012', 
        'pcd_scale_factor': 0.9593832114457854, 
        'pcd_rotation': tensor([[ 0.8309, -0.5565,  0.0000],
            [ 0.5565,  0.8309,  0.0000],
            [ 0.0000,  0.0000,  1.0000]]), 
        'pcd_rotation_angle': -0.5901586818599815, 
        'pts_filename': '../mmdetection3d/data/lidar/lidar/000012.bin', 
        'transformation_3d_flow': ['HF', 'R', 'S', 'T'],
        }), 
    'points': DataContainer(tensor([[ 1.7550, -3.2863, -0.7242,  0.6094],
        [ 0.1797, -0.0200, -0.0351,  0.5352],
        [ 1.7838, -3.1605, -0.1902,  0.4883],
        ...,
        [ 1.8791,  6.5414, -1.9516,  0.3828],
        [ 0.0457, -5.8602, -1.5703,  0.5000],
        [ 0.9308, -4.5193,  0.4850,  0.4922]])), 
    'gt_bboxes_3d': DataContainer(LiDARInstance3DBoxes(tensor([[16.7282, 25.5388, -1.6029,  3.3260,  2.5691,  1.0573, -0.1212]]))), 
    'gt_labels_3d': DataContainer(tensor([0])),
},

# kitti data sample
{   'img_metas': DataContainer({
        'lidar2img': array([[ 6.09695435e+02, -7.21421631e+02, -1.25125790e+00,-1.23041824e+02],
            [ 1.80384201e+02,  7.64479828e+00, -7.19651550e+02,-1.01016693e+02],
            [ 9.99945343e-01,  1.24365499e-04,  1.04513029e-02,-2.69386917e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]], dtype=float32), 
        'flip': True, 
        'pcd_horizontal_flip': True, 
        'pcd_vertical_flip': False, 
        'box_mode_3d': <Box3DMode.LIDAR: 0>, 
        'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'pcd_trans': array([0., 0., 0.]), 
        'sample_idx': 2714, 
        'pcd_scale_factor': 1.0181611778807427, 
        'pcd_rotation': tensor([[ 0.7583,  0.6520,  0.0000],
            [-0.6520,  0.7583,  0.0000],
            [ 0.0000,  0.0000,  1.0000]]), 
        'pcd_rotation_angle': 0.7101589688208476, 
        'pts_filename': 'data/kitti/training/velodyne_reduced/002714.bin', 
        'transformation_3d_flow': ['HF', 'R', 'S', 'T']}), 
    'points': DataContainer(tensor([[ 4.2802,  5.0511, -1.6739,  0.2700],
        [12.9197,  2.1790, -1.5639,  0.2900],
        [11.1743,  4.6073, -1.5639,  0.1800],
        ...,
        [ 8.6005,  4.5226, -1.5965,  0.3200],
        [12.0629,  8.5899, -1.5303,  0.3800],
        [19.2174,  5.9113, -1.4081,  0.2800]])), 
    'gt_bboxes_3d': DataContainer(LiDARInstance3DBoxes(tensor([[40.0980,  5.8109, -1.1493,  3.0850,  1.8531,  1.4967, -2.4122],
        [65.4660, 25.5541, -0.4611,  4.0014,  1.7105,  1.4458,  0.7210],
        [23.0288, 18.8765, -1.3363,  4.5817,  1.6596,  1.6494,  0.7010],
        [18.5333, 15.6278, -1.5416,  3.7367,  1.6596,  1.5680,  0.7010],
        [40.2288, 28.5367, -1.7855,  3.9708,  1.6596,  1.7105, -2.5622],
        [29.5159, 24.4872, -1.6644,  4.5512,  1.6189,  1.4356, -2.4422],
        [46.8427,  4.2325, -2.4018,  4.3475,  1.8836,  1.5578, -0.6190],
        [36.1716, 17.4429, -1.4047,  3.9810,  1.6494,  1.4662, -0.6690],
        [25.5931, 22.2855, -1.5570,  3.6756,  1.7003,  1.5476,  0.7110],
        [24.6503, 25.9523, -1.1069,  4.5206,  1.7411,  1.5272, -2.4522],
        [18.8061, 10.6254, -1.6028,  3.1665,  1.4560,  1.4967, -2.4322],
        [ 7.7290, 28.5813, -2.0430,  3.4210,  1.6494,  1.4967,  0.6010],
        [18.8304,  1.7574, -1.6835,  3.7265,  1.6087,  1.4560, -2.2622],
        [13.7637, 17.9593, -1.5743,  4.2559,  1.8938,  1.5069,  2.3310]]))), 
    'gt_labels_3d': DataContainer(tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
}


