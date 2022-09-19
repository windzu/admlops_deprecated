# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path

import mmcv
import numpy as np

from mmdet3d.core.bbox import box_np_ops, points_cam2img
from .lidar_data_utils import get_lidar_info


def create_lidar_info_file(data_path, pkl_prefix="lidar"):
    """解析数据集,创建中间格式 lidar_info_xxx.pkl 文件并存储
    数据格式：
        [
            {
                point_cloud: {
                    idx: filename # 文件名
                    num_features: 4
                    path: path to pointcloud # 绝对路径
                }
                annos: {
                    name: [num_gt] ground truth name array
                    location: [num_gt, 3] array
                    dimensions: [num_gt, 3] array
                    rotation_y: [num_gt] angle array
                    num_points_in_gt (list(int)): number of points in each gt
                }
            },
            ...
        ]

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Default: 'lidar'.
    """
    data_path = Path(data_path)
    train_ids = _read_file(str(data_path / "train.txt"))
    val_ids = _read_file(str(data_path / "val.txt"))
    test_ids = _read_file(str(data_path / "test.txt"))

    print("Generate info. this may take several minutes.")
    save_path = Path(data_path)  # 默认保存在data_path下

    # save train info
    lidar_infos_train = get_lidar_info(path=data_path, ids=train_ids)
    _calculate_num_points_in_gt(lidar_infos_train)
    filename = save_path / f"{pkl_prefix}_infos_train.pkl"
    print(f"Lidar info train file is saved to {filename}")
    mmcv.dump(lidar_infos_train, filename)

    # save val info
    lidar_infos_val = get_lidar_info(path=data_path, ids=val_ids)
    _calculate_num_points_in_gt(lidar_infos_val)
    filename = save_path / f"{pkl_prefix}_infos_val.pkl"
    print(f"Lidar info val file is saved to {filename}")
    mmcv.dump(lidar_infos_val, filename)

    # save trainval info (train_info + val_info)
    filename = save_path / f"{pkl_prefix}_infos_trainval.pkl"
    print(f"Lidar info trainval file is saved to {filename}")
    mmcv.dump(lidar_infos_train + lidar_infos_val, filename)

    # test info
    lidar_infos_test = get_lidar_info(path=data_path, ids=test_ids)
    filename = save_path / f"{pkl_prefix}_infos_test.pkl"
    print(f"Lidar info test file is saved to {filename}")
    mmcv.dump(lidar_infos_test, filename)


def _read_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    lines = [x.strip() for x in lines]
    return [str(line) for line in lines]


def _calculate_num_points_in_gt(infos, num_features=4):
    """计算在ground truth box内的点的 indices, 并保存在info["annos"]["num_points_in_gt"]中.

    Args:
        infos (list[dict]): Info of the input lidar data.
        num_features (int, optional): 读取点云文件的维度. Default: 4.

    但是因为我们的label是在lidar坐标系下的,所以这里不需要转换.所以需要对本函数进行一定的修改
    """
    for info in mmcv.track_iter_progress(infos):
        pc_info = info["point_cloud"]
        v_path = pc_info["path"]
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info["annos"]
        num_obj = len([n for n in annos["name"] if n != "DontCare"])

        dims = annos["dimensions"][:num_obj]
        loc = annos["location"][:num_obj]
        rots = annos["rotation_y"][:num_obj]

        # Boxes3d with rotation : [x, y, z, x_size, y_size, z_size, yaw]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)

        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos["dimensions"]) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def _create_reduced_point_cloud(data_path, info_path, save_path=None, back=False, num_features=4, front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    kitti_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(kitti_infos):
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]

        v_path = pc_info["velodyne_path"]
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib["R0_rect"]
        if front_camera_id == 2:
            P2 = calib["P2"]
        else:
            P2 = calib[f"P{str(front_camera_id)}"]
        Trv2c = calib["Tr_velo_to_cam"]
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, image_info["image_shape"])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + "_reduced")
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += "_back"
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, "w") as f:
            points_v.tofile(f)


def create_reduced_point_cloud(
    data_path,
    pkl_prefix,
    train_info_path=None,
    val_info_path=None,
    test_info_path=None,
    save_path=None,
    with_back=False,
):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f"{pkl_prefix}_infos_train.pkl"
    if val_info_path is None:
        val_info_path = Path(data_path) / f"{pkl_prefix}_infos_val.pkl"
    if test_info_path is None:
        test_info_path = Path(data_path) / f"{pkl_prefix}_infos_test.pkl"

    print("create reduced point cloud for training set")
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print("create reduced point cloud for validation set")
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print("create reduced point cloud for testing set")
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, test_info_path, save_path, back=True)
