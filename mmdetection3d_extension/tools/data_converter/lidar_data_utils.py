# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image
from skimage import io


def get_lidar_info(path, ids: list, num_worker=8):
    """获取lidar数据的信息
    数据格式：
    [
        {
            "point_cloud": {
                "idx": str # filename
                "num_features": int # 点云维度
                "path": str # real path to pointcloud file
            }
            "annos": {
                "name": np.array([num_gt]), # 标注的类别名称
                "location": np.array([num_gt, 3]), # x,y,z坐标
                "dimensions": np.array([num_gt, 3]), # x_size y_size z_size
                "rotation_y": np.array([num_gt]), # yaw
            }
        },
        ...
    ]

    Args:
        path (str): 数据集的根路径
        ids (list): 需要读取的数据的文件名list
        num_worker (int): 并行处理的数量

    """
    root_path = Path(path)

    def map_func(idx):
        """根据文件名获取该文件名所对应的原始数据、label等相关信息

        Args:
            idx (str): 文件名(不包含后缀)

        Returns:
            dict: info dict
        """
        info = {
            "point_cloud": {
                "idx": idx,
                "num_features": 4,
                "path": get_path(root_path, ["lidar"], idx, ".bin"),
            },
            "annos": {
                "name": [],
                "location": [],
                "dimensions": [],
                "rotation_y": [],
            },
        }

        # get label path
        label_path = get_path(root_path, ["label"], idx, ".txt")
        if label_path is not None:
            info["annos"] = get_label_anno(label_path)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        lidar_infos = executor.map(map_func, ids)

    return list(lidar_infos)


def get_path(root_path, folder_list, idx, file_suffix):
    """提供文件根路径,文件夹列表,文件名,文件后缀,返回文件的路径"""
    path = osp.join(root_path, *folder_list)
    path = osp.join(path, str(idx) + file_suffix)
    # check path exists
    if not osp.exists(path):
        print("{} does not exist".format(path))
        return None
    return path


def get_label_anno(label_path):
    """获取标注信息
    数据格式：
        {
            "name": np.array([num_gt]), # 标注的类别名称
            "location": np.array([num_gt, 3]), # x,y,z坐标
            "dimensions": np.array([num_gt, 3]), # x_size y_size z_size
            "rotation_y": np.array([num_gt]), # yaw
        },

    Args:
        label_path (str): label path

    Returns:
        dict: 继承自kitti的标注信息,对于其中没有的信息，留空
    """

    with open(label_path, "r") as f:
        lines = f.readlines()

    content = [line.strip().split(" ") for line in lines]

    annos = {
        "name": np.array([x[0] for x in content]),
        "location": np.array([[float(info) for info in x[1:4]] for x in content]).reshape(-1, 3),
        "dimensions": np.array([[float(info) for info in x[4:7]] for x in content]).reshape(-1, 3),
        "rotation_y": np.array([float(x[7]) for x in content]).reshape(-1),
    }

    return annos
