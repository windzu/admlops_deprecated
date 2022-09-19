# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from data_converter import lidar_converter as lidar
from data_converter import usd_converter as usd
from data_converter.create_gt_database import create_groundtruth_database
from mmdet3d_extension import datasets


def lidar_data_prep(root_path, info_prefix="lidar"):
    """准备lidar数据集
    目标生成三种类型的数据:
    1. lidar_infos_xxx.pkl 文件,其内容为符合自定义dataset class的中间格式文件,一般情况下有四个文件,分别为:
        - lidar_infos_train.pkl
        - lidar_infos_val.pkl
        - lidar_infos_test.pkl
        - lidar_infos_trainval.pkl
    2. lidar_dbinfos_train.pkl 文件,用于训练模型
    3. lidar_gt_database 文件夹,内部为gt对应的点云文件,文件名格式为:{raw_filename}_{class_name}_{gt_bbox_id}.bin

    Args:
        root_path (str): 数据集的根路径.
        info_prefix (str): 生成info文件时候指定的前缀,默认为 lidar.
    """
    # 创建 lidar_infos_xxx.pkl 文件
    lidar.create_lidar_info_file(data_path=root_path, pkl_prefix=info_prefix)

    # 创建 lidar_dbinfos_train.pkl 文件和 lidar_gt_database 文件夹
    create_groundtruth_database(
        dataset_class_name="LidarDataset",
        data_path=root_path,
        info_prefix=info_prefix,
        info_path=f"{root_path}/{info_prefix}_infos_train.pkl",
    )


def usd_data_prep(root_path, info_prefix="usd"):
    """准备lidar数据集
    目标生成三种类型的数据:
    1. usd_infos_xxx.pkl 文件,其内容为符合自定义dataset class的中间格式文件,一般情况下有四个文件,分别为:
        - usd_infos_train.pkl
        - usd_infos_val.pkl
        - usd_infos_test.pkl
        - usd_infos_trainval.pkl
    2. usd_dbinfos_train.pkl 文件,用于训练模型
    3. usd_gt_database 文件夹,内部为gt对应的点云文件,文件名格式为:{raw_filename}_{class_name}_{gt_bbox_id}.bin

    Args:
        root_path (str): 数据集的根路径.
        info_prefix (str): 生成info文件时候指定的前缀,默认为 usd.
    """
    # 创建 usd_infos_xxx.pkl 文件
    usd.create_usd_info_file(data_path=root_path, pkl_prefix=info_prefix)

    # # 创建 lidar_dbinfos_train.pkl 文件和 lidar_gt_database 文件夹
    # create_groundtruth_database(
    #     dataset_class_name="USDDataset",
    #     data_path=root_path,
    #     info_prefix=info_prefix,
    #     info_path=f"{root_path}/{info_prefix}_infos_train.pkl",
    # )


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", type=str, help="name of the dataset")
parser.add_argument("--root-path", type=str, help="specify the root path of dataset")
parser.add_argument("--extra-tag", type=str)
parser.add_argument("--workers", type=int, default=4, help="number of threads to be used")
args = parser.parse_args()

if __name__ == "__main__":
    if args.dataset == "lidar":
        lidar_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
        )
    elif args.dataset == "usd":
        usd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
        )
