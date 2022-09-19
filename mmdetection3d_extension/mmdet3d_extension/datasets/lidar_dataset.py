# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
import sys
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import (
    Box3DMode,
    Coord3DMode,
    LiDARInstance3DBoxes,
)
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class LidarDataset(Custom3DDataset):
    r"""LidarDataset. 主要参考KittiDataset的实现

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    # NOTE : 结合自己的数据集进行定义,会被 config 中 class_names覆盖
    CLASSES = ("truck",)

    def __init__(
        self,
        data_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range

    # NOTE : 新增
    def get_database_info(self,index):
        pass

    # NOTE : 重写
    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = {
            "sample_idx": info["point_cloud"]["idx"],  # 文件名
            "pts_filename": info["point_cloud"]["path"],
            "ann_info": None,
        }

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    # NOTE : 重写
    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        annos = self.data_infos[index]["annos"]

        anns_results = {
            "gt_bboxes_3d": [],
            "gt_labels_3d": [],
            "gt_names": [],
        }

        # Convert the annos to the format of LiDARInstance3DBoxes
        names = annos["name"]
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        # Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d).convert_to(self.box_mode_3d)

        # Convert the annos to the format of np.ndarray
        labels = []
        for cat in names:
            if cat in self.CLASSES:
                labels.append(self.CLASSES.index(cat))
            else:
                labels.append(-1)
        labels = np.array(labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=names,
        )
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int32)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [i for i, x in enumerate(ann_info["name"]) if x != "DontCare"]
        for key in ann_info.keys():
            img_filtered_annotations[key] = ann_info[key][relevant_annotation_indices]
        return img_filtered_annotations

    def format_results(self, outputs, pklfile_prefix="lidar"):
        """Format the results to pkl file.
        将检测结果格式化一下
        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix)

        # 输入输出检查
        assert len(outputs) == len(self.data_infos), "invalid list length of network outputs"

        det_annos = []
        print("\nConverting prediction to lidar format")
        for idx, pred_dicts in enumerate(mmcv.track_iter_progress(outputs)):

            annos = []
            info = self.data_infos[idx]
            sample_idx = info["point_cloud"]["idx"]  # filename

            anno = {
                "name": [],
                "location": [],
                "dimensions": [],
                "rotation_y": [],
                "score": [],
            }

            # debug
            # print("pred_dicts:", pred_dicts)
            if "pts_bbox" in pred_dicts:
                pred_dicts = pred_dicts["pts_bbox"]

            if len(pred_dicts["boxes_3d"]) > 0:
                boxes_3d = pred_dicts["boxes_3d"]
                scores_3d = pred_dicts["scores_3d"]
                labels_3d = pred_dicts["labels_3d"]

                for (box_3d, score_3d, label_3d) in zip(boxes_3d, scores_3d, labels_3d):
                    anno["name"].append(self.CLASSES[int(label_3d)])
                    anno["location"].append(box_3d[:3])
                    anno["dimensions"].append(box_3d[3:6])
                    anno["rotation_y"].append(box_3d[6])
                    anno["score"].append(score_3d)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    "name": np.array([]),
                    "location": np.zeros([0, 3]),
                    "dimensions": np.zeros([0, 3]),
                    "rotation_y": np.array([]),
                    "score": np.array([]),
                }
                annos.append(anno)

            annos[-1]["sample_idx"] = np.array([sample_idx] * len(annos[-1]["score"]), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith((".pkl", ".pickle")):
                out = f"{pklfile_prefix}.pkl"
            mmcv.dump(det_annos, out)
            print(f"Result is saved to {out}.")

        return det_annos, tmp_dir

    # NOTE : 重写
    def evaluate(
        self,
        results,
        metric=["bev", "3d"],
        logger=None,
        pklfile_prefix=None,
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """对检测结果做eval.

        TODO : 通过nus api做eval

        Args:
            results (list[dict]): Testing results of the dataset.
            >>> 格式如下：
                [
                    {
                        "boxes_3d": LiDARInstance3DBoxes(tensor([], size=(0, 7))),
                        "scores_3d": tensor([]),
                        "labels_3d": tensor([], dtype=torch.int64),
                    },
                    ...
                ]
                或者
                [
                    {
                        "pts_bbox": {
                            "boxes_3d": LiDARInstance3DBoxes(tensor([], size=(0, 7))),
                            "scores_3d": tensor([]),
                            "labels_3d": tensor([], dtype=torch.int64),
                        },
                    },
                    ...
                ]
            metric (str | list[str], optional): Metrics to be evaluated. Default: ["bev", "3d"].
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files,
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        from mmdet3d_extension.core.evaluation import lidar_eval

        gt_annos = [info["annos"] for info in self.data_infos]

        ap_result_str, ap_dict = lidar_eval(
            gt_annos=gt_annos,  # gt annotations
            dt_annos=result_files,  # detection results
            current_classes=self.CLASSES,
            eval_types=metric,  # default evaluate bev and 3d
        )
        print_log("\n" + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict["boxes_3d"]
        scores = box_dict["scores_3d"]
        labels = box_dict["labels_3d"]
        sample_idx = info["image"]["image_idx"]
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

        box_corners = box_preds.corners
        # box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        # minxy = torch.min(box_corners_in_image, dim=1)[0]
        # maxxy = torch.max(box_corners_in_image, dim=1)[0]
        # box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        # image_shape = box_preds.tensor.new_tensor(img_shape)
        # valid_cam_inds = (
        #     (box_2d_preds[:, 0] < image_shape[1])
        #     & (box_2d_preds[:, 1] < image_shape[0])
        #     & (box_2d_preds[:, 2] > 0)
        #     & (box_2d_preds[:, 3] > 0)
        # )
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = (box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])
        valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend="disk"),
            ),
            dict(type="DefaultFormatBundle3D", class_names=self.CLASSES, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ]
        if self.modality["use_camera"]:
            pipeline.insert(0, dict(type="LoadImageFromFile"))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]

            pts_path = data_info["point_cloud"]["velodyne_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            points, img_metas, img = self._extract_data(i, pipeline, ["points", "img_metas", "img"])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)["gt_bboxes_3d"].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            pred_bboxes = result["boxes_3d"].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir, file_name, show)

            # multi-modality visualization
            if self.modality["use_camera"] and "lidar2img" in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas["lidar2img"],
                    out_dir,
                    file_name,
                    box_mode="lidar",
                    show=show,
                )
