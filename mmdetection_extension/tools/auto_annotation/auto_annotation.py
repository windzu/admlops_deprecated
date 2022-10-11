import json
import numpy as np
import cv2
from urllib.request import urlopen
from rich.progress import track

import asyncio
from argparse import ArgumentParser

from mmdet.apis import async_inference_detector, inference_detector, init_detector, show_result_pyplot

COCO_TO_BDD100K = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


class AutoAnnotation:
    def __init__(self, input, type, config, checkpoint, device, score_thr=0.3):
        self.input = input
        self.type = type
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr

        # build the model from a config file and a checkpoint file
        self.model = init_detector(self.config, self.checkpoint, device=self.device)
        self.class_names = self.model.CLASSES

        # get all file list
        self.file_list = self._parse_file_list()

    def run(self):

        result_list = []
        if self.file_list:
            for file in track(self.file_list):
                # test a single image
                # judge if the file is a path or url
                if file.startswith("http://") or file.startswith("https://"):
                    resp = urlopen(file)
                    img = np.asarray(bytearray(resp.read()), dtype="uint8")
                    img = cv2.imdecode(img, -1)
                else:
                    img = cv2.imread(file)

                result = inference_detector(self.model, img)
                result = self._filter_result(file, result)
                if result:
                    result_list.append(result)

        self._save_result(result_list)

        print("Done")

    def _filter_result(self, file, result):
        """通过mmdet的模型预测出的结果,过滤出符合要求的结果

        Args:
            result(dict): 检测结果
        """
        # 1. format the result to the standard format
        (bboxes, labels, label_names) = self.format_result_to_standard_format(result, self.class_names, self.score_thr)

        # 2. filter the result to the required format
        if str(self.type) == "scalabel":
            result = self.format_result_to_scalabel_format(bboxes, labels, label_names)
            if result:
                result["name"] = file
                result["url"] = file
                return result
            else:
                result = None
        else:
            print("Invalid type")
            return None

    def _save_result(self, result_list):
        if str(self.type) == "scalabel":
            save_path = self.input.split(".json")[0] + "_auto_annotation.json"
            print("Save result to {}".format(save_path))
            self.save_scalabel(result_list, save_path)
        else:
            print("Invalid type")

    def _parse_file_list(self):
        """根据输入的文件路径或者文件夹路径以及其对应的数据类型，解析出图片列表

        Returns:
            list(str): 带有图片路径的列表
        """
        if str(self.type) == "scalabel":
            return self.parse_scalabel(self.input)
        else:
            print("Invalid type")
            return None

    @staticmethod
    def parse_scalabel(path):
        """解析scalabel格式的图像列表文件

        scalabel格式的图像列表文件是一个json文件,其内容如下：
        {
            {
                "url": "http://localhost:8686/items/weitang_image/zhixian1.jpg"
            },
            {
                "url": "http://localhost:8686/items/weitang_image/zhixian2.jpg"
            },
        ...
        }

        Args:
            input (str): scalabel格式的图像列表文件
        """
        # load json and get all url to list and return
        with open(path, "r") as f:
            data = json.load(f)
            return [item["url"] for item in data]

    @staticmethod
    def format_result_to_standard_format(result, class_names, score_thr=0.3):
        """将mmdet的检测结果转换成通用格式
        Args:
            result (Tensor or tuple): 检测结果
            class_names (list(str)): 类别名称列表
            score_thr (float): 分数阈值

        Returns:
            list(dict): 通用格式的检测结果

        """

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        # TODO : add support to segmentation masks

        # filter out the results
        assert bboxes is None or bboxes.ndim == 2, f" bboxes ndim should be 2, but its ndim is {bboxes.ndim}."
        assert labels.ndim == 1, f" labels ndim should be 1, but its ndim is {labels.ndim}."
        assert (
            bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        ), f" bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}."
        assert (
            bboxes is None or bboxes.shape[0] <= labels.shape[0]
        ), "labels.shape[0] should not be less than bboxes.shape[0]."

        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            # TODO : add support to filter segmentation masks

        # # debug
        # print("bboxes: ", bboxes)
        # print("labels: ", labels)

        # get label name
        label_names = [class_names[i] for i in labels]

        return (bboxes, labels, label_names)

    @staticmethod
    def format_result_to_scalabel_format(bboxes, labels, label_names):
        result = {
            "name": "",
            "url": "",
            "videoName": "",
            "timestamp": 0,
            "attributes": {},
            "labels": [],
            "sensor": -1,
        }
        # filter out the results which are not in the required format
        for i, (bbox, label, label_name) in enumerate(zip(bboxes, labels, label_names)):
            if label_name in COCO_TO_BDD100K:
                result["labels"].append(
                    {
                        "id": i,
                        "category": COCO_TO_BDD100K[label_name],
                        "attributes": {},
                        "manualShape": True,
                        "box2d": {
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3]),
                        },
                        "poly2d": None,
                        "box3d": None,
                    }
                )

        if len(result["labels"]) == 0:
            return None
        else:
            return result

    @staticmethod
    def save_scalabel(result_list, save_path):
        """保存scalabel格式的检测结果
        如果该文件已经存在，则会覆盖原有文件

        Args:
            result_list (list(dict)): 检测结果列表
            save_path (str): 保存路径
        """
        with open(save_path, "w") as f:
            json.dump(result_list, f, indent=4)
