# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser


import sys
import os
from argparse import ArgumentParser
import numpy as np
import torch
from copy import deepcopy
import time
import cv2


# mmlab
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.models import build_detector
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


# ros
import rospy
from sensor_msgs.msg import Image, CompressedImage
from autoware_msgs.msg import DetectedObject, DetectedObjectArray

# local
from postprocess import result_process


class ROSExtension:
    """ROS的一个扩展"""

    def __init__(
        self,
        model,
        camera_topic,
        detected_objects_topic,
        score_thr=0.1,
        republish=False,
        compressed_flag=True,
    ):
        self.model = model
        self.camera_topic = camera_topic
        self.detected_objects_topic = detected_objects_topic
        self.score_thr = score_thr
        self.republish = republish
        self.compressed_flag = compressed_flag

        self.device = next(self.model.parameters()).device

        ## build the data pipeline
        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        self.test_pipeline = Compose(cfg.data.test.pipeline)

    def start(self):
        rospy.init_node("camera_detection", anonymous=True)

        # init publisher : publisher detected objects which is detected by camera
        self.detected_objects_publisher = rospy.Publisher(
            self.detected_objects_topic, DetectedObjectArray, queue_size=1
        )

        # init publisher : republish raw msg for sync msg and detected result
        self.republish_topic = self.camera_topic + "/republish"
        self.camera_republish_publisher = rospy.Publisher(self.republish_topic, Image, queue_size=1)

        # init subscriber : subscribe camera topic
        if self.compressed_flag:
            self.camera_subscriber = rospy.Subscriber(
                self.camera_topic, CompressedImage, self.__callback, queue_size=1, buff_size=2**24
            )
        else:
            self.camera_subscriber = rospy.Subscriber(
                self.camera_topic, Image, self.__callback, queue_size=1, buff_size=2**24
            )

        rospy.spin()

    def __callback(self, msg):
        # 1. prepare data
        ## convert sensor_msgs/Image to numpy.ndarray
        if self.compressed_flag:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            img = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        imgs = [img]
        datas = []
        for img in imgs:
            data = dict(img=img)
            data = self.test_pipeline(data)
            datas.append(data)
        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]
        data = scatter(data, [self.device])[0]

        # 2. forward the model
        start_time = time.time()
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        end_time = time.time()
        print("inference time: {}".format(end_time - start_time))

        # 3. postprocess
        detected_object_array = result_process(
            result=results[0],
            score_thr=self.score_thr,
            CLASSES=self.model.CLASSES,
            frame_id=msg.header.frame_id,
        )
        self.detected_objects_publisher.publish(detected_object_array)

        # 4. republish
        if self.republish:
            # convert img to Image and publish
            img_msg = Image()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = msg.header.frame_id
            img_msg.height = img.shape[0]
            img_msg.width = img.shape[1]
            img_msg.encoding = "rgb8"
            img_msg.is_bigendian = 0
            img_msg.step = img.shape[1] * 3
            img_msg.data = img.tobytes()
            self.camera_republish_publisher.publish(img_msg)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--topic", help="ros camera topic")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score_thr", type=float, default=0.0, help="bbox score threshold")
    parser.add_argument("--republish", type=bool, default=True, help="if republish for sync msg and detected result")
    parser.add_argument("--compressed_flag", type=bool, default=True, help="if compressed image")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline[0].type = "LoadImageFromWebcam"

    # init model
    model = init_detector(config, args.checkpoint, device=args.device)
    print("---model init done---")

    ros_extension = ROSExtension(
        model=model,
        detected_objects_topic=args.topic + "/detected_objects",
        camera_topic=args.topic,
        score_thr=args.score_thr,
        republish=args.republish,
        compressed_flag=args.compressed_flag,
    )
    print("---waiting for topic %s msgs---:" % args.topic)
    ros_extension.start()


if __name__ == "__main__":
    main()
