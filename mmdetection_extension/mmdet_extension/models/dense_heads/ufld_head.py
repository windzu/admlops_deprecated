from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch import nn

from mmcv.runner import BaseModule
from mmdet.builder import HEADS, build_loss


@HEADS.register_module()
class UFLDHead(BaseModule):
    """Base class for UFLDHead.

    Args:
        in_channels (int): Number of channels in the input feature map.
        dims (tuple): Dimensions of the output feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    TODO: 完善UFLDHead的实现
        - 想清楚ufld的label以什么样的个是集成进coco的标注中
        - 想清楚ufld的loss计算方式
    """

    def __init__(
        self,
        in_channels=256,
        dims=(101, 56, 4),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
        init_cfg=None,
    ):
        super(UFLDHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.dims = dims
        self.init_cfg = init_cfg
        self.total_dim = np.prod(self.dims)

        self.loss_cls = build_loss(loss_cls)

        self.fc_cls = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

    def forward(self, x):
        """Forward function."""
        return self.fc_cls(x).view(-1, *self.dims)

    def loss(self, result, gt_lanes):
        """Calculate loss."""
        return self.loss_cls(result, gt_lanes)

    def forward_train(self, x, gt_lanes):
        """Forward function during training."""
        result = self.forward(x)
        return self.loss(result, gt_lanes)
