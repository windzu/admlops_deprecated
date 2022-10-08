import torch
import warnings

from mmdet.models.builder import DETECTORS, build_backbone, build_neck, build_head
from mmdet.models.detectors import SingleStageDetector


@DETECTORS.register_module()
class YOLOW(SingleStageDetector):
    """用于自动驾驶的onestage的多任务检测器

    TODO : 完善YOLOW的实现
        - 想清楚多任务包含哪些内容
        - 似乎只需要添加一个对gt_lane的支持就可以了
        - 流程的实现似乎只需要参考yolox就可以,不同的地方就是head的实现

    """

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        mask_head,
        lane_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        input_size=(640, 640),
        init_cfg=None,
    ):
        super(YOLOW, self).__init__(init_cfg)
        if pretrained:
            warnings.warn("DeprecationWarning: pretrained is deprecated, " 'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)
        if mask_head is not None:
            self.mask_head = build_head(mask_head)
        if lane_head is not None:
            self.lane_head = build_head(lane_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        gt_lanes=None,
    ):
        """
        TODO : 添加对多任务的train支持,所以需要重写forward_train
          - 添加对 gt_masks 的支持
          - 添加对 gt_lanes 的支持

            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                    Typically these should be mean centered and std scaled.
                img_metas (list[dict]): A List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    :class:`mmdet.datasets.pipelines.Collect`.
                gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                    image in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): Class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                    boxes can be ignored when computing the loss.
                gt_masks (None | Tensor) : true segmentation masks for each box
                    used if the architecture supports a segmentation task.
                gt_lanes (None | Tensor) : true lane masks for each box
                    used if the architecture supports a lane task.

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)  # TODO : 这里在干嘛？
        x = self.extract_feat(img)

        # 通过不同的head来计算loss
        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_loss)

        if self.with_mask_head:
            mask_loss = self.mask_head.forward_train(x, img_metas, gt_masks)
            losses.update(mask_loss)

        if self.with_lane_head:
            lane_loss = self.lane_head.forward_train(x, img_metas, gt_lanes)
            losses.update(lane_loss)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.multitasking_head.simple_test(feat, img_metas, rescale=rescale)

        # TODO : 添加对多任务的test支持
        #         bbox_results = [
        #             bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list
        #         ]
        #         return bbox_results
        pass

    @property
    def with_bbox_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, "mask_head") and self.mask_head is not None

    @property
    def with_lane_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, "lane_head") and self.lane_head is not None
