python $MMDEPLOY_DIR/tools/deploy.py \
    $MMDEPLOY_DIR/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $MMDETECTION_DIR/configs/yolox/yolox_s_8x8_300e_coco.py \
    $MMDETECTION_DIR/checkpoints/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    $MMDETECTION_DIR/demo/demo.jpg \
    --work-dir work_dir \
    --device cuda:0 \
    --show \
    --dump-info