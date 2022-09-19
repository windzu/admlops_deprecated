python $MMDEPLOY_DIR/tools/test.py \
    $MMDEPLOY_DIR/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    $MMDETECTION_DIR/configs/yolox/yolox_s_8x8_300e_coco.py \
    --model work_dir/end2end.engine
    --out out.pkl \
    --device cuda:0 \
