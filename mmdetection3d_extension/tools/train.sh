#!/bin/sh
# pointpillars kitti 3 class
# cd $MMLAB_EXTENSION_PATH/mmdetection3d_extension/ && \
# python tools/train.py $MMLAB_EXTENSION_PATH/mmdetection3d_extension/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_pointcloud-3d-3class.py

# pointpillars kitti 1 class
cd $MMLAB_EXTENSION_PATH/mmdetection3d_extension/ && \
python tools/train.py $MMLAB_EXTENSION_PATH/mmdetection3d_extension/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_pointcloud-3d-car.py

# pointpillars nus 1 class
# cd $MMLAB_EXTENSION_PATH/mmdetection3d_extension/ && \
# python tools/train.py $MMLAB_EXTENSION_PATH/mmdetection3d_extension/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py

