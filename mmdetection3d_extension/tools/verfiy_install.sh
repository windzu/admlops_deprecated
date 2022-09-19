conda activate mmdet3d && \
cd $MMLAB_EXTENSION_PATH/mmdetection3d && \
python3 demo/pcd_demo.py demo/data/kitti/kitti_000008.bin \
    ./configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py \
    ./checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth \
    --show