# Introduction

目前自动驾驶领域，基于Camera或Lidar的3D检测方法层出不穷，对于Camera而言，为了达到3D检测目的必须解决IPM(Inverse Perspective Mapping)问题，从数学层面比较可靠的方式必需要通过多帧或多视角的方式在时域或空域完成对场景的“隐性重建”

2D图像具有非常高的信息密度，但目前的所有方法中，均无法解决在IPM过程中的信息丢失问题，例如目前最有前景的BEV方法，存在BEV特征不够丰富、过远距离准确度急剧下降等问题，该问题的本质原因其实就是IPM过程中的信息丢失问题。而该问题会直接导致在小目标和远距离检测任务中Recall和Confidence的下降，而这两个任务对于解决自动驾驶的Corner Case至关重要

此外基于Camera的2D检测可以通过较低的成本完成相对较为简单的自动驾驶任务，这对于某些限定场景依然十分适用所以基于Camera的2D检测任务目前依然是自动驾驶任务绕不开的方向

本工程是一个2D检测任务框架，可以帮助使用者快速的完成数据准备、模型搭建、模型训练等任务，从而提高使用者的工作效率

## Supported

### Model

- [x] YOLOPv2

- [ ] Nanodet

- [ ] UFLD

### Tool

- [x] 自动标注数据为scalabel格式

- [x] ROS快速测试接口

### Datasets

- [x] COCO

- [ ] VOC

- [ ] BDD100K

## Environment Configuration

**v3.x**

```bash
On The Way
```

**v2.25.1**

```bash
export CONDA_ENV_NAME=mmdet2.25.1 && \
export MMDET_TAG=2.25.1 && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export TORCH_VERSION=1.12.0 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip install openmim && \
mim install mmcv-full && \
cd $ADMLOPS_PATH/mmdetection && \
git checkout v$MMDET_TAG && \
pip install -v -e . && \
cd $ADMLOPS_PATH/mmdetection_extension && \
git checkout dev && \
pip install -v -e . 
```

## Tutorials

为了方便大家快速上手使用以及提高对本工程的理解，本工程还提供了一系列Tutorials，一般是结合自动驾驶中常见的任务而展开的，希望能给大家提供一些思路

如果在使用过程中发现bug或者文档错误，非常欢迎您能提`issus`或`pr`，我将在收到通知后的第一时间尽快修复

最后，如果觉得本工程对您有帮助，希望能给一个star，我将会非常开心，感谢！
