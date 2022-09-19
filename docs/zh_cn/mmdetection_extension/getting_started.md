# Installation

> 目前仅提提供conda的安装方式


> 其中mmcv中的cuda算子编译依赖宿主机安装的cuda,所以在不同cuda版本下安装略有区别,下面分别提供ubuntu18.04+cuda10.2和ubuntu20.04+cuda11.3的示例

## ubuntu18.04 cuda10.2

```bash
# 待补充
```

## ubuntu20.04 cuda11.3

```bash
export CONDA_ENV_NAME=mmdet && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export MMCV_VERSION=1.6.0 && \
export MMCV_CUDA_VERSION=113 && \
export TORCH_VERSION=1.12.0 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip3 install openmim && \
mim install mmcv-full==$MMCV_VERSION && \
cd $MMLAB_EXTENSION_PATH/mmdetection && \
pip3 install -e . && \
cd $MMLAB_EXTENSION_PATH/mmdetection_extension && \
pip3 install -e . 
```