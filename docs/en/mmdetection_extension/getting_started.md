# Installation

> At present, only the installation method of conda is provided


> The cuda operator compilation in mmcv depends on the cuda installed by the host, so the installation is slightly different under different cuda versions. The following provides examples of ubuntu18.04+cuda10.2 and ubuntu20.04+cuda11.3


## ubuntu18.04 cuda10.2

```bash
# To be added
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
pip install openmim && \
mim install mmcv-full==$MMCV_VERSION && \
cd $MMLAB_EXTENSION_PATH/mmdetection && \
pip install -e . && \
cd $MMLAB_EXTENSION_PATH/mmdetection_extension && \
pip install -e . 
```