# Introduction

> 自动驾驶感知任务介绍

admlops意为用于自动驾驶感知任务的MLOps

自动驾驶感知任务应该是一个闭环任务，其生命周期应该至少包括数据、模型和部署三个阶段。 因此，在MLOps的设计过程中，至少需要涵盖这三个方向。 未来将添加更多功能

## Data

数据对于深度学习的重要性不言而喻，一个好的数据集应该兼顾“深度”和“广度”。 广度可以借助开源数据集和自身不断的收集来满足，而深度则需要结合实际测试，针对性分析，重新收集来完成。

### Dataset

* 开源数据集
  
  * 采集：采集和下载自动驾驶相关数据集
  
  * 分析：对采集到的数据集进行数据分析，评估其包含的数据分布是否满足当前任务的需求
  
  * 总结整理：对于有用的开源数据，按照任务的类别进行分类，与同一任务的数据集一起管理
  
  * 转换：根据流水线的需要，开发相应的转换脚本，转换成流水线需要的数据集格式

* 自定义数据
  
  * 数据集定义：根据任务需要自定义所需数据，包括其类别、标注标准、标注格式等。
  
  * 数据采集：开发数据采集工具，需要满足上述定义要求，需要考虑数据筛选、持续采集等任务的需要
  
  * 数据标注：根据定义，写出详细的标注需求，然后交给数据标注公司
  
  * 分析、归纳、转化：参考开源数据集中的描述

### 数据集迭代

多次采集数据不可能解决所有场景问题，因为自动驾驶任务中的极端情况很难一次性解决，需要根据实际测试情况不断采集新的数据补充已有的 . 数据集逐渐覆盖尽可能多的场景

### Dataset Pipeline Design

随着自动驾驶行业的发展，必然会出现更多性能更好的模型，这些模型加载数据的方式很可能会与之前的通用方式有所不同，比如不久前流行的BEV方案。 因此，为了适应这些模型，需要编写相应的加载数据的管道，以便在现有数据集的基础上快速完成新模型的训练和测试。

## Modeling

> 模型部分基于open-mmlab实验室开源的一系列优秀框架。 它可以轻松复制过去的一些经典模型，并在其基础上构建了一个新模型来验证自己的想法，感谢 open-mmlab

### 模型复现 & 框架扩展 & 网络重构

根据论文和开源代码等信息，在MLOps中复制了一些自动驾驶领域的经典和前沿模型。 对于一些比较新的或者不是主流的网络结构，open-mmlab 还没有支持。 这些网络结构需要手动实现，我通常在扩展模块中编写这些实现。 另外，如果想验证自己的一些想法，可以通过简单的修改配置文件来重构模型结构，从而快速完成一些实验

### 训练 & 测试 & 评估

方便训练模型，通过测试和评估的结果帮助我们评估模型的效果和数据集的效果

## Deploy

模型的部署是一个完整的工程问题，而部署中最大的问题往往是所使用的部署平台提供的operator不支持部署网络中的一些operator。

尽管如此，我们还是尽量基于一些成熟的部署框架，比如tensorrt、openvino等，这将大大简化部署任务。

如果要进一步优化部署，还需要知识蒸馏、剪枝等部署技巧。

# Preparation

> 为了保证操作的一致性，请务必按照下述步骤完成相关的准备工作

## 大文件存储文件夹

> checkpoint 、训练数据等大文件计划通过软链接至工程当中

```bash
admlops_repository
├── checkpoints
│   ├── mmdet
│   ├── mmdet3d
└── data
    ├── mmdet
    ├── mmdet3d
```

# General Configuration

## 拉取工程

```bash
git clone https://github.com/windzu/admlops.git --depth 1 && \
cd admlops && \
git submodule update --init --recursive
```

## 添加环境变量

> 为了后续的方便，请务必添加，根据自己的shell选择.bashrc 或者.zshrc之类的

```bash
# 添加工程路径
cd admlops && \
echo "export ADMLOPS_PATH=$(pwd)" >> ~/.zshrc

# 添加存储库路径
cd admlops_repository && \
echo "export ADMLOPS_REP_PATH=$(pwd)" >> ~/.zshrc
```

## 添加数据软链接

> 注意 ： 数据软链接在docker环境下是不奏效的，所以如果是通过docker进行环境的配置，需要通过volume参数将数据文件夹挂载进去

```bash
ln -s $ADMLOPS_REP_PATH/checkpoints $ADMLOPS_PATH && \
ln -s $ADMLOPS_REP_PATH/data $ADMLOPS_PATH
```

# Development Environment Setup

搭建时候有如下因素需要考虑：

- 因为mmdet和mmdet3d等环境并不互相兼容，所以需要环境需要分离
- 对于主要使用的开发环境，通过源码安装，这样更方便查看源码，以便对照结构添加extension，这也是在工程中添加submodule的原因。例如配置mmdetetion3d环境，mmdetection3d通过其源码编译安装，但是对于其依赖的mmdetetion和mmcv，直接通过mim安装即可，这样也可以保证版本不会冲突
- 环境配置与与系统版本和cuda版本相关，根据需求选择合适的安装脚本

## mmdetection (conda)

### Ubuntu18.04+CUDA10.2

```bash
# on the way
```

### Ubuntu20.04+CUDA11.3

```bash
export CONDA_ENV_NAME=mmdet && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export TORCH_VERSION=1.12.0 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip install openmim && \
mim install mmcv-full && \
cd $ADMLOPS_PATH/mmdetection && \
pip install -e . && \
cd $ADMLOPS_PATH/mmdetection_extension && \
pip install -e . 
```

## mmdetection3d (conda)

### Ubuntu18.04+CUDA10.2

```bash
# on the way
```

### Ubuntu20.04+CUDA11.3

```bash
export CONDA_ENV_NAME=mmdet3d && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export TORCH_VERSION=1.12.0 && \
export MMCV_VERSION=1.6.0 && \
export MMDET_VERSION=2.25.0 && \
export MMCV_CUDA_VERSION=113 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip install openmim && \
mim install mmcv-full && \
mim install mmdet && \
mim install mmsegmentation && \
cd $ADMLOPS_PATH/mmdetection3d && \
pip install -e . && \
cd $ADMLOPS_PATH/mmdetection3d_extension && \
pip install -e .
```

## mmdeploy (docker)

> 分两个步骤，build docker image 和 create docker container

### build image

> 在mmdeploy的dockerfile基础上添加了一些内容，具体内容在mmdeploy_extension/docker/GPU下的dockerfile中，里面有详细的说明

```bash
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker build docker/GPU -t mmdeploy:main \
--build-arg USE_SRC_INSIDE=true 
```

### create container

> docker-compose中做了一些环境变量和网络的相关设置,具体内容在mmdeploy_extension/docker-compose.yml文件中有详细说明

> note：如果该环境仅仅是用于测试，而不需要结合ros进行部署使用，可以参照如下，如果不是，请参照下一小段中的的docker网络配置，需要修改一下docker-compose

```bash
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker-compose up -d 
```

# Deployment Environment Setup

> 如果还需要基于此docker进行部署，则还有一些额外设置，内容如下

- container需要与主机配置主从机器：这需要主机创建bridge网络并指定网段，container在创建的时候要指定网络以及自己的ip
- 在container和host中添加ros的主从机配置

## 创建docker bridge网络

- 网络类型：bridge
- 网段：172.28.0.0

```bash
docker network create \
  --driver=bridge \
  --subnet=172.28.0.0/16 \
  admlops
```

## docker-compose中指定网络并设置ip

```bash
services:
  mmdeploy:
    networks:
      - admlops

# use alread existing network
networks:
  admlops:
    external: true
```

## container中设置主从机

> 因为container只有一个ip 所以直接获取其地址填入中，而不是通过手动分配固定地址的方式

```bash
# 1. 进入container
docker exec -it mmdeploy /bin/bash

# 2. 设置主从机
echo export ROS_IP=`hostname -I` >> ~/.bashrc && \
echo export ROS_HOSTNAME=`hostname -I` >> ~/.bashrc && \
echo export ROS_MASTER_URI=http://172.28.0.1:11311 >> ~/.bashrc
```
