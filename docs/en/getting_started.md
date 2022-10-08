# Introduction

> Introduction about Autonomous Driving Perception Tasks

admlops means MLOps for autonomous driving perception tasks

The autonomous driving perception task should be a closed-loop task, and its life cycle should include at least three stages, such as data, model, and deployment. Therefore, in the design process of MLOps, at least these three directions need to be covered. More features will be added in the future

## Data

The importance of data for deep learning is self-evident, and a good dataset should take into account both "depth" and "breadth". The breadth can be satisfied with the help of open source data sets and its own continuous collection, while the depth needs to be combined with actual tests, targeted analysis, and re-collection to complete.

### Dataset

* Open source dataset
  
  * Collection: Collect and download data sets related to autonomous driving
  
  * Analysis: Do data analysis on the collected data set, and evaluate whether the data distribution it contains meets the needs of the current task
  
  * Summarize and organize: For useful open source data, classify it according to the category of tasks, and manage it together with the data set of the same task
  
  * Conversion: According to the needs of the pipeline, develop the corresponding conversion script and convert it to the data set format required by the pipeline

* custom data
  
  * Data set definition: Customize the required data according to the needs of the task, including its category, labeling standard, labeling format, etc.
  
  * Data collection: develop data collection tools, which need to meet the above definition requirements, and need to consider the needs of data screening, continuous collection and other tasks
  
  * Data annotation: According to the definition, write detailed annotation requirements, and then hand it over to the data annotation company
  
  * Analysis, induction, transformation: refer to the description in the open source dataset

### Dataset Iteration

It is impossible to solve all scene problems by collecting data several times, because the corner cases in the automatic driving task are difficult to solve at once, so it is necessary to continuously collect new data according to the actual test situation to supplement the existing ones. data set to gradually cover as many scenes as possible

### Dataset Pipeline Design

With the development of the autonomous driving industry, there will inevitably be more models with better performance, and it is very likely that the way these models load data will be different from the previous general methods, such as the popular BEV scheme not long ago. Therefore, in order to adapt to these models, it is necessary to write the corresponding pipeline for loading data, so that the training and testing of the new model can be quickly completed based on the existing data set.

## Modeling

> The model part is based on a series of excellent frameworks open-sourced by the open-mmlab laboratory. It can easily reproduce some classic models in the past, and has built a new model based on it to verify its own. idea, thanks to open-mmlab

### Model Reproduction & Framework Extension & Network Refactoring

According to information such as papers and open source codes, some classic and cutting-edge models in the field of autonomous driving are reproduced in MLOps. For some relatively new or not mainstream network structures, open-mmlab has not yet supported them. These networks The structure needs to be implemented manually, and I usually write these implementations in the extension module. In addition, if you want to verify some of your ideas, you can refactor the model structure by simply modifying the configuration file, so as to quickly complete some experiments

### Train & Test & Evaluate

It is convenient to train the model and help us evaluate the effect of the model and the effect of the dataset through the results of testing and evaluation

## Deploy

The deployment of the model is a complete engineering problem, and the biggest problem in deployment is often caused by the deployment operator provided by the deployment platform used does not support some operators in the deployment network.

Even so, we still try our best to base on some mature deployment frameworks, such as tensorrt, openvino, etc., which will greatly simplify the deployment task.

If you want to further optimize the deployment, you also need deployment skills such as knowledge distillation and pruning.

# Preparation

> In order to ensure the consistency of the operation, please be sure to complete the relevant preparations according to the following steps

## large file storage folder

> Large files such as checkpoint and training data are planned to be linked to the project through soft links

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

## pull project

```bash
git clone https://github.com/windzu/admlops.git --depth 1 && \
cd admlops && \
git submodule update --init --recursive
```

## add environment variable

> For the convenience of the follow-up, please be sure to add it and choose .bashrc or .zshrc according to your shell

```bash
# Add project path
cd admlops && \
echo "export ADMLOPS_PATH=$(pwd)" >> ~/.zshrc

# Add repository path
cd admlops_repository && \
echo "export ADMLOPS_REP_PATH=$(pwd)" >> ~/.zshrc
```

## Add data soft link

> Note: The data soft link does not work in the docker environment, so if you configure the environment through docker, you need to mount the data folder through the volume parameter

```bash
ln -s $ADMLOPS_REP_PATH/checkpoints $ADMLOPS_PATH && \
ln -s $ADMLOPS_REP_PATH/data $ADMLOPS_PATH
```

# Development Environment Setup

The following factors need to be considered when building:

- Because environments such as mmdet and mmdet3d are not compatible with each other, the environments need to be separated
- For the main development environment, install it through the source code, which makes it easier to view the source code, so that the extension can be added according to the structure, which is also the reason for adding submodules to the project. For example, to configure the mmdetetion3d environment, mmdetection3d is compiled and installed through its source code, but for the mmdetetion and mmcv it depends on, it can be installed directly through mim, which can also ensure that the versions will not conflict
- The environment configuration is related to the system version and cuda version, select the appropriate installation script according to the needs

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

> In two steps, build docker image and create docker container

### build image

> Added some content on the basis of mmdeploy's dockerfile, the specific content is in the dockerfile under mmdeploy_extension/docker/GPU, which has detailed instructions

```bash
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker build docker/GPU -t mmdeploy:main \
--build-arg USE_SRC_INSIDE=true 
```

### create container

> Some environment variables and network related settings are made in docker-compose. The specific content is described in detail in the mmdeploy_extension/docker-compose.yml file

> note: If the environment is only used for testing and does not need to be deployed in combination with ros, you can refer to the following, if not, please refer to the docker network configuration in the next paragraph, you need to modify docker-compose

```bash
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker-compose up -d 
```

# Deployment Environment Setup

> If you also need to deploy based on this docker, there are some additional settings as follows

- The container needs to configure the master-slave machine with the host: this requires the host to create a bridge network and specify a network segment, and the container needs to specify the network and its own ip when creating
- Add the master-slave configuration of ros in container and host

## create docker bridge network

- network type: bridge
- subnet: 172.28.0.0

```bash
docker network create \
  --driver=bridge \
  --subnet=172.28.0.0/16 \
  admlops
```

## specify network and set ip in docker-compose

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

> Because the container has only one ip, it directly obtains its address and fills in it, instead of manually assigning a fixed address

```bash
# 1. enter container
docker exec -it mmdeploy /bin/bash

# 2. set master and slave
echo export ROS_IP=`hostname -I` >> ~/.bashrc && \
echo export ROS_HOSTNAME=`hostname -I` >> ~/.bashrc && \
echo export ROS_MASTER_URI=http://172.28.0.1:11311 >> ~/.bashrc
```
