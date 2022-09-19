# Introduction

> the automatic driving direction will be given priority when the function is implemented.

The `closed loop` of a deep learning task can be simply divided into the following three parts

- Data : Collection, Label, Data Management
- Model : Build Model, Training, Testing, Evaluation
- Deploy : Model Deployment

As the complexity of tasks increases, it becomes more difficult to rely on a single person to complete the whole process of work, and if you rely on the cooperation of multiple people, there will often be "wrangling" between collaborators due to level mismatch and conflicting work content. In response to this situation, the industry proposed the concept of `MLOps`

But at present, according to my research, there is no mature `MLOps` solution, and many open source tools have many restrictions, so before there are no mature and useful tools, you can only use open source tools to simply piece together one and use it. ~

This project relies on a series of tools in [open-mmlab](https://github.com/open-mmlab) to complete the modeling and deployment work in the MLOps task, because the main used mmdeploy and mmdetection3d are still there Quick updates, so engineers who want to use this project, please read the relevant documents and update comments, and hope that everyone can cooperate and communicate, mention more issues and pr, and jointly create a work that is easy to use, and set aside some time. fish

# Open-mmlab Introduction

> Because this project is built on the basis of openmmlab with some minor modifications, the premise of using this project is to have a certain understanding of the related work of open-mmlab. The following is the relevant introduction

The series of frameworks in open-mmlab have dependencies, and their essence is the inheritance performance of OO (object-oriented).

If you want to quickly understand open-mmlab, you must first understand the `factory design mode`, which is implemented through registry in open-mmlab. For details, please refer to: [registry details](https://zhuanlan.zhihu.com/p/355271993) or is the corresponding code (the code is very short, it is recommended to look at it)

**basic introduction**

- mmcv：Base library in open-mmlab
  
    The basic functions in the other frameworks of open-mmlab are derived from it or inherited from it

- mmdetection：An integrated framework for 2D detection and segmentation
  
    Provide landmark model support in this field, as well as other common model support with good reputation, such as RCNN series, yolo series, etc. The update speed is generally about half a year later than that of academia

- mmdetection3d：An integrated framework for 3D detection and segmentation
  
    It is modified on the basis of mmdetection. After all, many 3d detection methods are actually 2d methods. The update speed is generally about half a year later than that of academia, but because of the popularity of automatic driving and the 2D track rolling Everyone is turning to 3D, so its recent update is relatively "rapid"

- mmdeploy：a deployment framework
  
    The purpose of its appearance is to get through the last step of the project landing in open-mmlab, and its update speed is the slowest, generally a very mature model will be supported (in other words, if the support speed is fast, then everyone will be unemployed. 😬) Each supported model generally supports a variety of backends, at least onnx and tensorrt have not yet reached the official version of v1.0. For the supported models, some will provide related sdk intimately, so that the front and back are connected. You don't need to write it yourself.

# Environment configuration

> For the convenience of subsequent use, it is best to prepare the necessary configuration of checkpoints, data and environment variables according to the following requirements

## Prepare Data

> Prepare the data required for training and testing, pre-trained checkpoints, and place them according to the specified structure for subsequent configuration data symbolic links

### Checkpoints 

The file structure is as follows

```bash
mmlab_checkpoints
├── mmdetection
│   └── checkpoints
└── mmdetection3d
    └── checkpoints
```

### Data

The file structure is as follows

```bash
mmlab_dataset
├── mmdetection
│   └── data
└── mmdetection3d
    └── data
```

## Pull Project

> In order to reduce the amount of data, generally only pull the latest commit 

```bash
git clone https://github.com/windzu/mmlab_extension.git --depth 1 && \
cd mmlab_extension && \
git submodule update --init --recursive
```

## Add Environment Variables

> based self shell to choose .bashrc or .zshrc

```bash
cd mmlab_extension && \
echo "export MMLAB_EXTENSION_PATH=$(pwd)" >> ~/.bashrc 

cd mmlab_checkpoints && \
echo "export MMLAB_CHECKPOINTS_PATH=$(pwd)" >> ~/.bashrc

cd mmlab_dataset && \
echo "export MMLAB_DATASET_PATH=$(pwd)" >> ~/.bashrc
```

## Configure Data Symbolic Link

```bash
# checkpoints
rm -rf $MMLAB_EXTENSION_PATH/mmdetection/checkpoints && \
ln -s $MMLAB_CHECKPOINTS_PATH/mmdetection/checkpoints $MMLAB_EXTENSION_PATH/mmdetection/checkpoints && \
rm -rf $MMLAB_EXTENSION_PATH/mmdetection3d/checkpoints && \
ln -s $MMLAB_CHECKPOINTS_PATH/mmdetection3d/checkpoints $MMLAB_EXTENSION_PATH/mmdetection3d/checkpoints

# dataset
rm -rf $MMLAB_EXTENSION_PATH/mmdetection/data  && \
ln -s $MMLAB_DATASET_PATH/mmdetection/data $MMLAB_EXTENSION_PATH/mmdetection/data && \
rm -rf $MMLAB_EXTENSION_PATH/mmdetection3d/data  && \
ln -s $MMLAB_DATASET_PATH/mmdetection3d/data $MMLAB_EXTENSION_PATH/mmdetection3d/data 

```
