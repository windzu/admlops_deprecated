# 引言

> 作者本人目前为自动驾驶感知方向，所以功能实现时候会优先自动驾驶方向

一个深度学习任务的`闭环`简单可以分为如下三个部分

- 数据：采集、标注、数据管理
- 模型：网络搭建、训练、测试、评估
- 部署：模型部署

随着任务复杂性的提高，依赖单人完成全流程工作变的困难起来，而如果依赖多人的合作，又往往会出现因为水平不匹配、工作内容冲突带来的合作者间的“扯皮”问题，针对此情况，业界提出了`MLOps`的概念

但目前就本人调研来看，还没有成熟的`MLOps`解决方案，众多开源工具也是限制颇多，所以在没有成熟的好用工具之前，只能自己利用开源工具简单你的拼凑一个出来使用了～

本工程是依赖[open-mmlab](https://github.com/open-mmlab)中的一系列工具完成了MLOps任务中的modeling和deployment的工作，因为其中所主要使用的mmdeploy和mmdetection3d还在快速的更新，所以想使用本工程的工程师，请熟读相关文档以及update comment，并很希望大家能合作交流，多提issues和pr，共同打造一个趁手好用的工作，多留出一些时间来摸鱼

# open-mmlab介绍

> 因为本工程是在openmmlab的基础上做了一些小修改而搭建起来的，所以使用好本工程的前提是对open-mmlab的相关工作要有一定的了解，下面是相关简介

open-mmlab中的系列框架是有依赖关系的，其本质是OO(面向对象)的继承表现

想快速了解open-mmlab一定要先了解`工厂设计模式`，在open-mmlab是通过registey来实现的，详情参考：[registry详解](https://zhuanlan.zhihu.com/p/355271993)或者是对应的代码(代码很短，推荐看一下)

**简单介绍一下个人常用的其中几个工程**

- mmcv：open-mmlab中的基础库
  
    open-mmlab其他各个框架中的基础功能都是来自于它或者继承自它

- mmdetection：2d检测、分割的集成框架
  
    提供该领域里程碑式的模型支持，以及其他常见的口碑较好的模型支持，例如RCNN系列、yolo系列等。更新速度一般晚于学术界半年左右

- mmdetection3d：3d检测、分割的集成框架
  
    是在mmdetection基础上修改而来的，毕竟很多3d检测的方法用的其实还是2d的方法，更新速度一般晚于学术界半年左右，但是因为随着自动驾驶的火热以及2D赛道卷不动了大家纷纷转向3D，所以其最近更新比较“迅猛”

- mmdeploy：一个部署框架
  
    其出现的目的是为了打通open-mmlab中项目落地的最后一步，其更新速度最慢，一般是一个已经非常成熟的模型才会被支持(话说回来，如果支持速度很快，那大家都失业吧😬)   各个被支持的模型一般支持多种后端，最少是支持onnx和tensorrt   目前还没有到v1.0的正式版    对于被支持的模型，有的还会贴心的提供相关的sdk，这样连前后处理都不需要自己写了～

# 环境配置

> 为了后续使用的方便，最好按照如下需求准备好必要的checkpoints、data、环境变量的配置

## 准备数据

> 准备训练、测试需要的数据、预训练的checkpoints，并按照指定结构放置，用于后续配置数据软链接

### checkpoints数据

文件结构如下

```bash
mmlab_checkpoints
├── mmdetection
│   └── checkpoints
└── mmdetection3d
    └── checkpoints
```

### data数据

文件结构如下

```bash
mmlab_dataset
├── mmdetection
│   └── data
└── mmdetection3d
    └── data
```

## 拉取工程

> 为了减少数据量，一般仅拉取最新的commit

```bash
git clone https://github.com/windzu/mmlab_extension.git --depth 1 && \
cd mmlab_extension && \
git submodule update --init --recursive
```

## 添加环境变量

> 根据自己使用的shell来配置，这里仅以bash为例

```bash
cd mmlab_extension && \
echo "export MMLAB_EXTENSION_PATH=$(pwd)" >> ~/.bashrc 

cd mmlab_checkpoints && \
echo "export MMLAB_CHECKPOINTS_PATH=$(pwd)" >> ~/.bashrc

cd mmlab_dataset && \
echo "export MMLAB_DATASET_PATH=$(pwd)" >> ~/.bashrc
```

## 配置数据软链接

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
