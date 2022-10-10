# Introduction

At present, in the field of autonomous driving, 3D detection methods based on Camera or Lidar emerge in an endless stream. For Camera, in order to achieve the purpose of 3D detection, the IPM (Inverse Perspective Mapping) problem must be solved. The mathematically reliable method must pass multiple frames or multiple perspectives. way to complete the "implicit reconstruction" of the scene in the temporal or spatial domain

2D images have very high information density, but none of the current methods can solve the problem of information loss in the IPM process. For example, the most promising BEV method currently has insufficient BEV features and a sharp drop in accuracy over long distances. and other problems, the essential cause of this problem is actually the problem of information loss during the IPM process. This problem will directly lead to the decline of Recall and Confidence in small target and long-distance detection tasks, which are crucial for solving the Corner Case of autonomous driving.

In addition, Camera-based 2D detection can complete relatively simple autonomous driving tasks at a lower cost, which is still very suitable for some limited scenarios. Therefore, Camera-based 2D detection tasks are still the direction that autonomous driving tasks cannot avoid.

This project is a 2D detection task framework, which can help users quickly complete tasks such as data preparation, model building, and model training, thereby improving the user's work efficiency

## Supported Tasks

The tasks supported by this project are as follows
- Single camera detection (completed)
- Single camera split (completed)
- Single camera multitasking (in development)
- Multi-camera detection (in development)
- Multi-camera segmentation (in development)
- Multi-camera multi-tasking (in development)

Among the above tasks, the same task corresponds to multiple network models, and users can choose the best one according to their own needs.

## Datasets

The datasets supported by this project are as follows
- BDD100K
- COCO
- VOC
- USD (custom dataset)

For the above data sets, scripts for mutual conversion are also provided. For details, please refer to the corresponding specific documents

## Tutorials

In order to facilitate everyone to get started quickly and improve their understanding of this project, this project also provides a series of nanny-level tutorials. These tutorials are generally based on common tasks in automatic driving. I hope to provide you with some ideas.

The tutorials provided by this project are as follows
- yolox : complete signal light detection through yolox
- yolopv2 : complete target detection, passable area segmentation, lane line segmentation through yolopv2

If you find bugs or documentation errors during use, you are very welcome to send an issue or pr, and I will fix it as soon as I receive the notification

Finally, if you think this project is helpful to you, I hope to give [admlops](https://github.com/windzu/admlops) a star, I will be very happy, thank you!