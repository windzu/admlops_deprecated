# BDD100K

BDD100K是一个包含多种2D任务的视觉数据库，其包含任务如下
- Object Detection
- Semantic Segmentation
- Multiple Object Tracking
- Instance Segmentation
- Drivable Area
- Segmentation Tracking
- Pose Estimation

## Label Format

> 关于label的详细介绍可以参考其[官方文档](https://doc.bdd100k.com/format.html) 或者 scalabel中关于格式描述的[文档](https://doc.scalabel.ai/format.html#),因为bdd100k的标注是遵循scalabel标准制定的

关于bdd100k中label格式的介绍，其官方文档写的已经非常清晰了，下面罗列一些重要的信息以防被忽略

- Object Detection 任务中类别的label id **从1开始，而不是0**
- 在 Eval 阶段，有些类别不予考虑
- 


## Format Conversion

### to_color

### to_coco