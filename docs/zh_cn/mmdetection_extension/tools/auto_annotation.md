# Auto Annotation

借助open-mmlab的强大集成能力，我们使用其中精度最高的模型来帮助我们进行数据集的自动化标注

## How

不同类型的数据集往往有着不同的数据格式和不同的数据标注标准，我们将自动标注划分为了如下流程
- 解析待检测文件，包括但不限于
    - 从指定文件夹读取文件
    - 从指定配置文件读取文件
- 构建模型检测
- 对检测结果后处理，社区不符合目标数据集的结果
- 保存为相应格式的标注文件

## Usage

### Scalabel

> 自动化标注Scalabel格式的数据

Scalabel格式数据的文件列表保存在一个文件列表中，通过其中的url方可访问到该数据共需要指定如下几个参数
- input : scalabel数据所存储的数据列表文件路径
- type : 待自动标注的数据集格式
- config : 所使用的模型的 config
- checkpoint : 所使用的模型的 checkpoint

```bash
conda activate mmdet && \
cd $ADMLOPS_PATH/mmdetection_extension/tools/auto_annotation && \
python main.py \
--input /home/wind/Projects/wind_activate/scalable_docker/items/local_weitang_image_list1.json \
--type scalabel \
--config $ADMLOPS_PATH/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py \
--checkpoint $ADMLOPS_PATH/checkpoints/mmdet/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
```

在检测完毕后，会在输入路径的根目录下，生成一个后缀名为 auto_annotation 的文件