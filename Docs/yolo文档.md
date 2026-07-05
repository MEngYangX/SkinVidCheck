# yolo.py & config.py 使用文档

> 本文档详细说明 `yolo.py`（训练主脚本）和 `config.py`（训练配置）的设计、使用与扩展。
> 适用于 **YOLOv8 / YOLOv26** 皮肤病变检测（湿疹 vs 痤疮）训练场景。

***

## 📑 目录

- [1. 概述](#1-概述)
- [2. config.py 详解](#2-configpy-详解)
- [3. 使用示例](#5-使用示例)

***

## 1. 概述

| 文件          | 行数  | 核心职责                                                  |
| ----------- | --- | ----------------------------------------------------- |
| `config.py` | 144 | 集中管理训练超参数（路径、硬件、训练参数、增强、损失、验证阈值、模型归档）                 |
| `yolo.py`   | 665 | 训练主流程：环境准备 → 加载数据 → 训练 → 验证 → 测试 → 可视化归档 → best.pt 归档 |

两者关系：

```
config.py（静态配置） ──► yolo.py（动态执行）
                            ├── 读取 Config 类
                            ├── 构建运行时目录
                            ├── 清理项目根下的 runs/ 目录
                            ├── 调用 ultralytics.YOLO.train()
                            │     └─ project=OUTPUT_DIR → 不写 runs/
                            ├── 调用 .val() 重定向到 run_dir/val/、run_dir/test_results/
                            │     └─ 也不再写 runs/segment/...
                            ├── organize_charts() 归档图表到 analysis/ 下（含 val/、test_results/）
                            ├── archive_best_model() 把 best.pt 复制到 Config.MODELS_DIR
                            └── 生成中文图表与报告到 analysis/ val/ test_results/ 子目录
```

<br />

***

## 2. config.py 详解

### 2.1 文件结构

`config.py` 只有一个类 `Config`，所有配置都是**类属性**（静态变量），无需实例化。

```python
from config import Config
print(Config.EPOCHS)        # 300
print(Config.DATASET_ROOT)  # D:\SkinVidCheck\train_data
```

### 2.2 7 大配置类别

#### ① 路径配置（5 个）

| 参数             | 默认值                                      | 说明                                         |
| -------------- | ---------------------------------------- | ------------------------------------------ |
| `MODEL_PATH`   | `D:\SkinVidCheck\yolo26m-seg.pt`         | 预训练权重路径（也支持 ultralytics 内置名如 `yolov8n.pt`） |
| `DATASET_ROOT` | `D:\SkinVidCheck\train_data`             | 数据集根目录（含 `images/` 和 `labels/`）            |
| `CLASSES_FILE` | `D:\SkinVidCheck\train_data\classes.txt` | 类别定义文件（每行一个类别）                             |
| `OUTPUT_DIR`   | `D:\SkinVidCheck\training_output`        | 训练产物输出根目录                                  |
| `MODELS_DIR`   | `D:\SkinVidCheck\Models`                 | **新增**。best.pt 自动归档目录（命名 `<时间戳>.pt`）       |

#### ② 硬件配置（3 个）

| 参数               | 默认值    | 说明                                   |
| ---------------- | ------ | ------------------------------------ |
| `DEVICE`         | `"0"`  | `"0"`/`"0,1"` = GPU；`"cpu"` = CPU    |
| `WORKERS`        | `0`    | 数据加载线程数；**Windows 强烈建议设为 0**，避免多进程错误 |
| `HALF_PRECISION` | `True` | FP16 混合精度，可节省约 40% 显存                |

#### ③ 训练参数（10 个）

| 参数              | 默认值       | 说明                           | 调优建议               |
| --------------- | --------- | ---------------------------- | ------------------ |
| `IMG_SIZE`      | `800`     | 输入正方形边长（Letterbox 自动缩放）      | 医学图像建议 640-1024    |
| `EPOCHS`        | `300`     | 总训练轮数                        | 小数据集可设 100-150     |
| `BATCH_SIZE`    | `8`       | 批次大小                         | 显存不足时降为 4 或 2      |
| `LEARNING_RATE` | `0.0003`  | 初始学习率（lr0）                   | 微调建议 0.0001-0.0005 |
| `OPTIMIZER`     | `"AdamW"` | 优化器：`SGD/Adam/AdamW/RMSProp` | 小数据集推荐 AdamW       |
| `PATIENCE`      | `35`      | 早停耐心值（验证集 mAP 多少轮不升则停）       | 设为 EPOCHS 的 15-20% |
| `SEED`          | `42`      | 随机种子                         | 设 `None` 关闭        |
| `CLOSE_MOSAIC`  | `30`      | 最后 N 轮关闭 Mosaic 增强           | 建议为 EPOCHS 的 5-10% |
| `COS_LR`        | `True`    | 余弦学习率调度                      | 小数据集强烈建议开启         |
| `CACHE`         | `True`    | 是否缓存图像到内存                    | 小数据集开，节省 IO        |

#### ④ 数据增强（11 个）

| 参数            | 默认值    | 作用         | 医学图像建议               |
| ------------- | ------ | ---------- | -------------------- |
| `MOSAIC`      | `0.8`  | 4 图拼接增强    | 1.0（提升多样性）           |
| `MIXUP`       | `0.05` | 图像混合增强     | 0.0-0.1（避免混合病灶特征）    |
| `COPY_PASTE`  | `0.0`  | 复制粘贴目标     | 0.0（避免病灶位置不合理）       |
| `HSV_H`       | `0.03` | 色调抖动 ±3%   | **保持保守**，改变颜色 = 改变诊断 |
| `HSV_S`       | `0.4`  | 饱和度 ±40%   | 中等（模拟光照）             |
| `HSV_V`       | `0.4`  | 明度 ±40%    | 中等（模拟曝光）             |
| `DEGREES`     | `5.0`  | 旋转 ±5°     | ≤10°（避免病灶失真）         |
| `TRANSLATE`   | `0.15` | 平移 ±15%    | 中等                   |
| `SCALE`       | `0.6`  | 缩放 60-140% | 模拟拍摄距离               |
| `SHEAR`       | `0.0`  | 剪切变换       | 0.0（医学图像禁形变）         |
| `PERSPECTIVE` | `0.0`  | 透视变换       | 0.0（医学图像禁 3D 形变）     |
| `FLIPUD`      | `0.0`  | 上下翻转       | 0.0（医学图像通常禁用）        |
| `FLIPLR`      | `0.5`  | 左右翻转       | 0.5（皮肤对称部位可用）        |
| `ERASING`     | `0.1`  | 随机擦除概率     | 0.1-0.2              |

#### ⑤ 损失权重（5 个）

| 参数                 | 默认值    | 作用                   |
| ------------------ | ------ | -------------------- |
| `BOX_LOSS_WEIGHT`  | `8.0`  | 边界框回归损失权重（医学定位关键，调高） |
| `CLS_LOSS_WEIGHT`  | `0.6`  | 分类损失权重               |
| `DFL_LOSS_WEIGHT`  | `1.5`  | 分布焦点损失权重（YOLOv8+ 特有） |
| `POSE_LOSS_WEIGHT` | `12.0` | 关键点损失（检测任务忽略）        |
| `KOBJ_LOSS_WEIGHT` | `1.0`  | 关键点目标性损失             |

#### ⑥ 验证/测试（3 个）

| 参数               | 默认值    | 说明                  |
| ---------------- | ------ | ------------------- |
| `CONF_THRESHOLD` | `0.25` | 置信度阈值（低于此值的检测被丢弃）   |
| `IOU_THRESHOLD`  | `0.45` | NMS 的 IoU 阈值（重叠框合并） |
| `MAX_DET`        | `300`  | 单张图最大检测数            |

#### ⑦ 其他（4 个）

| 参数                | 默认值           | 说明                     |
| ----------------- | ------------- | ---------------------- |
| `MULTI_SCALE`     | `0.0`         | 多尺度训练（关 Mosaic 后可启用）   |
| `LABEL_SMOOTHING` | `0.0`         | 标签平滑（防过拟合，0.0-0.1）     |
| `AGNOSTIC_NMS`    | `False`       | 类别无关 NMS（医学图像建议 False） |
| `EXP_NAME`        | `'train_exp'` | 实验名（备用）                |

<br />

***

## 3. 使用示例

### 5.1 训练

```bash
python yolo.py
```

