# dataset\_splitter.py 使用文档

***

## 📑 目录

- [1. 概述](#1-概述)
- [2. 安装与运行](#2-安装与运行)
- [3. 命令行参数](#3-命令行参数)
- [4. 输入输出结构](#4-输入输出结构)
- [5. 使用示例](#7-使用示例)

***

## 1. 概述

| 项      | 说明                                                      |
| ------ | ------------------------------------------------------- |
| 依赖     | 仅 Python 标准库（`argparse / shutil / random / pathlib`）    |
| 作用     | 把 `images/` 和 `labels/` 两个文件夹**配对**切分为 `train/val/test` |
| 默认比例   | `8:1:1`（训练:验证:测试）                                       |
| 输出结构   | 完整的 YOLO 数据集布局                                          |
| 默认随机种子 | `42`                                                    |

**典型调用**：

```bash
python dataset_splitter.py -i MyData/finish -o train_data
```

***

## 2. 安装与运行

无需安装任何第三方包（仅使用 Python 标准库）。

```bash
# 直接运行
python dataset_splitter.py --help

# 标准用法
python dataset_splitter.py -i <输入目录> -o <输出目录>
```

<br />

***

## 3. 命令行参数

```
usage: dataset_splitter.py [-h] -i INPUT -o OUTPUT [--seed SEED]
                            [--ratio RATIO] [--ext EXT]
                            [--use-existing-split]

将数据集按8:1:1比例分割为训练集、验证集和测试集

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT   输入文件夹路径（包含 images 和 labels 子文件夹）
  -o OUTPUT, --output OUTPUT  输出文件夹路径
  --seed SEED           随机种子（默认 42）
  --ratio RATIO         分割比例，格式为 train:val:test（默认 8:1:1）
  --ext EXT             指定图像扩展名（如 .jpg/.png），不指定则处理所有文件
  --use-existing-split  使用输出目录中已存在的 val/test 作为固定集，其余所有
                        图像全部作为训练集（不进行随机分割）。val/test 文件夹
                        必须已存在且不会被修改。
```

### 参数详解

| 参数                     | 缩写   | 必填 | 默认      | 说明                                                                 |
| ---------------------- | ---- | -- | ------- | ------------------------------------------------------------------ |
| `--input`              | `-i` | ✅  | —       | 输入根目录（含 `images/` 和 `labels/` 两个子目录）                               |
| `--output`             | `-o` | ✅  | —       | 输出根目录（会自动创建 `images/{train,val,test}` 和 `labels/{train,val,test}`） |
| `--seed`               | —    | ❌  | `42`    | 随机种子，用于可复现的切分（仅随机模式）                                               |
| `--ratio`              | —    | ❌  | `8:1:1` | 切分比例，格式 `train:val:test`（不要求和为 100，会自动归一化，仅随机模式）                   |
| `--ext`                | —    | ❌  | `None`  | 限定图像扩展名（如 `.jpg`）；不指定时处理 `images/` 下所有文件                           |
| `--use-existing-split` | —    | ❌  | `False` | **新增**。使用输出目录中已存在的 val/test 作为固定集，详见 §5.10                         |

<br />

***

## 4. 输入输出结构

### 输入目录结构（必须）

```
<输入目录>/
├── images/           # 图像文件（jpg/png/jpeg/...）
│   ├── img001.jpg
│   ├── img002.png
│   ├── img003.jpeg
│   └── ...
└── labels/           # 标签文件（与图像同名的 .txt/.xml/.json/...）
    ├── img001.txt
    ├── img002.txt
    ├── img003.json
    └── ...
```

**配对规则**：

- 图像与标签的**文件名（去扩展名）必须相同**
- 例：`img001.jpg` 配对 `img001.txt`
- 标签支持 `.txt / .xml / .json / .yaml / .yml`（甚至无扩展名）
- 找不到配对标签的图像**会被跳过**（打印警告）

### 输出目录结构（自动生成）

```
<输出目录>/
├── images/
│   ├── train/        # 训练集图像
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/          # 验证集图像
│   └── test/         # 测试集图像
├── labels/
│   ├── train/        # 训练集标签
│   ├── val/
│   └── test/
└── split_summary.txt  # 切分摘要（自动生成）
```

<br />

***

## 5. 使用示例

### 5.1 标准用法（项目内）

```bash
# 切分 MyData/finish → train_data（8:1:1）
python dataset_splitter.py -i MyData/finish -o train_data
```

### 5.2 自定义比例 7:2:1

```bash
python dataset_splitter.py -i MyData/finish -o train_data --ratio 7:2:1
```

### 5.3 指定图像扩展名

```bash
# 只处理 .jpg
python dataset_splitter.py -i data -o out --ext .jpg

# 只处理 .png
python dataset_splitter.py -i data -o out --ext .png
```

### 5.4 固定随机种子（可复现）

```bash
python dataset_splitter.py -i data -o out --seed 12345
```

### 5.5 二次切分（增量更新）

如要向现有数据集追加新数据：

```bash
# 把新数据放到原 input 目录再次切分（已存在的会被跳过，不会重复）
python dataset_splitter.py -i MyData/finish_v2 -o train_data
```

输出日志会显示：

```
  train: 已复制 100 对文件，跳过 400 对已存在
  val:   已复制 20 对文件，跳过 50 对已存在
  test:  已复制 30 对文件，跳过 50 对已存在
```

⚠️ **注意**：这种方式会把新数据随机分到 3 个 split 中，**不会破坏旧的切分**。

### 5.6 固定分割模式（保持 val/test 不变）🆕

**场景**：你已经有了一套固定的 val/test（例如：医学数据集按患者 ID 预分割），想**只重新生成 train 集**而不动 val/test。

**前提条件**：`output_dir/images/val/` 和 `output_dir/images/test/` 必须已存在。

```bash
# 假设你已经手工准备好了 train_data/images/{val,test}/
# 现在想用 MyData/finish 中的所有"非 val/test"图片重新生成 train/

python dataset_splitter.py -i MyData/finish -o train_data --use-existing-split
```

**实际输出**：

```
[固定分割模式] 使用已存在的 val/test 作为固定集
  val 固定集：100 张
  test 固定集：101 张
  训练集：801 张（input 排除 val/test base 名后）
  ⚠ 已排除 201 个 input 文件（与 val/test 同 base 名）

  同步 train/ 目录（清空后重新填充）...

开始复制文件...

处理训练集...
  train: 已复制 801 对文件，跳过 0 对已存在
  ✓ val/100 张保持不动
  ✓ test/101 张保持不动

==================================================
固定分割模式完成：
  ✓ 向 train/ 中添加了 801 个文件（input 排除 val/test 同名后）
  ✓ val/ 保持原样（100 张）
  ✓ test/ 保持原样（101 张）
==================================================
```

**与普通模式的区别**：

- ❌ 不解析 `--ratio`（参数被忽略）
- ❌ 不解析 `--seed`（参数被忽略）
- ❌ 不调用 `split_data()`（无随机打乱）
- ✅ train/ 目录会**先清空再填充**（保证 train/val/test 互不重复）
- ✅ val/test 文件夹**完全不动**（不复制、不删除、不覆盖）

**适用场景**：

- 固定测试集（模型评估必须用同一批测试集）
- 按患者 ID 预分割的医学数据
- 重新生成 train 但保留验证集（如调整了训练数据）
- 半监督学习（val/test 固定，train 含大量无标签数据）

### 5.7 在 Python 代码中调用

虽然 `dataset_splitter.py` 设计为命令行工具，但你也可以在脚本中复用函数：

```python
from pathlib import Path
from dataset_splitter import (
    parse_ratio, get_paired_files, create_directory_structure,
    split_data, copy_files
)

input_dir = Path("MyData/finish")
output_dir = Path("train_data")

# 配对扫描
paired = get_paired_files(
    input_dir / "images",
    input_dir / "labels",
    ext=".jpg"  # 可选
)

# 创建目录
folders = create_directory_structure(output_dir)

# 切分
train, val, test = split_data(paired, 0.8, 0.1, 0.1, seed=42)

# 复制
copy_files(train, folders['images_train'], folders['labels_train'], "train")
copy_files(val,   folders['images_val'],   folders['labels_val'],   "val")
copy_files(test,  folders['images_test'],  folders['labels_test'],  "test")
```

