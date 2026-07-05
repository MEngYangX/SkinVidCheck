# SkinVidCheck

<p align="center">
  基于 YOLOv26-seg 的皮肤病变检测
</p>

<p align="center">
  <sub><i>注：本项目部分文档与代码由 AI 辅助生成。</i></sub>
</p>

<p align="center">
  <a href="https://github.com/MEngYangX/SkinVidCheck/stargazers">
    <img src="https://img.shields.io/github/stars/MEngYangX/SkinVidCheck?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/MEngYangX/SkinVidCheck/network/members">
    <img src="https://img.shields.io/github/forks/MEngYangX/SkinVidCheck?style=social" alt="GitHub forks">
  </a>
  <a href="https://github.com/MEngYangX/SkinVidCheck/issues">
    <img src="https://img.shields.io/github/issues/MEngYangX/SkinVidCheck" alt="GitHub issues">
  </a>
  <a href="https://github.com/MEngYangX/SkinVidCheck/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/MEngYangX/SkinVidCheck" alt="License">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python 3.10">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/PyTorch-2.11%2Bcu130-red?logo=pytorch" alt="PyTorch 2.11+cu130">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/ultralytics-8.4.26-0099ff" alt="ultralytics 8.4.26">
  </a>
</p>

***

## 项目简介

SkinVidCheck 是一个面向**皮肤病变检测**的深度学习项目，采用 **YOLOv26-seg 分割模型**，目前支持 **湿疹识别**，并预留扩展到 22 类皮肤病的可能。

项目核心流程：

1. 用 **Label Studio** 对皮肤图像进行标注
2. 将标注结果转换为 YOLO 格式
3. 用 `dataset_splitter.py` 按 8:1:1 切分为 train/val/test
4. 用 `yolo.py` 一键完成训练、验证、测试、可视化与模型归档

> 所有训练产物按时间戳保存在 `training_output/<时间戳>/` 下，最佳权重自动归档到 `Models/<时间戳>.pt`，无需手动整理。

***

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/MEngYangX/SkinVidCheck.git
cd SkinVidCheck
```

### 2. 创建并激活 Conda 环境

```bash
conda env create -f environment.yml
conda activate SkinVidCheck
```

### 3. 验证环境

```bash
# PyTorch + CUDA 检查
python Test/test_pytorch.py

# ultralytics 基础推理测试
python Test/test_yolo.py
```

### 4. 准备数据

```bash
# 将 MyData/finish 按 8:1:1 切分为 train_data
python dataset_splitter.py -i MyData/finish -o train_data
```

如果你已有固定的 val/test 集，只想重新生成 train：

```bash
python dataset_splitter.py -i MyData/finish -o train_data --use-existing-split
```

### 5. 开始训练

```bash
python yolo.py
```

训练完成后：

- 完整训练产物：`training_output/<时间戳>/`
- 图表与报告：`training_output/<时间戳>/analysis/`
- 最佳权重：`training_output/<时间戳>/weights/best.pt`
- 自动归档的模型：`Models/<时间戳>.pt`

***

## 数据集

本项目训练数据来源于以下公开 Kaggle 数据集：

| 数据集                                         | 类别示例                          | 来源                                                                                                                                       |
| ------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Skin Diseases**                           | 感染性皮肤病、湿疹、痤疮、色素性疾病、良性肿瘤、恶性肿瘤  | [Kaggle - Skin Diseases](https://www.kaggle.com/datasets/ascanipek/skin-diseases)                                                        |
| **Augmented Skin Conditions Image Dataset** | 痤疮、皮肤癌、湿疹、角化症、粟丘疹、玫瑰痤疮        | [Kaggle - Augmented Skin Conditions Image Dataset](https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset) |
| **Skin Disease Dataset**                    | 痤疮、湿疹、银屑病、白癜风、皮肤癌、狼疮等 23 类皮肤病 | [Kaggle - Skin Disease Dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)                                            |

***

## 免责声明

1. 本项目及其模型**仅供学术研究与技术学习使用**，不构成任何医疗诊断或治疗建议。
2. 模型输出结果**不可直接用于临床决策**。若涉及健康问题，请务必咨询具有执业资格的专业医疗人员。
3. 项目作者不对因使用本项目代码、模型或数据而产生的任何直接或间接后果承担责任。

***

## 文档

| 文档                                                        | 内                            |
| --------------------------------------------------------- | ---------------------------- |
| [Docs/项目结构.md](Docs/项目结构.md)                              | 完整项目结构说明                     |
| [Docs/yolo文档.md](Docs/yolo文档.md)                          | `yolo.py` 与 `config.py` 详细文档 |
| [Docs/dataset\_splitter文档.md](Docs/dataset_splitter文档.md) | 数据集切分工具文档                    |
| [Docs/label-studio使用教程.md](Docs/label-studio使用教程.md)      | Label Studio 标注教程            |
| [Docs/常用命令.md](Docs/常用命令.md)                              | 项目常用命令速查                     |

***

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。
