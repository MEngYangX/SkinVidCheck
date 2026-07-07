#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置文件
"""

from pathlib import Path

class Config:
    """
    训练配置类
    硬件环境: RTX 5070 12GB
    """
    
    # ==================== 路径配置 ====================
    # 预训练模型路径
    MODEL_PATH = r"D:\SkinVidCheck\yolo26m-seg.pt"
    
    # 数据集根目录
    DATASET_ROOT = r"D:\SkinVidCheck\train_data"
    
    # 类别定义文件路径
    CLASSES_FILE = r"D:\SkinVidCheck\train_data\classes.txt"
    
    # 训练输出目录
    OUTPUT_DIR = r"D:\SkinVidCheck\training_output"

    # 模型归档目录（训练完成后自动将 best.pt 复制至此，命名为 <时间戳>.pt）
    MODELS_DIR = r"D:\SkinVidCheck\Models"
    
    # ==================== 硬件配置 ====================
    # GPU 设备编号，"0" 表示第一块 GPU，"cpu" 表示 CPU
    DEVICE = "0"
    
    # 数据加载线程数，Windows 建议设为 0 避免多进程错误
    WORKERS = 0
    
    # 半精度训练（FP16）
    HALF_PRECISION = True
    
    # ==================== 训练参数 ====================
    # 输入图像尺寸（正方形）
    IMG_SIZE = 800
    
    # 总训练轮数
    EPOCHS = 300
    
    # 批次大小
    BATCH_SIZE = 8
    
    # 初始学习率
    # YOLOv26x 使用 AdamW 时，小数据集建议 0.0003-0.0005
    LEARNING_RATE = 0.0003

    # 最终学习率 = lr0 * lrf，YOLOv26 默认 lrf=0.01
    # 即最终降至 0.000003，适合精细微调
    
    # 优化器类型  YOLOv26 支持: 'SGD', 'Adam', 'AdamW', 'RMSProp'
    OPTIMIZER = "AdamW"
    
    # 早停耐心值（验证集 mAP 不提升的轮数）
    PATIENCE = 45
    
    # 随机种子，保证结果可复现
    SEED = 42

    # 关闭 Mosaic 的轮数（最后 N 轮关闭，稳定训练）  建议设为总轮数的 5-10%
    CLOSE_MOSAIC = 20
    
    # 是否使用余弦学习率调度
    COS_LR = True
    
    # 是否缓存图像到内存
    CACHE = True
    
    # ==================== 数据增强参数 ====================
    # YOLOv26 核心增强参数
    
    # Mosaic 增强：4图拼接，小数据集建议 1.0 提升多样性  注意：医学图像如果病灶位置关键，可适当降低至 0.5-0.8
    MOSAIC = 0.8
    
    # MixUp 增强：图像混合  医学图像建议 0.0-0.1，过高会混合病变特征导致误诊
    MIXUP = 0.05
    
    # Copy-Paste 增强：复制粘贴目标（YOLOv26 新参数）  医学图像建议 0.0，避免病灶位置不合理
    COPY_PASTE = 0.0
    
    # HSV 色彩空间增强
    # 皮肤病变颜色诊断价值高，保持保守
    HSV_H = 0.03      # 色调：±3%，避免改变病变颜色本质
    HSV_S = 0.4        # 饱和度：±40%，模拟光照变化 
    HSV_V = 0.4        # 明度：±40%，模拟曝光差异
    
    # 几何变换
    DEGREES = 5.0      # 旋转：±5度，医学图像不宜过大
    TRANSLATE = 0.15   # 平移：±15%，模拟病灶偏移
    SCALE = 0.6        # 缩放：60%-140%，模拟拍摄距离变化
    SHEAR = 0.0        # 剪切：医学图像建议 0，避免形变
    
    # 透视变换（YOLOv26 新参数）  医学图像建议 0，避免三维形变导致病变形状失真
    PERSPECTIVE = 0.0
    
    # 翻转增强
    FLIPUD = 0.0       # 上下翻转：医学图像通常禁用
    FLIPLR = 0.5       # 左右翻转：通常启用，皮肤对称部位可用
    
    # 随机擦除（Erasing，YOLOv26 参数）  医学图像建议 0.1-0.2
    ERASING = 0.1
    
    # ==================== 损失函数权重 ====================
    # YOLOv26 损失权重调整
    
    # 边界框回归损失权重  医学检测定位精度重要，保持 7.5 或略增至 8.0
    BOX_LOSS_WEIGHT = 8.0
    
    # 分类损失权重
    CLS_LOSS_WEIGHT = 0.6
    
    # 分布焦点损失权重（DFL，YOLOv26 特有）
    DFL_LOSS_WEIGHT = 1.5  # 默认值
    
    # 关键点损失权重（YOLOv26 支持姿态估计，检测任务忽略）
    POSE_LOSS_WEIGHT = 12.0  # 默认值
    
    # 关键点目标性损失权重
    KOBJ_LOSS_WEIGHT = 1.0   # 默认值
    
    # ==================== 验证/测试参数 ====================
    # 置信度阈值
    CONF_THRESHOLD = 0.25
    
    # NMS IoU 阈值
    IOU_THRESHOLD = 0.45
    
    # 最大检测数（单图）
    MAX_DET = 300
    
    # ==================== 其他 YOLOv26 参数 ====================
    # 多尺度训练（随机关闭 Mosaic 时启用）
    MULTI_SCALE = 0.0
    
    # 标签平滑（YOLOv26 分类参数）
    LABEL_SMOOTHING = 0.0
    
    # NMS 时是否合并重叠框（医学图像建议 False，避免漏检）
    AGNOSTIC_NMS = False
    
    # 训练实验名称
    EXP_NAME = 'train_exp'