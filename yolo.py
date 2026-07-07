#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 皮肤病变检测训练脚本
修复：验证指标显示、目录结构按时间戳组织、统一图表归档
"""

import os
import shutil
import yaml
import torch
import platform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# 导入配置文件
from config import Config

# ==================== 中文字体配置 ====================

def setup_chinese_font():
    """配置 Matplotlib 支持中文显示"""
    system = platform.system()
    
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[字体配置] 当前系统: {system}, 已设置中文字体支持")


# ==================== 数据集配置 ====================

def create_data_yaml(config: Config, run_dir: Path):
    """自动生成 data.yaml 配置文件"""
    dataset_root = Path(config.DATASET_ROOT)
    
    # 检查必要目录
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test"
    ]
    
    print("\n[数据集检查] 检查必要目录...")
    for dir_path in required_dirs:
        if dir_path.exists():
            try:
                file_count = len(list(dir_path.iterdir()))
                print(f"  ✓ {dir_path} ({file_count} 个文件)")
            except Exception as e:
                print(f"  ⚠ {dir_path} (无法读取: {e})")
        else:
            print(f"  ✗ {dir_path} 不存在")
    
    # 读取类别
    classes = []
    if os.path.exists(config.CLASSES_FILE):
        try:
            with open(config.CLASSES_FILE, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"\n[类别信息] 从 classes.txt 读取: {classes}")
        except Exception as e:
            print(f"[警告] 读取类别文件失败: {e}")
            classes = ["湿疹", "痤疮"]  # 使用默认类别
    else:
        print(f"[警告] 未找到类别文件，使用默认类别")
        classes = ["湿疹", "痤疮"]
    
    # 构建 yaml
    data_config = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    # 将 data.yaml 放在运行目录下
    output_path = run_dir / "data.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    print(f"[配置生成] 已生成: {output_path}")
    return str(output_path)


# ==================== CUDA 检查 ====================

def setup_cuda():
    """检查 CUDA 环境"""
    print("\n[硬件检查] CUDA 状态...")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {gpu_name} ({gpu_memory:.2f} GB)")
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"  警告: 获取 GPU 信息失败: {e}")
            return False
    return False


# ==================== 图表归档 ====================

def organize_charts(run_dir: Path, analysis_dir: Path):
    """
    将训练 / 验证 / 测试阶段生成的所有图表归档到 analysis 文件夹

    目标目录结构：
        analysis/
        ├── *curve*.png, *matrix*.png, results.png, labels*.jpg, train_batch*.jpg, val_batch*.jpg   ← 训练阶段
        ├── val/                 ← 验证阶段（来自 run_dir/val/）
        └── test_results/        ← 测试阶段（来自 run_dir/test_results/）
    """
    print(f"\n[图表归档] 整理分析图表到: {analysis_dir}")

    # 确保 analysis 目录存在
    analysis_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    # ===== 阶段 1: 训练阶段图表（run_dir/ 根目录 → analysis/ 根目录） =====
    train_chart_patterns = [
        '*curve*.png',      # F1_curve, P_curve, PR_curve, R_curve
        '*matrix*.png',     # confusion_matrix, confusion_matrix_normalized
        'results.png',      # 训练结果汇总图
        'labels*.jpg',      # labels.jpg, labels_correlogram.jpg
        'train_batch*.jpg', # 训练批次可视化
        'val_batch*.jpg',   # 训练过程中产生的验证样本可视化
    ]

    for pattern in train_chart_patterns:
        for file_path in run_dir.glob(pattern):
            if file_path.is_file():
                target_path = analysis_dir / file_path.name
                try:
                    if target_path.exists():
                        target_path.unlink()
                    shutil.move(str(file_path), str(target_path))
                    print(f"  ✓ {file_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ {file_path.name} (移动失败: {e})")

    # ===== 阶段 2: 验证 / 测试阶段图表（run_dir/{val,test_results}/ → analysis/{val,test_results}/） =====
    # 注：model.val() 时已通过 project=str(run_dir) 让 ultralytics 直接写到这两个子目录
    for subdir_name in ['val', 'test_results']:
        src = run_dir / subdir_name
        dst = analysis_dir / subdir_name
        if not src.exists() or not src.is_dir():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for file_path in src.iterdir():
            if file_path.is_file():
                target_path = dst / file_path.name
                try:
                    if target_path.exists():
                        target_path.unlink()
                    shutil.move(str(file_path), str(target_path))
                    print(f"  ✓ {subdir_name}/{file_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ {subdir_name}/{file_path.name} (移动失败: {e})")
        # 移动完文件后清理空目录（保持 run_dir 干净）
        try:
            if not any(src.iterdir()):
                src.rmdir()
                print(f"  ✓ 已清理空目录: {src.name}/")
        except Exception as e:
            print(f"  ⚠ 清理 {src.name}/ 失败: {e}")

    if moved_count == 0:
        print("  (未找到需要归档的图表)")
    else:
        print(f"  共归档 {moved_count} 个文件")


# ==================== 可视化（全中文） ====================

def plot_training_results(csv_path: Path, analysis_dir: Path):
    """
    绘制中文训练过程图表
    
    Args:
        csv_path: results.csv 的路径
        analysis_dir: 分析图表输出目录
    """
    print("\n[可视化] 生成中文训练图表...")
    
    if not csv_path.exists():
        print(f"  警告: 未找到 {csv_path}，跳过图表生成")
        return
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print("  警告: CSV 文件为空")
            return
            
        epochs = df['epoch'].values
        
        # 创建分析目录
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建大图 (2行4列)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('YOLOv8 训练过程监控面板', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 训练损失 - 边界框
        ax = axes[0, 0]
        ax.plot(epochs, df['train/box_loss'], 'b-', linewidth=2, label='训练损失')
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('损失值', fontsize=11)
        ax.set_title('边界框回归损失 (Box Loss)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. 训练损失 - 分类
        ax = axes[0, 1]
        ax.plot(epochs, df['train/cls_loss'], 'r-', linewidth=2, label='训练损失')
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('损失值', fontsize=11)
        ax.set_title('分类损失 (Classification Loss)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. 训练损失 - DFL
        ax = axes[0, 2]
        if 'train/dfl_loss' in df.columns:
            ax.plot(epochs, df['train/dfl_loss'], 'g-', linewidth=2, label='训练损失')
            ax.set_title('分布焦点损失 (DFL Loss)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'DFL Loss\n未启用', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('损失值', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. 验证损失汇总
        ax = axes[0, 3]
        ax.plot(epochs, df['val/box_loss'], 'b-', linewidth=2, label='边界框损失', alpha=0.8)
        ax.plot(epochs, df['val/cls_loss'], 'r-', linewidth=2, label='分类损失', alpha=0.8)
        if 'val/dfl_loss' in df.columns:
            ax.plot(epochs, df['val/dfl_loss'], 'g-', linewidth=2, label='DFL损失', alpha=0.8)
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('验证损失', fontsize=11)
        ax.set_title('验证集损失对比', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 精确率曲线
        ax = axes[1, 0]
        ax.plot(epochs, df['metrics/precision(B)'], 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('精确率 (Precision)', fontsize=11)
        ax.set_title('边界框精确率 (Box Precision)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 6. 召回率曲线
        ax = axes[1, 1]
        ax.plot(epochs, df['metrics/recall(B)'], 'orange', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('召回率 (Recall)', fontsize=11)
        ax.set_title('边界框召回率 (Box Recall)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 7. mAP@0.5 曲线
        ax = axes[1, 2]
        ax.plot(epochs, df['metrics/mAP50(B)'], 'darkgreen', linewidth=2, marker='^', markersize=4, label='mAP@0.5')
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('mAP 值', fontsize=11)
        ax.set_title('平均精度 mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(1.0, df['metrics/mAP50(B)'].max() * 1.1))
        ax.grid(True, alpha=0.3)
        
        # 8. mAP@0.5:0.95 曲线
        ax = axes[1, 3]
        ax.plot(epochs, df['metrics/mAP50-95(B)'], 'darkblue', linewidth=2, marker='d', markersize=4, label='mAP@0.5:0.95')
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
        ax.set_ylabel('mAP 值', fontsize=11)
        ax.set_title('平均精度 mAP@0.5:0.95', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(1.0, df['metrics/mAP50-95(B)'].max() /90*1.1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = analysis_dir / "训练过程监控图.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  已保存: {plot_path}")
        
        # 单独生成损失对比图（更清晰）
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('训练 vs 验证 损失对比', fontsize=14, fontweight='bold')
        
        # 训练损失
        ax1.plot(epochs, df['train/box_loss'], label='边界框损失', linewidth=2, color='blue')
        ax1.plot(epochs, df['train/cls_loss'], label='分类损失', linewidth=2, color='red')
        ax1.set_xlabel('训练轮次', fontsize=12)
        ax1.set_ylabel('损失值', fontsize=12)
        ax1.set_title('训练集损失', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 验证损失
        ax2.plot(epochs, df['val/box_loss'], label='边界框损失', linewidth=2, color='blue')
        ax2.plot(epochs, df['val/cls_loss'], label='分类损失', linewidth=2, color='red')
        ax2.set_xlabel('训练轮次', fontsize=12)
        ax2.set_ylabel('损失值', fontsize=12)
        ax2.set_title('验证集损失', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_path = analysis_dir / "损失对比图.png"
        plt.savefig(loss_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  已保存: {loss_path}")
        
        plt.close('all')
        
    except Exception as e:
        print(f"  图表生成警告: {e}")
        import traceback
        traceback.print_exc()


# ==================== 辅助函数 ====================

def get_metric(results, key_with_b, key_without_b, default=0.0):
    """
    安全地获取指标，尝试带 (B) 后缀和不带后缀的键名
    """
    if results is None or not hasattr(results, 'results_dict'):
        return default
    
    rd = results.results_dict
    
    # 首先尝试带 (B) 的键名
    if key_with_b in rd:
        return rd[key_with_b]
    
    # 然后尝试不带 (B) 的键名
    if key_without_b in rd:
        return rd[key_without_b]
    
    # 最后尝试从 box 对象获取（如果是 DetMetrics 对象）
    try:
        if hasattr(results, 'box'):
            if 'mAP50' in key_with_b and hasattr(results.box, 'map50'):
                return results.box.map50
            if 'mAP50-95' in key_with_b and hasattr(results.box, 'maps'):
                return results.box.maps
            if 'precision' in key_with_b and hasattr(results.box, 'mp'):
                return results.box.mp
            if 'recall' in key_with_b and hasattr(results.box, 'mr'):
                return results.box.mr
    except:
        pass
    
    return default


def _generate_model_readme(
    dst_dir: Path,
    run_dir: Path,
    val_results,
    test_results,
) -> Path:
    """
    在归档目录下生成 README.md，含模型概述与性能指标

    Args:
        dst_dir: 归档目录（即 Models/<时间戳>/）
        run_dir: 训练输出目录（用于从 run_dir.name 取时间戳）
        val_results: ultralytics 验证结果对象（可为 None）
        test_results: ultralytics 测试结果对象（可为 None）

    Returns:
        README.md 的路径
    """

    def _fmt(results, key_with_b, key_without_b):
        if results is None:
            return "N/A"
        return f"{get_metric(results, key_with_b, key_without_b, 0.0):.4f}"

    model_basename = os.path.basename(Config.MODEL_PATH)
    timestamp = run_dir.name
    archived_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    readme_path = dst_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 模型归档\n\n")
        f.write(f"- 训练时间戳：`{timestamp}`\n")
        f.write(f"- 归档时间：{archived_at}\n\n")

        f.write("## 模型概述\n\n")
        f.write("| 项目 | 数值 |\n")
        f.write("| --- | --- |\n")
        f.write(f"| 底座模型 | `{model_basename}` |\n")
        f.write(f"| 输入尺寸 | {Config.IMG_SIZE} × {Config.IMG_SIZE} |\n")
        f.write(f"| 训练轮数 | {Config.EPOCHS}（早停 patience={Config.PATIENCE}） |\n")
        f.write(f"| 批次大小 | {Config.BATCH_SIZE} |\n")
        f.write(f"| 优化器 | {Config.OPTIMIZER}，学习率 {Config.LEARNING_RATE} |\n")
        f.write(f"| 数据集 | `{Config.DATASET_ROOT}` |\n\n")

        f.write("## 性能指标\n\n")
        f.write("### 验证集\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("| --- | --- |\n")
        f.write(f"| mAP@0.5 | {_fmt(val_results, 'metrics/mAP50(B)', 'metrics/mAP50')} |\n")
        f.write(f"| mAP@0.5:0.95 | {_fmt(val_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95')} |\n")
        f.write(f"| 精确率 (Precision) | {_fmt(val_results, 'metrics/precision(B)', 'metrics/precision')} |\n")
        f.write(f"| 召回率 (Recall) | {_fmt(val_results, 'metrics/recall(B)', 'metrics/recall')} |\n\n")

        f.write("### 测试集\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("| --- | --- |\n")
        f.write(f"| mAP@0.5 | {_fmt(test_results, 'metrics/mAP50(B)', 'metrics/mAP50')} |\n")
        f.write(f"| mAP@0.5:0.95 | {_fmt(test_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95')} |\n")

    print(f"  [模型归档] README.md → {readme_path}")
    return readme_path


def archive_best_model(
    run_dir: Path,
    models_dir: Path,
    val_results=None,
    test_results=None,
) -> Path:
    """
    将 best.pt 与 README.md 归档到 Models/<时间戳>/ 目录

    目录结构：
        D:\SkinVidCheck\training_output\2026-07-05_18-50-15\weights\best.pt
        → D:\SkinVidCheck\Models\2026-07-05_18-50-15\
            ├── best.pt
            └── README.md

    Args:
        run_dir: 训练输出目录（含 weights/best.pt）
        models_dir: Models 归档根目录
        val_results: ultralytics 验证结果对象（用于 README 性能指标）
        test_results: ultralytics 测试结果对象（用于 README 性能指标）

    Returns:
        归档目录路径（成功），或 None（失败 / 源文件不存在）
    """
    src = run_dir / 'weights' / 'best.pt'
    if not src.exists():
        print(f"  [模型归档] 跳过：未找到 {src}")
        return None

    # 确保 Models 目录存在
    models_dir.mkdir(parents=True, exist_ok=True)

    # 目标目录：Models/<run_dir_name>/
    dst_dir = models_dir / run_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 复制 best.pt
    dst_pt = dst_dir / 'best.pt'
    try:
        shutil.copy2(src, dst_pt)
        size_mb = dst_pt.stat().st_size / (1024 * 1024)
        print(f"  [模型归档] best.pt → {dst_pt} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  [模型归档] 复制 best.pt 失败: {e}")
        return None

    # 生成 README.md
    try:
        _generate_model_readme(dst_dir, run_dir, val_results, test_results)
    except Exception as e:
        print(f"  [模型归档] 生成 README.md 失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"  [模型归档] 完成 → {dst_dir} (含 best.pt + README.md)")
    return dst_dir


# ==================== 主训练流程 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("皮肤病变检测训练系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 生成时间戳和目录结构
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(Config.OUTPUT_DIR) / timestamp
    analysis_dir = run_dir / "analysis"
    
    print(f"\n[目录设置]")
    print(f"  运行目录: {run_dir}")
    print(f"  分析图表: {analysis_dir}")
    
    # 创建目录
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 清理 ultralytics 之前遗留的 runs/ 目录（确保项目根目录干净）
    runs_dir = Path(Config.OUTPUT_DIR).parent / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
        print(f"  已清理旧 runs/: {runs_dir}")

    # 配置中文字体
    setup_chinese_font()
    
    # 检查 CUDA
    if not setup_cuda():
        print("错误：CUDA不可用，将使用CPU训练（速度较慢）")
        Config.DEVICE = "cpu"
    
    # 生成数据配置（放在 run_dir 下）
    data_yaml_path = create_data_yaml(Config, run_dir)
    
    # 加载模型
    print(f"\n[模型加载] 加载: {Config.MODEL_PATH}")
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = YOLO(Config.MODEL_PATH)
        else:
            # 从 Config 动态派生 fallback 模型名（保持版本/任务类型一致）
            # 例如 Config.MODEL_PATH = r"D:\SkinVidCheck\yolo26m-seg.pt"
            #   → fallback = "yolo26m-seg.pt"（仍为 YOLOv26 + 分割，与 Config 对齐）
            fallback_name = os.path.basename(Config.MODEL_PATH)
            print(f"  未找到本地模型 {Config.MODEL_PATH}，将下载预训练权重: {fallback_name}")
            model = YOLO(fallback_name)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print(f"  模型类别数: {len(model.names)} -> 将调整为2类")
    
    # 打印训练配置
    print(f"\n[训练配置]")
    print(f"  图像尺寸: {Config.IMG_SIZE}x{Config.IMG_SIZE} (Letterbox)")
    print(f"  批次大小: {Config.BATCH_SIZE}")
    print(f"  训练轮数: {Config.EPOCHS} (早停耐心值: {Config.PATIENCE})")
    print(f"  优化器: {Config.OPTIMIZER}, 学习率: {Config.LEARNING_RATE}")
    print(f"  设备: {Config.DEVICE}")
    
    # 打印数据增强配置
    print(f"\n[数据增强配置]")
    print(f"  Mosaic:  {Config.MOSAIC} (四图拼接概率)")
    print(f"  MixUp:   {Config.MIXUP} (图像混合概率)")
    print(f"  HSV增强: H={Config.HSV_H}, S={Config.HSV_S}, V={Config.HSV_V}")
    print(f"  几何变换: 旋转={Config.DEGREES}°, 平移={Config.TRANSLATE}, 缩放={Config.SCALE}")
    print(f"  翻转: 上下={Config.FLIPUD}, 左右={Config.FLIPLR}")
    
    # 开始训练
    print(f"\n[开始训练] 按 Ctrl+C 可安全中断...")
    
    try:
        # 训练模型
        results = model.train(
            data=data_yaml_path,
            epochs=Config.EPOCHS,
            imgsz=Config.IMG_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            workers=Config.WORKERS,
            optimizer=Config.OPTIMIZER,
            lr0=Config.LEARNING_RATE,
            patience=Config.PATIENCE,
            
            # 数据增强
            augment=True,
            mosaic=Config.MOSAIC,
            mixup=Config.MIXUP,
            hsv_h=Config.HSV_H,
            hsv_s=Config.HSV_S,
            hsv_v=Config.HSV_V,
            degrees=Config.DEGREES,
            translate=Config.TRANSLATE,
            scale=Config.SCALE,
            flipud=Config.FLIPUD,
            fliplr=Config.FLIPLR,
            
            # 其他设置
            seed=Config.SEED,
            half=Config.HALF_PRECISION,
            cache=Config.CACHE,
            project=Config.OUTPUT_DIR,  # 基础目录
            name=timestamp,             # 使用时间戳作为子目录名
            exist_ok=True,
            verbose=True,
            cos_lr=Config.COS_LR,
            
            # 损失权重
            box=Config.BOX_LOSS_WEIGHT,
            cls=Config.CLS_LOSS_WEIGHT,
            dfl=Config.DFL_LOSS_WEIGHT,
        )
        
        print(f"\n[训练完成]")
        
        # 从 results.csv 读取实际训练轮数
        csv_path = run_dir / "results.csv"
        actual_epochs = Config.EPOCHS
        if csv_path.exists():
            try:
                df_temp = pd.read_csv(csv_path)
                actual_epochs = len(df_temp)
                print(f"  计划训练轮数: {Config.EPOCHS}")
                print(f"  实际完成轮数: {actual_epochs}")
            except:
                print(f"  训练轮数: {Config.EPOCHS}")
        else:
            print(f"  训练轮数: {Config.EPOCHS}")
        
        best_pt_path = run_dir / 'weights' / 'best.pt'
        print(f"  最佳模型已保存至: {best_pt_path}")
        
    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止训练")
        return
    except Exception as e:
        print(f"\n[训练错误] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 验证阶段
    print(f"\n[验证阶段] 在验证集上评估...")
    val_results = None
    try:
        val_results = model.val(
            data=data_yaml_path,
            split='val',
            imgsz=Config.IMG_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            save_json=True,
            plots=True,
            project=str(run_dir),     # 写到 training_output/<时间戳>/val/ 而非 runs/segment/val/
            name='val',
            exist_ok=True,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD
        )
        
        # 修复：使用带 (B) 后缀的键名获取指标
        map50 = get_metric(val_results, 'metrics/mAP50(B)', 'metrics/mAP50', 0.0)
        map50_95 = get_metric(val_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 0.0)
        precision = get_metric(val_results, 'metrics/precision(B)', 'metrics/precision', 0.0)
        recall = get_metric(val_results, 'metrics/recall(B)', 'metrics/recall', 0.0)
        
        print(f"\n  验证集结果:")
        print(f"    mAP@0.5:      {map50:.4f}")
        print(f"    mAP@0.5:0.95: {map50_95:.4f}")
        print(f"    精确率:       {precision:.4f}")
        print(f"    召回率:       {recall:.4f}")
        
    except Exception as e:
        print(f"  验证阶段出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试集评估
    print(f"\n[测试阶段] 在测试集上评估...")
    test_results = None
    try:
        test_results = model.val(
            data=data_yaml_path,
            split='test',
            imgsz=Config.IMG_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            save_json=True,
            plots=True,
            project=str(run_dir),     # 写到 training_output/<时间戳>/test_results/ 而非 runs/segment/test_results/
            name='test_results',
            exist_ok=True,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD
        )
        
        # 修复：使用带 (B) 后缀的键名获取指标
        map50 = get_metric(test_results, 'metrics/mAP50(B)', 'metrics/mAP50', 0.0)
        map50_95 = get_metric(test_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 0.0)
        
        print(f"\n  测试集结果:")
        print(f"    mAP@0.5:      {map50:.4f}")
        print(f"    mAP@0.5:0.95: {map50_95:.4f}")
        
    except Exception as e:
        print(f"  测试阶段出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成中文图表
    try:
        csv_path = run_dir / "results.csv"
        plot_training_results(csv_path, analysis_dir)
    except Exception as e:
        print(f"  图表生成警告: {e}")
        import traceback
        traceback.print_exc()
    
    # 归档所有图表（将 YOLO 生成的图表移动到 analysis 文件夹）
    organize_charts(run_dir, analysis_dir)

    # 归档 best.pt 与 README 到 Models/<时间戳>/ 文件夹（路径可由 Config.MODELS_DIR 自定义）
    models_dir = Path(Config.MODELS_DIR)
    archived_model_path = archive_best_model(run_dir, models_dir, val_results, test_results)

    # 生成最终报告
    try:
        report_path = run_dir / "训练报告.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("皮肤病变检测训练报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {Config.MODEL_PATH}\n")
            f.write(f"数据集: {Config.DATASET_ROOT}\n")
            f.write(f"图像尺寸: {Config.IMG_SIZE}x{Config.IMG_SIZE}\n\n")

            f.write(f"最佳模型归档: {archived_model_path if archived_model_path else '未归档'}\n\n")
            
            f.write("[训练配置]\n")
            f.write(f"训练轮数: {Config.EPOCHS}\n")
            f.write(f"批次大小: {Config.BATCH_SIZE}\n")
            f.write(f"优化器: {Config.OPTIMIZER}\n")
            f.write(f"学习率: {Config.LEARNING_RATE}\n\n")
            
            f.write("[数据增强配置]\n")
            f.write(f"Mosaic: {Config.MOSAIC}, MixUp: {Config.MIXUP}\n")
            f.write(f"HSV: H={Config.HSV_H}, S={Config.HSV_S}, V={Config.HSV_V}\n")
            f.write(f"旋转: {Config.DEGREES}, 平移: {Config.TRANSLATE}, 缩放: {Config.SCALE}\n")
            f.write(f"翻转: 上下{Config.FLIPUD}, 左右{Config.FLIPLR}\n\n")
            
            if val_results:
                f.write("[验证集结果]\n")
                f.write(f"mAP@0.5:      {get_metric(val_results, 'metrics/mAP50(B)', 'metrics/mAP50', 0.0):.4f}\n")
                f.write(f"mAP@0.5:0.95: {get_metric(val_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 0.0):.4f}\n")
                f.write(f"精确率:       {get_metric(val_results, 'metrics/precision(B)', 'metrics/precision', 0.0):.4f}\n")
                f.write(f"召回率:       {get_metric(val_results, 'metrics/recall(B)', 'metrics/recall', 0.0):.4f}\n\n")
            
            if test_results:
                f.write("[测试集结果]\n")
                f.write(f"mAP@0.5:      {get_metric(test_results, 'metrics/mAP50(B)', 'metrics/mAP50', 0.0):.4f}\n")
                f.write(f"mAP@0.5:0.95: {get_metric(test_results, 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 0.0):.4f}\n\n")
            
            f.write("[类别信息]\n")
            f.write("类别0: 湿疹\n")
            f.write("类别1: 痤疮\n")
            f.write(f"\n分析图表位置: {analysis_dir}\n")
            f.write("说明: 所有分析图表（包括混淆矩阵、PR曲线、训练过程等）均已归档到此文件夹\n")
        
        print(f"\n[报告已保存] {report_path}")
    except Exception as e:
        print(f"保存报告失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("训练流程执行完毕！")
    print(f"输出目录: {run_dir}")
    print(f"分析图表: {analysis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()