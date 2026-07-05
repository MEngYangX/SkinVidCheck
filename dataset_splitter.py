#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割工具
将images和labels文件夹按8:1:1比例分割为train/val/test
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将数据集按8:1:1比例分割为训练集、验证集和测试集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python split_dataset.py -i ./data -o ./output
  python split_dataset.py --input ./my_data --output ./split_data --seed 42
        """
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件夹路径（包含images和labels两个子文件夹）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出文件夹路径'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    parser.add_argument(
        '--ratio',
        type=str,
        default='8:1:1',
        help='分割比例，格式为train:val:test（默认8:1:1）'
    )
    parser.add_argument(
        '--ext',
        type=str,
        default=None,
        help='指定处理的图像扩展名（如.jpg/.png），不指定则处理所有文件'
    )
    parser.add_argument(
        '--use-existing-split',
        action='store_true',
        help='使用输出目录中已存在的 val/test 作为固定集，'
             '其余所有图像全部作为训练集（不进行随机分割）。'
             'val/test 文件夹必须已存在且不会被修改。'
    )

    return parser.parse_args()


def parse_ratio(ratio_str: str) -> Tuple[float, float, float]:
    """解析比例字符串"""
    try:
        parts = ratio_str.split(':')
        if len(parts) != 3:
            raise ValueError("比例格式必须为 train:val:test")
        
        train_r = float(parts[0])
        val_r = float(parts[1])
        test_r = float(parts[2])
        
        total = train_r + val_r + test_r
        if total == 0:
            raise ValueError("比例之和不能为0")
            
        # 归一化
        return (train_r/total, val_r/total, test_r/total)
    except Exception as e:
        print(f"错误：比例解析失败 - {e}")
        exit(1)


def get_paired_files(images_dir: Path, labels_dir: Path, ext: str = None) -> List[Tuple[str, Path, Path]]:
    """
    获取配对的图像和标签文件
    返回: [(文件名, 图像路径, 标签路径), ...]
    """
    paired_files = []
    
    # 获取所有图像文件
    if ext:
        image_files = list(images_dir.glob(f'*{ext}'))
    else:
        image_files = [f for f in images_dir.iterdir() if f.is_file()]
    
    print(f"在 {images_dir} 中找到 {len(image_files)} 个文件")
    
    # 查找对应的标签文件
    unmatched = []
    for img_path in image_files:
        base_name = img_path.stem  # 不带扩展名的文件名
        label_path = labels_dir / f"{base_name}.txt"  # 假设标签是.txt格式
        
        if not label_path.exists():
            # 尝试其他可能的标签扩展名
            for label_ext in ['.txt', '.xml', '.json', '.yaml', '.yml', '']:
                alt_label = labels_dir / f"{base_name}{label_ext}"
                if alt_label.exists():
                    label_path = alt_label
                    break
        
        if label_path.exists():
            paired_files.append((base_name, img_path, label_path))
        else:
            unmatched.append(img_path.name)
    
    if unmatched:
        print(f"警告：以下 {len(unmatched)} 个文件没有找到对应标签，将被跳过：")
        for name in unmatched[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(unmatched) > 5:
            print(f"  ... 还有 {len(unmatched)-5} 个")
    
    return paired_files


def create_directory_structure(output_dir: Path) -> dict:
    """创建输出目录结构"""
    splits = ['train', 'val', 'test']
    folders = {}

    for subdir in ['images', 'labels']:
        for split in splits:
            dir_path = output_dir / subdir / split
            dir_path.mkdir(parents=True, exist_ok=True)
            folders[f"{subdir}_{split}"] = dir_path
            print(f"检查目录：{dir_path}")

    return folders


def get_existing_split_base_names(output_dir: Path) -> Tuple[set, set, set]:
    """
    读取已存在的 val/test 文件夹中所有文件的 base name

    Returns:
        (val_names, test_names, excluded): 三个集合
        - val_names: val 中所有 base 名
        - test_names: test 中所有 base 名
        - excluded: val_names ∪ test_names（用于过滤 train）

    Raises:
        SystemExit: 如果 val 或 test 目录不存在
    """
    val_dir = output_dir / 'images' / 'val'
    test_dir = output_dir / 'images' / 'test'

    if not val_dir.exists():
        print(f"错误：--use-existing-split 模式下，{val_dir} 必须存在")
        exit(1)
    if not test_dir.exists():
        print(f"错误：--use-existing-split 模式下，{test_dir} 必须存在")
        exit(1)

    val_names = {p.stem for p in val_dir.iterdir() if p.is_file()}
    test_names = {p.stem for p in test_dir.iterdir() if p.is_file()}
    excluded = val_names | test_names

    return val_names, test_names, excluded


def clear_directory(dir_path: Path):
    """
    清空目录中的所有内容（保留目录本身）
    用于 --use-existing-split 模式下的 train/ 同步
    """
    if not dir_path.exists():
        return
    for item in dir_path.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"  ⚠ 删除 {item} 失败: {e}")


def split_data(files: List, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List, List, List]:
    """
    按比例分割数据集
    返回: (train_files, val_files, test_files)
    """
    random.seed(seed)
    # 创建副本避免修改原列表
    files_copy = files.copy()
    random.shuffle(files_copy)
    
    total = len(files_copy)
    if total == 0:
        return [], [], []
    
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    # 测试集取剩余部分，避免舍入误差
    test_num = total - train_num - val_num
    
    train_files = files_copy[:train_num]
    val_files = files_copy[train_num:train_num + val_num]
    test_files = files_copy[train_num + val_num:]
    
    # 检查余数分配（默认给训练集）
    actual_train = len(train_files)
    actual_val = len(val_files)
    actual_test = len(test_files)
    
    print(f"\n分割结果：")
    print(f"  训练集(train)：{actual_train} ({actual_train/total*100:.1f}%)")
    print(f"  验证集(val)：  {actual_val} ({actual_val/total*100:.1f}%)")
    print(f"  测试集(test)： {actual_test} ({actual_test/total*100:.1f}%)")
    print(f"  总计：{total}")
    
    return train_files, val_files, test_files


def copy_files(file_list: List[Tuple], dest_img_dir: Path, dest_label_dir: Path, split_name: str):
    """复制文件到目标目录"""
    copied = 0
    skipped = 0

    for base_name, img_path, label_path in file_list:
        try:
            # 复制图像
            dest_img = dest_img_dir / img_path.name
            if dest_img.exists():
                skipped += 1
            else:
                shutil.copy2(img_path, dest_img)

            # 复制标签
            dest_label = dest_label_dir / label_path.name
            if dest_label.exists():
                skipped += 1
            else:
                shutil.copy2(label_path, dest_label)
                copied += 1

        except Exception as e:
            print(f"  错误：复制 {base_name} 失败 - {e}")

    # 总是打印（即使 skipped==0），便于用户确认复制结果
    print(f"  {split_name}: 已复制 {copied} 对文件，跳过 {skipped//2} 对已存在")


def main():
    args = parse_args()
    
    # 解析路径
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    
    # 检查输入目录
    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'
    
    if not input_dir.exists():
        print(f"错误：输入目录不存在 - {input_dir}")
        exit(1)
    if not images_dir.exists():
        print(f"错误：images目录不存在 - {images_dir}")
        exit(1)
    if not labels_dir.exists():
        print(f"错误：labels目录不存在 - {labels_dir}")
        exit(1)
    
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")

    # 获取配对文件
    paired_files = get_paired_files(images_dir, labels_dir, args.ext)

    if len(paired_files) == 0:
        print("错误：没有找到配对的图像和标签文件！")
        print("请确保：")
        print("  1. images文件夹中有图像文件")
        print("  2. labels文件夹中有同名（不同扩展名）的标签文件")
        exit(1)

    print(f"\n找到 {len(paired_files)} 对匹配的文件")

    # 创建目录结构
    print("\n创建目录结构...")
    folders = create_directory_structure(output_dir)

    # 分割数据
    split_mode = 'random'  # 标记当前模式，用于 summary
    if args.use_existing_split:
        # ===== 固定分割模式 =====
        print(f"\n[固定分割模式] 使用已存在的 val/test 作为固定集")
        val_names, test_names, excluded = get_existing_split_base_names(output_dir)
        print(f"  val 固定集：{len(val_names)} 张")
        print(f"  test 固定集：{len(test_names)} 张")

        # 过滤 input：从配对文件中排除 val/test 同名项
        train_files = [(n, i, l) for (n, i, l) in paired_files if n not in excluded]
        excluded_in_input = len(paired_files) - len(train_files)
        print(f"  训练集：{len(train_files)} 张（input 排除 val/test base 名后）")
        if excluded_in_input > 0:
            print(f"  ⚠ 已排除 {excluded_in_input} 个 input 文件（与 val/test 同 base 名）")

        # 同步 train/：先清空再填（保证 train/val/test 互不重复）
        print(f"\n  同步 train/ 目录（清空后重新填充）...")
        clear_directory(folders['images_train'])
        clear_directory(folders['labels_train'])

        # 复制 train
        print("\n开始复制文件...")
        if train_files:
            print(f"\n处理训练集...")
            copy_files(train_files, folders['images_train'], folders['labels_train'], "train")
        else:
            print(f"\n  ⚠ 训练集为空，无文件可复制")

        # val/test 保持不动
        print(f"  ✓ val/{len(val_names)} 张保持不动")
        print(f"  ✓ test/{len(test_names)} 张保持不动")

        # 设置 summary 用的变量
        val_files = []    # 不参与复制
        test_files = []   # 不参与复制
        train_r = val_r = test_r = None
        split_mode = 'use-existing-split'
    else:
        # ===== 随机分割模式（默认） =====
        train_r, val_r, test_r = parse_ratio(args.ratio)
        train_files, val_files, test_files = split_data(
            paired_files, train_r, val_r, test_r, args.seed
        )

        # 复制文件
        print("\n开始复制文件...")

        if train_files:
            print(f"\n处理训练集...")
            copy_files(train_files, folders['images_train'], folders['labels_train'], "train")

        if val_files:
            print(f"\n处理验证集...")
            copy_files(val_files, folders['images_val'], folders['labels_val'], "val")

        if test_files:
            print(f"\n处理测试集...")
            copy_files(test_files, folders['images_test'], folders['labels_test'], "test")

    # 生成分割摘要文件
    summary_file = output_dir / 'split_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集分割摘要\n")
        f.write(f"{'='*40}\n")
        f.write(f"原始数据：{input_dir}\n")
        f.write(f"输出位置：{output_dir}\n")
        if split_mode == 'use-existing-split':
            f.write(f"分割模式：使用已存在 val/test（无随机分割）\n")
            f.write(f"  val 固定集：{len(val_names)} 张\n")
            f.write(f"  test 固定集：{len(test_names)} 张\n")
            f.write(f"  训练集：{len(train_files)} 张（input 排除 val/test base 名后）\n")
            f.write(f"\n文件总数（input）：{len(paired_files)}\n")
            f.write(f"  用于 train：{len(train_files)}\n")
            f.write(f"  排除（与 val/test 重复）：{excluded_in_input}\n")
            f.write(f"\n注意：val/test 文件夹未被修改\n")
        else:
            f.write(f"分割比例：{args.ratio} (实际: {train_r:.2f}:{val_r:.2f}:{test_r:.2f})\n")
            f.write(f"随机种子：{args.seed}\n\n")
            f.write(f"文件总数：{len(paired_files)}\n")
            f.write(f"  训练集：{len(train_files)}\n")
            f.write(f"  验证集：{len(val_files)}\n")
            f.write(f"  测试集：{len(test_files)}\n")
    
    print(f"\n{'='*50}")
    if split_mode == 'use-existing-split':
        print("固定分割模式完成：")
        print(f"  ✓ 向 train/ 中添加了 {len(train_files)} 个文件（input 排除 val/test 同名后）")
        print(f"  ✓ val/ 保持原样（{len(val_names)} 张）")
        print(f"  ✓ test/ 保持原样（{len(test_names)} 张）")
    else:
        print("分割完成！")
    print(f"摘要已保存至：{summary_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()