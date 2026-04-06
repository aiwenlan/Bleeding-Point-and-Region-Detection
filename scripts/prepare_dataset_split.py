#!/usr/bin/env python3
"""
BlooDet 数据集分割脚本
按照论文实现细节要求：训练集75段视频，测试集20段视频

使用方法:
python scripts/prepare_dataset_split.py --data_root ./SurgBlood
"""

import os
import sys
import argparse
import random
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='准备BlooDet数据集分割')
    parser.add_argument('--data_root', type=str, default='./SurgBlood', help='数据集根目录')
    parser.add_argument('--train_videos', type=int, default=75, help='训练集视频数量')
    parser.add_argument('--test_videos', type=int, default=20, help='测试集视频数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def create_dataset_split(data_root, train_videos=75, test_videos=20, seed=42):
    """
    创建数据集分割
    
    Args:
        data_root: 数据集根目录
        train_videos: 训练集视频数量
        test_videos: 测试集视频数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    data_root = Path(data_root)
    frames_dir = data_root / 'frames'
    splits_dir = data_root / 'splits'
    
    # 创建分割目录
    splits_dir.mkdir(exist_ok=True)
    
    # 获取所有视频目录
    if not frames_dir.exists():
        print(f"❌ 帧目录不存在: {frames_dir}")
        return False
    
    all_videos = [d.name for d in frames_dir.iterdir() if d.is_dir()]
    
    if len(all_videos) < train_videos + test_videos:
        print(f"❌ 视频数量不足: 需要{train_videos + test_videos}个，实际{len(all_videos)}个")
        return False
    
    print(f"📊 数据集信息:")
    print(f"   - 总视频数: {len(all_videos)}")
    print(f"   - 训练集需求: {train_videos}个视频")
    print(f"   - 测试集需求: {test_videos}个视频")
    print(f"   - 随机种子: {seed}")
    
    # 随机打乱视频列表
    random.shuffle(all_videos)
    
    # 分割数据集
    train_videos_list = all_videos[:train_videos]
    test_videos_list = all_videos[train_videos:train_videos + test_videos]
    
    # 验证集从训练集中选择一部分
    val_size = max(5, train_videos // 10)  # 至少5个，或训练集的10%
    val_videos_list = train_videos_list[-val_size:]
    train_videos_list = train_videos_list[:-val_size]
    
    print(f"\n📂 分割结果:")
    print(f"   - 训练集: {len(train_videos_list)}个视频")
    print(f"   - 验证集: {len(val_videos_list)}个视频")
    print(f"   - 测试集: {len(test_videos_list)}个视频")
    
    # 保存分割文件
    splits = {
        'train': train_videos_list,
        'val': val_videos_list,
        'test': test_videos_list
    }
    
    for split_name, video_list in splits.items():
        split_file = splits_dir / f'{split_name}.txt'
        
        with open(split_file, 'w') as f:
            for video in video_list:
                f.write(f"{video}\n")
        
        print(f"✅ {split_name}.txt 已保存: {len(video_list)}个视频")
    
    # 生成统计报告
    report_file = splits_dir / 'split_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BlooDet 数据集分割报告\n")
        f.write("="*50 + "\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"数据根目录: {data_root}\n")
        f.write("\n")
        f.write("分割详情:\n")
        f.write(f"  训练集: {len(train_videos_list)}个视频\n")
        f.write(f"  验证集: {len(val_videos_list)}个视频\n") 
        f.write(f"  测试集: {len(test_videos_list)}个视频\n")
        f.write(f"  总计: {len(train_videos_list) + len(val_videos_list) + len(test_videos_list)}个视频\n")
        f.write("\n")
        
        # 详细列表
        for split_name, video_list in splits.items():
            f.write(f"{split_name.upper()}集视频列表:\n")
            for i, video in enumerate(video_list, 1):
                f.write(f"  {i:2d}. {video}\n")
            f.write("\n")
    
    print(f"📋 分割报告已保存: {report_file}")
    
    # 验证分割文件
    print(f"\n🔍 验证分割文件...")
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                print(f"   - {split_name}.txt: {len(lines)}行")
        else:
            print(f"   - ❌ {split_name}.txt 不存在")
    
    return True

def validate_dataset_structure(data_root):
    """验证数据集结构"""
    data_root = Path(data_root)
    
    required_dirs = ['frames', 'masks', 'points']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (data_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️ 缺少必需目录: {missing_dirs}")
        return False
    
    # 检查视频数量一致性
    frames_videos = set(d.name for d in (data_root / 'frames').iterdir() if d.is_dir())
    masks_videos = set(d.name for d in (data_root / 'masks').iterdir() if d.is_dir())
    points_videos = set(d.name for d in (data_root / 'points').iterdir() if d.is_dir())
    
    if not (frames_videos == masks_videos == points_videos):
        print(f"⚠️ 视频目录不一致:")
        print(f"   - frames: {len(frames_videos)}个")
        print(f"   - masks: {len(masks_videos)}个") 
        print(f"   - points: {len(points_videos)}个")
        return False
    
    print(f"✅ 数据集结构验证通过，共{len(frames_videos)}个视频")
    return True

def main():
    args = parse_args()
    
    print("🩸 BlooDet 数据集分割工具")
    print("="*50)
    
    # 验证数据集结构
    if not validate_dataset_structure(args.data_root):
        print("❌ 数据集结构验证失败")
        return 1
    
    # 创建数据集分割
    success = create_dataset_split(
        args.data_root, 
        args.train_videos, 
        args.test_videos,
        args.seed
    )
    
    if success:
        print("\n✅ 数据集分割完成!")
        print(f"📁 分割文件保存在: {Path(args.data_root) / 'splits'}")
        
        # 显示使用提示
        print(f"\n🚀 使用提示:")
        print(f"   训练: python train.py --config configs/default.yaml")
        print(f"   评估: python evaluate.py --checkpoint checkpoints/best.pth --split test")
        
    else:
        print("❌ 数据集分割失败")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
