#!/usr/bin/env python3
"""
SurgBlood数据集预处理脚本
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import random

def create_data_structure(root_dir):
    """创建数据目录结构"""
    root_path = Path(root_dir)
    
    dirs = ['frames', 'annotations', 'splits']
    for dir_name in dirs:
        (root_path / dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {root_path / dir_name}")

def create_sample_annotations(output_dir, num_samples=100):
    """创建示例标注文件"""
    annotations_dir = output_dir / 'annotations'
    
    for i in range(num_samples):
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        if random.random() > 0.5:  # 50%概率有出血
            center_x = random.randint(100, 412)
            center_y = random.randint(100, 412)
            radius = random.randint(20, 80)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            mask_path = annotations_dir / f"sample_{i:03d}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            
            point_data = {
                "exists": 1,
                "x": center_x,
                "y": center_y,
                "image_size": [512, 512]
            }
        else:
            point_data = {
                "exists": 0,
                "x": 0,
                "y": 0,
                "image_size": [512, 512]
            }
        
        point_path = annotations_dir / f"sample_{i:03d}_point.json"
        with open(point_path, 'w') as f:
            json.dump(point_data, f, indent=2)

def create_data_splits(output_dir, train_ratio=0.8, val_ratio=0.1):
    """创建训练/验证/测试集分割"""
    splits_dir = output_dir / 'splits'
    
    annotations_dir = output_dir / 'annotations'
    samples = []
    
    for mask_file in annotations_dir.glob("*_mask.png"):
        sample_name = mask_file.stem.replace('_mask', '')
        samples.append(sample_name)
    
    random.shuffle(samples)
    
    total_samples = len(samples)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    def save_split(samples, filename):
        with open(splits_dir / filename, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
    
    save_split(train_samples, 'train.txt')
    save_split(val_samples, 'val.txt')
    save_split(test_samples, 'test.txt')
    
    print(f"Data splits created:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

def create_dummy_frames(output_dir, num_samples=100):
    """创建示例帧图像"""
    frames_dir = output_dir / 'frames'
    
    for i in range(num_samples):
        frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        if random.random() > 0.3:
            x1 = random.randint(50, 400)
            y1 = random.randint(50, 400)
            x2 = random.randint(x1 + 50, 450)
            y2 = random.randint(y1 + 50, 450)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        frame_path = frames_dir / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)

def main():
    parser = argparse.ArgumentParser(description='Prepare SurgBlood dataset')
    parser.add_argument('--output', type=str, default='./SurgBlood', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to create')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("Creating SurgBlood dataset structure...")
    create_data_structure(output_path)
    
    print("Creating dummy frames...")
    create_dummy_frames(output_path, args.num_samples)
    
    print("Creating sample annotations...")
    create_sample_annotations(output_path, args.num_samples)
    
    print("Creating data splits...")
    create_data_splits(output_path)
    
    print(f"\nDataset preparation completed!")
    print(f"Output directory: {output_path}")
    print(f"Total samples: {args.num_samples}")

if __name__ == "__main__":
    main()
