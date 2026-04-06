"""
BlooDet Data Transforms
数据增强和预处理变换
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import random


class BlooDet_Transform:
    """BlooDet数据变换类 - 支持视频序列和标注的同步变换"""
    
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.img_size = cfg.data.img_size
        
        # 数据增强配置
        aug_cfg = getattr(cfg, 'augmentation', None)
        self.enabled = getattr(aug_cfg, 'enabled', True) if aug_cfg else True
        self.enabled = self.enabled and is_train
        
        if self.enabled:
            self.h_flip_prob = getattr(aug_cfg, 'horizontal_flip', 0.5) if aug_cfg else 0.5
            self.v_flip_prob = getattr(aug_cfg, 'vertical_flip', 0.0) if aug_cfg else 0.0
            self.rotation_range = getattr(aug_cfg, 'rotation', 15) if aug_cfg else 15
            self.brightness_range = getattr(aug_cfg, 'brightness', 0.1) if aug_cfg else 0.1
            self.contrast_range = getattr(aug_cfg, 'contrast', 0.1) if aug_cfg else 0.1
            self.saturation_range = getattr(aug_cfg, 'saturation', 0.1) if aug_cfg else 0.1
            self.hue_range = getattr(aug_cfg, 'hue', 0.05) if aug_cfg else 0.05
            
            # 时序数据增强
            temporal_cfg = getattr(aug_cfg, 'temporal_augmentation', None) if aug_cfg else None
            self.temporal_enabled = getattr(temporal_cfg, 'enabled', True) if temporal_cfg else True
            self.frame_dropout_prob = getattr(temporal_cfg, 'frame_dropout', 0.1) if temporal_cfg else 0.1
            self.temporal_shift_range = getattr(temporal_cfg, 'temporal_shift', 2) if temporal_cfg else 2
    
    def __call__(self, frames_t, gt_mask, gt_point, gt_exists):
        """
        应用数据变换
        
        Args:
            frames_t: [T, 3, H, W] 视频帧序列
            gt_mask: [1, H, W] 掩码标注
            gt_point: [2] 点坐标
            gt_exists: [1] 存在性标记
        
        Returns:
            变换后的数据
        """
        if not self.enabled:
            return frames_t, gt_mask, gt_point, gt_exists
        
        # 1. 几何变换（需要同时变换图像和标注）
        frames_t, gt_mask, gt_point = self._apply_geometric_transforms(
            frames_t, gt_mask, gt_point, gt_exists
        )
        
        # 2. 颜色变换（只影响图像）
        frames_t = self._apply_color_transforms(frames_t)
        
        # 3. 时序变换（影响帧序列）
        if self.temporal_enabled:
            frames_t = self._apply_temporal_transforms(frames_t)
        
        return frames_t, gt_mask, gt_point, gt_exists
    
    def _apply_geometric_transforms(self, frames_t, gt_mask, gt_point, gt_exists):
        """应用几何变换"""
        T, C, H, W = frames_t.shape
        
        # 随机水平翻转
        if random.random() < self.h_flip_prob:
            frames_t = torch.flip(frames_t, dims=[3])  # 水平翻转
            gt_mask = torch.flip(gt_mask, dims=[2])    # 水平翻转掩码
            
            # 翻转点坐标
            if gt_exists.item() > 0.5:
                gt_point[0] = W - gt_point[0]  # x坐标翻转
        
        # 随机垂直翻转
        if random.random() < self.v_flip_prob:
            frames_t = torch.flip(frames_t, dims=[2])  # 垂直翻转
            gt_mask = torch.flip(gt_mask, dims=[1])    # 垂直翻转掩码
            
            # 翻转点坐标
            if gt_exists.item() > 0.5:
                gt_point[1] = H - gt_point[1]  # y坐标翻转
        
        # 随机旋转
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            frames_t, gt_mask, gt_point = self._rotate_data(
                frames_t, gt_mask, gt_point, gt_exists, angle
            )
        
        return frames_t, gt_mask, gt_point
    
    def _rotate_data(self, frames_t, gt_mask, gt_point, gt_exists, angle):
        """旋转数据"""
        T, C, H, W = frames_t.shape
        
        # 旋转图像序列
        rotated_frames = []
        for t in range(T):
            frame = frames_t[t].permute(1, 2, 0).numpy()  # [H, W, C]
            
            # 旋转矩阵
            center = (W // 2, H // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 旋转图像
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (W, H))
            rotated_frames.append(torch.from_numpy(rotated_frame).permute(2, 0, 1))
        
        frames_t = torch.stack(rotated_frames, dim=0)
        
        # 旋转掩码
        mask_np = gt_mask[0].numpy()
        rotated_mask = cv2.warpAffine(mask_np, rotation_matrix, (W, H))
        gt_mask = torch.from_numpy(rotated_mask).unsqueeze(0)
        
        # 旋转点坐标
        if gt_exists.item() > 0.5:
            point_homogeneous = np.array([gt_point[0].item(), gt_point[1].item(), 1.0])
            rotated_point = rotation_matrix @ point_homogeneous
            gt_point = torch.tensor([rotated_point[0], rotated_point[1]], dtype=torch.float32)
            
            # 确保坐标在有效范围内
            gt_point[0] = torch.clamp(gt_point[0], 0, W-1)
            gt_point[1] = torch.clamp(gt_point[1], 0, H-1)
        
        return frames_t, gt_mask, gt_point
    
    def _apply_color_transforms(self, frames_t):
        """应用颜色变换"""
        T, C, H, W = frames_t.shape
        
        # 随机亮度调整
        if self.brightness_range > 0:
            brightness_factor = random.uniform(1-self.brightness_range, 1+self.brightness_range)
            frames_t = torch.clamp(frames_t * brightness_factor, 0, 1)
        
        # 随机对比度调整
        if self.contrast_range > 0:
            contrast_factor = random.uniform(1-self.contrast_range, 1+self.contrast_range)
            mean = frames_t.mean(dim=(2, 3), keepdim=True)
            frames_t = torch.clamp((frames_t - mean) * contrast_factor + mean, 0, 1)
        
        # 随机饱和度调整（转换到HSV空间）
        if self.saturation_range > 0 or self.hue_range > 0:
            frames_t = self._adjust_hsv(frames_t)
        
        return frames_t
    
    def _adjust_hsv(self, frames_t):
        """HSV空间调整"""
        T, C, H, W = frames_t.shape
        
        adjusted_frames = []
        for t in range(T):
            frame = frames_t[t].permute(1, 2, 0).numpy()  # [H, W, C]
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # 转换到HSV
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 调整饱和度
            if self.saturation_range > 0:
                sat_factor = random.uniform(1-self.saturation_range, 1+self.saturation_range)
                hsv[:, :, 1] *= sat_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # 调整色调
            if self.hue_range > 0:
                hue_shift = random.uniform(-self.hue_range, self.hue_range) * 180
                hsv[:, :, 0] += hue_shift
                hsv[:, :, 0] = hsv[:, :, 0] % 180
            
            # 转换回RGB
            frame_adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            frame_tensor = torch.from_numpy(frame_adjusted / 255.0).permute(2, 0, 1)
            adjusted_frames.append(frame_tensor)
        
        return torch.stack(adjusted_frames, dim=0)
    
    def _apply_temporal_transforms(self, frames_t):
        """应用时序变换"""
        T, C, H, W = frames_t.shape
        
        # 随机帧丢弃（模拟视频丢帧）
        if self.frame_dropout_prob > 0:
            for t in range(1, T-1):  # 不丢弃第一帧和最后一帧
                if random.random() < self.frame_dropout_prob:
                    # 用前一帧替换
                    frames_t[t] = frames_t[t-1].clone()
        
        # 随机时序偏移（模拟时序不同步）
        if self.temporal_shift_range > 0:
            shift = random.randint(-self.temporal_shift_range, self.temporal_shift_range)
            if shift != 0:
                frames_t = torch.roll(frames_t, shift, dims=0)
        
        return frames_t


class TestTransform:
    """测试时的数据变换（仅归一化和resize）"""
    
    def __init__(self, cfg):
        self.img_size = cfg.data.img_size
    
    def __call__(self, frames_t, gt_mask, gt_point, gt_exists):
        # 测试时不进行数据增强，只确保尺寸正确
        return frames_t, gt_mask, gt_point, gt_exists


class MedicalAugmentation:
    """医学图像特定的数据增强"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        aug_cfg = getattr(cfg, 'augmentation', None)
        medical_cfg = getattr(aug_cfg, 'medical_specific', None) if aug_cfg else None
        
        self.lighting_variation = getattr(medical_cfg, 'lighting_variation', True) if medical_cfg else True
        self.motion_blur = getattr(medical_cfg, 'motion_blur', True) if medical_cfg else True
        self.occlusion_simulation = getattr(medical_cfg, 'occlusion_simulation', True) if medical_cfg else True
    
    def __call__(self, frames_t):
        """应用医学图像特定增强"""
        aug_cfg = getattr(self.cfg, 'augmentation', None)
        if not aug_cfg or not hasattr(aug_cfg, 'medical_specific'):
            return frames_t
        
        T, C, H, W = frames_t.shape
        
        # 内窥镜光照变化模拟
        if self.lighting_variation and random.random() < 0.3:
            frames_t = self._simulate_endoscopic_lighting(frames_t)
        
        # 运动模糊模拟
        if self.motion_blur and random.random() < 0.2:
            frames_t = self._simulate_motion_blur(frames_t)
        
        # 组织遮挡模拟
        if self.occlusion_simulation and random.random() < 0.15:
            frames_t = self._simulate_tissue_occlusion(frames_t)
        
        return frames_t
    
    def _simulate_endoscopic_lighting(self, frames_t):
        """模拟内窥镜光照变化"""
        T, C, H, W = frames_t.shape
        
        # 创建径向光照梯度
        center_x, center_y = W // 2, H // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        # 计算到中心的距离
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2))
        
        # 创建光照掩码（中心亮，边缘暗）
        lighting_mask = 1.0 - 0.3 * (dist / max_dist)
        lighting_mask = lighting_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 应用光照变化
        frames_t = frames_t * lighting_mask
        
        return torch.clamp(frames_t, 0, 1)
    
    def _simulate_motion_blur(self, frames_t):
        """模拟运动模糊"""
        T, C, H, W = frames_t.shape
        
        # 随机选择几帧进行模糊
        blur_frames = random.sample(range(T), min(2, T))
        
        for t in blur_frames:
            frame = frames_t[t]
            
            # 随机运动方向和强度
            angle = random.uniform(0, 360)
            length = random.randint(3, 8)
            
            # 创建运动模糊核
            kernel = self._create_motion_blur_kernel(angle, length)
            
            # 应用模糊（需要转换为numpy进行卷积）
            frame_np = frame.permute(1, 2, 0).numpy()
            blurred_np = cv2.filter2D(frame_np, -1, kernel)
            frames_t[t] = torch.from_numpy(blurred_np).permute(2, 0, 1)
        
        return frames_t
    
    def _create_motion_blur_kernel(self, angle, length):
        """创建运动模糊核"""
        kernel = np.zeros((length, length))
        
        # 计算运动方向
        cx, cy = length // 2, length // 2
        dx = int(length * np.cos(np.radians(angle)) / 2)
        dy = int(length * np.sin(np.radians(angle)) / 2)
        
        # 绘制运动轨迹
        cv2.line(kernel, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, 1)
        
        # 归一化
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _simulate_tissue_occlusion(self, frames_t):
        """模拟组织遮挡"""
        T, C, H, W = frames_t.shape
        
        # 随机创建遮挡区域
        num_occlusions = random.randint(1, 3)
        
        for _ in range(num_occlusions):
            # 随机遮挡位置和大小
            x1 = random.randint(0, W // 2)
            y1 = random.randint(0, H // 2)
            x2 = random.randint(x1 + 20, min(x1 + W // 3, W))
            y2 = random.randint(y1 + 20, min(y1 + H // 3, H))
            
            # 创建椭圆形遮挡
            mask = torch.zeros(H, W)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            
            # 使用opencv创建椭圆掩码
            mask_np = mask.numpy().astype(np.uint8)
            cv2.ellipse(mask_np, center, axes, 0, 0, 360, 1, -1)
            mask = torch.from_numpy(mask_np).float()
            
            # 应用遮挡（变暗）
            occlusion_factor = random.uniform(0.3, 0.7)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            frames_t = frames_t * (1 - mask * (1 - occlusion_factor))
        
        return frames_t


def get_transform(cfg, is_train=True):
    """获取数据变换函数"""
    if is_train:
        transform = BlooDet_Transform(cfg, is_train=True)
        
        # 如果启用医学特定增强
        aug_cfg = getattr(cfg, 'augmentation', None)
        medical_cfg = getattr(aug_cfg, 'medical_specific', None) if aug_cfg else None
        if medical_cfg and getattr(medical_cfg, 'enabled', False):
            medical_aug = MedicalAugmentation(cfg)
            
            def combined_transform(frames_t, gt_mask, gt_point, gt_exists):
                # 先应用基础变换
                frames_t, gt_mask, gt_point, gt_exists = transform(
                    frames_t, gt_mask, gt_point, gt_exists
                )
                # 再应用医学特定增强
                frames_t = medical_aug(frames_t)
                return frames_t, gt_mask, gt_point, gt_exists
            
            return combined_transform
        else:
            return transform
    else:
        return TestTransform(cfg)
