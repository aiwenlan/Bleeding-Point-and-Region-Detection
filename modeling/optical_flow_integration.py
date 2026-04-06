"""
BlooDet光流集成模块
基于PWC-Net实现光流估计和时序运动建模
严格按照论文要求，必须使用真正的PWC-Net模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import os

from PWC_Net.PWCNet import PWCDCNet
PWC_AVAILABLE = True

class OpticalFlowEstimator(nn.Module):
    """光流估计器 - 基于PWC-Net"""
    
    def __init__(self, pwcnet_ckpt: str = None, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # 使用PWC_Net的PWCDCNet
        if not pwcnet_ckpt:
            raise ValueError("pwcnet_ckpt is required. Please provide the path to PWC-Net weights.")
        if not os.path.exists(pwcnet_ckpt):
            raise FileNotFoundError(f"PWC-Net weights not found at {pwcnet_ckpt}")
        
        # 创建PWC-Net模型并加载本地权重
        self.pwc_net = PWCDCNet(strModel='default', ckpt_path=pwcnet_ckpt).cuda().train(False)
        # print(f"Loaded PWC-Net from {pwcnet_ckpt}")
        # print(f"🔧 PWC-Net模型精度: {next(self.pwc_net.parameters()).dtype}")
        
        # 冻结PWC-Net参数
        for param in self.pwc_net.parameters():
            param.requires_grad = False
        
        self.pwc_net.eval()
    
    def preprocess_frames(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基本预处理帧 - 只做分离和数据类型处理
        frames: [N, T, 3, H, W] - 输入视频帧序列
        返回: (frame1, frame2) 两张分离的图片
        """
        if frames.dim() != 5:
            raise ValueError(f"Expected input shape [N, T, 3, H, W], got {frames.shape}")
        
        N, T, C, H, W = frames.shape
        if T != 2:
            raise ValueError(f"PWC-Net expects exactly 2 frames, got {T}")
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        
        # 分离两帧
        frame1 = frames[:, 0]  # [N, 3, H, W]
        frame2 = frames[:, 1]  # [N, 3, H, W]
        
        # 基本预处理（与run.py一致）：
        # 1. RGB -> BGR 转换 (run.py期望BGR格式)
        # 数据加载时已经转换为RGB，这里需要转回BGR给PWC-Net
        frame1 = frame1[:, [2, 1, 0], :, :]  # RGB -> BGR
        frame2 = frame2[:, [2, 1, 0], :, :]  # RGB -> BGR
        
        # 2. 确保数据是连续的 (ascontiguousarray)
        frame1 = frame1.contiguous()
        frame2 = frame2.contiguous()
        
        # 3. 确保数据类型为float32
        frame1 = frame1.float()
        frame2 = frame2.float()
        
        # 4. 检查值范围，避免重复归一化
        # 数据加载时已经做了 /255.0 归一化，这里只需要检查范围
        if frame1.max() > 1.0:
            # 如果数据范围超过1，说明还没有归一化，需要除以255
            frame1 = frame1 / 255.0
            frame2 = frame2 / 255.0
            # print(f"🔧 归一化帧数据到[0,1]范围: frame1=[{frame1.min():.6f}, {frame1.max():.6f}], frame2=[{frame2.min():.6f}, {frame2.max():.6f}]")
        # else:
        #     # 数据已经在[0,1]范围内，不需要再次归一化
        #     print(f"🔧 帧数据已在[0,1]范围内: frame1=[{frame1.min():.6f}, {frame1.max():.6f}], frame2=[{frame2.min():.6f}, {frame2.max():.6f}]")
        
        # 注意：HWC->CHW转换已在数据加载时完成（输入是[N,T,3,H,W]）
        
        # 注意：64对齐和插值调整在forward方法中处理，避免重复
        
        return frame1, frame2
    
    def postprocess_flow(self, flow: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        后处理光流以匹配目标尺寸
        flow: [N, 2, H, W]
        target_size: (H, W)
        """
        if flow.shape[-2:] != target_size:
            # 缩放光流
            scale_h = target_size[0] / flow.shape[-2]
            scale_w = target_size[1] / flow.shape[-1]
            
            flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=False)
            flow[:, 0] *= scale_w  # x方向缩放
            flow[:, 1] *= scale_h  # y方向缩放
        
        return flow
    
    def forward(self, frames: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        估计光流
        frames: [N, T, 3, H, W] - 输入视频帧序列，T必须为2
        返回: [N, 2, H, W] 光流场
        """
        original_size = frames.shape[-2:]
        if target_size is None:
            target_size = original_size
        
        # 预处理 - 分离两帧，转为RGB格式
        frame1, frame2 = self.preprocess_frames(frames)
        # print(f"🔍 分离两帧后frame1数据值：")
        # print(frame1.min(), frame1.max())
        # print(frame1)
        # print(f"🔍 分离两帧后frame2数据值：")
        # print(frame2.min(), frame2.max())
        # print(frame2)
        # 按照run.py的完整预处理流程
        # 1. 断言检查（与run.py一致）
        assert(frame1.shape[1] == frame2.shape[1])  # 通道数相同
        assert(frame1.shape[2] == frame2.shape[2])  # 高度相同
        assert(frame1.shape[3] == frame2.shape[3])  # 宽度相同
        
        intWidth = frame1.shape[3]
        intHeight = frame1.shape[2]
        
        # print(f"🔍 PWC-Net 输入尺寸: {intHeight}x{intWidth}")
        
        # 2. 强制调整到run.py的固定尺寸 1024x436
        target_width = 512
        target_height = 512
        # print(f"🔍 调整到run.py固定尺寸: {target_height}x{target_width}")
        
        if (intHeight, intWidth) != (target_height, target_width):
            frame1 = F.interpolate(frame1, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
            frame2 = F.interpolate(frame2, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
            # print(f"✅ 调整到run.py固定尺寸: {frame1.shape}")
        # print(f"🔍 frame1预处理后数据值：")
        # print(frame1.min(), frame1.max())
        # print(frame1)
        # print(f"🔍 frame2预处理后数据值：")
        # print(frame2.min(), frame2.max())
        # print(frame2)
        # 更新尺寸变量
        intWidth = target_width
        intHeight = target_height
        
        # 3. 移动到CUDA并调整维度（与run.py一致）
        tenPreprocessedOne = frame1.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedTwo = frame2.cuda().view(1, 3, intHeight, intWidth)
        
        # 4. 计算64对齐的尺寸（与run.py一致）
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        # print(f"🔍 PWC-Net 预处理尺寸: {intPreprocessedHeight}x{intPreprocessedWidth} (64对齐)")
        
        # 5. 调整到64对齐尺寸（与run.py一致）
        tenPreprocessedOne = torch.nn.functional.interpolate(
            input=tenPreprocessedOne, 
            size=(intPreprocessedHeight, intPreprocessedWidth), 
            mode='bilinear', 
            align_corners=False
        )
        tenPreprocessedTwo = torch.nn.functional.interpolate(
            input=tenPreprocessedTwo, 
            size=(intPreprocessedHeight, intPreprocessedWidth), 
            mode='bilinear', 
            align_corners=False
        )
        # print(f"✅ 调整到64对齐尺寸: {tenPreprocessedOne.shape}")
        
        # 更新变量名以匹配run.py
        frame1 = tenPreprocessedOne
        frame2 = tenPreprocessedTwo
        # print(f"🔍 frame1输入前数据值：")
        # print(frame1.dtype)
        # print(frame1.min(), frame1.max())
        # print('不等于0：', (frame1!=0).sum())
        # print('不等于1：', (frame1!=1).sum())
        # print('不等于255：', (frame1!=255).sum())
        # print(f"🔍 frame2输入前数据值：")
        # print(frame2.dtype)
        # print(frame2.min(), frame2.max())
        # print(frame2)
        # 光流估计 (使用PWC_Net) - 禁用混合精度
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度，确保float32
                flow = self.pwc_net(frame1, frame2)
            if isinstance(flow, tuple):  # 训练模式返回多尺度流
                flow = flow[0]  # 取最高分辨率
            # print(f"🔍 PWC-Net 光流估计成功: {flow.min()}, {flow.max()}")
            # print(f"✅ PWC-Net 光流估计成功: {flow.shape}")
            
            # 调整回原始尺寸
            if (intHeight, intWidth) != (intPreprocessedHeight, intPreprocessedWidth):
                flow = torch.nn.functional.interpolate(
                    input=flow, 
                    size=(intHeight, intWidth), 
                    mode='bilinear', 
                    align_corners=False
                )
                # 调整光流缩放因子
                flow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
                flow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
                # print(f"✅ 调整回原始尺寸: {flow.shape}")
        
        # 后处理
        flow = self.postprocess_flow(flow, target_size)
        
        return flow




