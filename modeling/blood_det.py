"""
BlooDet
集成所有增强模块:SAM2编码器、增强点分支、增强掩码分支、跨分支引导
"""

import torch
import torch.nn as nn
from .sam2_wrapper import SAM2Backbone
from .point_branch import PointBranch
from .mask_branch import MaskBranch
# 跨分支引导通过参数传递实现，不需要单独的CrossBranchGuidance类

class BlooDet(nn.Module):
    """
    BlooDet - 完整的出血检测模型
    
    架构组件:
    1. SAM2图像编码器 - 多尺度特征提取
    2. 点分支 - 多任务点定位
    3. 掩码分支 - 边缘引导区域分割
    4. 跨分支引导 - 双向信息交换
    5. 记忆机制 - 时序信息建模
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. SAM2图像编码器 (用于特征提取)
        self.sam2_img_encoder = SAM2Backbone(
            sam2_config=getattr(cfg.model, 'sam2_config', 'sam2_hiera_b+.yaml'),
            ckpt_path=getattr(cfg.model, 'sam2_ckpt', None)
        )
        
        # 2. 记忆库系统 - 由各分支内部管理，无需在主模型中实例化
        
        # 3. 出血点分支 (增强版)
        self.point_branch = PointBranch(cfg)
        
        # 4. 出血区域分支
        self.mask_branch = MaskBranch(cfg)
        
        # 5. 设置跨分支Memory Bank引用
        # Point 分支可以获取 Mask 记忆
        self.point_branch.mask_memory_bank = self.mask_branch.mask_memory_bank
        # Mask 分支可以获取 Point 记忆（point_map）
        self.mask_branch.point_memory_bank = self.point_branch.point_memory_bank
        
        # 模型状态
        self._is_initialized = True
    
    def forward(self, frames_seq):
        """
        前向传播
        frames_seq: [N, T=8, 3, H, W] 输入帧序列
        """
        if not self._is_initialized:
            raise RuntimeError("模型未正确初始化")
        
        # 验证输入
        if frames_seq.dim() != 5:
            raise ValueError(f"输入维度错误，期望5D [N, T, C, H, W]，得到{frames_seq.dim()}D")
        
        # 确保 frames_seq 为 float32
        frames_seq = frames_seq.float()
        
        # Step 1: SAM2 特征提取
        # 返回 List[Dict[str, Tensor]]，每一帧一个多尺度特征字典
        feats_seq = self.sam2_img_encoder(frames_seq)
        
        # ⭐ 方案一：完全串行执行两个分支（不使用 CUDA Stream）
        # 先执行 Mask 分支（会从 point_memory_bank 里取 point_map，如果有的话）
        mask_outputs = self.mask_branch(
            feats_seq,
            point_map=None,      # 不显式传 point_map，内部会从 point_memory_bank 获取或回退到 edge
            prev_mask_feats=None # 由 MaskBranch 内部从 mask memory bank 获取
        )
        
        # 再执行 Point 分支（会从 mask_memory_bank 里取历史 mask 相关特征）
        point_outputs = self.point_branch(
            feats_seq,
            prev_mask_maps=None,  # 由 PointBranch 内部从 mask memory bank 获取
            prev_mask_feats=None, # 由 PointBranch 内部从 mask memory bank 获取
            frames_seq=frames_seq
        )
        
        # Step 4: 跨分支引导通过 Memory Bank 共享实现
        # Point → Mask: point_map 通过 point_memory_bank 共享（mask_branch 从 point_memory_bank 获取）
        # Mask → Point: prev_mask_maps, prev_mask_feats 通过 mask_memory_bank 共享（point_branch 从 mask_memory_bank 获取）
        # （这些逻辑都在各自分支内部完成，这里不用再额外处理）
        
        # Step 5: 更新记忆库 - 由各分支内部管理（在各自 forward 中完成）
        
        # Step 6: 构建最终输出
        output = self._build_output(point_outputs, mask_outputs)
        
        return output
    
    def _build_output(self, point_outputs, mask_outputs):
        """构建模型输出"""
        # 验证必需输出
        if not isinstance(point_outputs, dict):
            raise ValueError("point_outputs必须是字典类型")
        
        required_keys = ['coords', 'score']
        for key in required_keys:
            if key not in point_outputs:
                raise ValueError(f"point_outputs缺少必需的键: {key}")
        
        # 基础输出（点分支）
        output = {
            "point": point_outputs['coords'],              # [N, 2] 点坐标
            "point_score": point_outputs['score'],         # [N, 1] 存在性分数
            "point_map": point_outputs.get('point_map'),   # [N, 1, H, W] 点图（若有）
        }
        
        # 掩码分支输出 (如果可用)
        if mask_outputs is not None:
            output.update({
                "mask": mask_outputs['mask_map'],                 # [N, 1, H, W] 掩码
                "edge_features": mask_outputs.get('edge_features'),
                "sam_mask": mask_outputs.get('sam_mask'),
            })
        
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BlooDet',
            'architecture': 'SAM2-based dual-branch',
            'training_stage': 'full',
            'is_initialized': self._is_initialized,
            'parameters': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'frozen_parameters': total_params - trainable_params,
            },
            'components': {
                'sam2_img_encoder': 'SAM2Backbone (shared image encoder)',
                'point_branch': 'PointBranch (with internal SAM2 decoder, memory bank & optical flow)',
                'mask_branch': 'MaskBranch (with internal SAM2 decoder, memory bank & edge generator)',
            },
        }
