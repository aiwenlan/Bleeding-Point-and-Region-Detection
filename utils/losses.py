"""
BlooDet 损失函数
基于论文公式9的标准多任务损失函数，全模型端到端训练

损失函数组成 (论文公式9):
L = λ_m * L_mask + λ_e * L_edge + λ_s * L_score + λ_p * L_point

使用固定权重，不进行多阶段训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List

class BlooDet_Loss(nn.Module):
    """
    BlooDet 标准损失函数 - 全模型端到端训练
    
    包含 (论文公式9):
    1. Mask Loss (掩码损失) - Focal Loss + Dice Loss
    2. Edge Loss (边缘损失) - Focal Loss + Dice Loss  
    3. Point Loss (出血点损失) - Smooth L1 Loss with indicator function
    4. Score Loss (置信度损失) - Binary Cross-Entropy Loss
    
    使用固定权重，不进行多阶段训练
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 固定损失权重 (λ_m=1, λ_e=1, λ_s=1, λ_p=0.5)
        self.lambda_mask = cfg.loss.lambda_mask    # 1.0
        self.lambda_edge = cfg.loss.lambda_edge    # 1.0
        self.lambda_score = cfg.loss.lambda_score  # 1.0
        self.lambda_point = cfg.loss.lambda_point  # 0.5
        
        # 基础损失函数
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, predictions, targets):
        """
        前向传播计算损失 - 全模型端到端训练
        
        Args:
            predictions: dict 模型预测结果
            targets: dict 真实标签
        """
        losses = {}
        total_loss = 0.0
        
        # 严格按照论文公式9实现: L = λ_m * L_mask + λ_e * L_edge + λ_s * L_score + λ_p * L_point
        
        # 1. Mask Loss (掩码损失) - 论文要求: Focal Loss + Dice Loss
        if 'mask' in predictions:
            mask_loss = self._compute_mask_loss(predictions, targets)
            losses['mask_loss'] = mask_loss
            total_loss += self.lambda_mask * mask_loss
        
        # 2. Edge Loss (边缘损失) - 论文要求: Focal Loss + Dice Loss
        if 'edge_features' in predictions:
            edge_loss = self._compute_edge_loss(predictions, targets)
            losses['edge_loss'] = edge_loss
            total_loss += self.lambda_edge * edge_loss
        
        # 3. Point Loss (出血点损失) - 论文公式10: Σ 1{pi≠[0,0]} · L1(ŷpi, ypi)
        if 'point' in predictions:
            point_loss = self._compute_point_loss(predictions, targets)
            losses['point_loss'] = point_loss
            total_loss += self.lambda_point * point_loss
        
        # 4. Score Loss (置信度损失) - 论文要求: Binary Cross-Entropy Loss
        if 'point_score' in predictions:
            score_loss = self._compute_score_loss(predictions, targets)
            losses['score_loss'] = score_loss
            total_loss += self.lambda_score * score_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_mask_loss(self, predictions, targets):
        """计算掩码损失 - 论文要求: Focal Loss + Dice Loss (单尺度)"""
        pred_mask = predictions['mask']  # [N, 1, H, W]
        gt_mask = targets['mask']        # [N, 1, H, W]
        
        # 论文要求: Focal Loss + Dice Loss (移除多尺度损失)
        focal = self.focal_loss(pred_mask, gt_mask)
        dice = self.dice_loss(torch.sigmoid(pred_mask), gt_mask)
        
        return focal + dice
    
    def _compute_edge_loss(self, predictions, targets):
        """计算边缘损失 - 基于Edge Generator输出"""
        pred_edge = predictions['edge_features']  # [N, 1, H, W] - 来自MaskBranch的edge_features
        
        # 生成边缘真实标签
        if 'edge_gt' in targets:
            gt_edge = targets['edge_gt']
        else:
            gt_edge = self._generate_edge_ground_truth(targets['mask'])
        
        # Focal Loss (用于处理边缘像素不平衡)
        focal = self.focal_loss(pred_edge, gt_edge)
        
        # Dice Loss (用于衡量边缘重叠度)
        dice = self.dice_loss(torch.sigmoid(pred_edge), gt_edge)
        
        return focal + dice
    
    def _compute_point_loss(self, predictions, targets):
        """
        计算出血点损失 - 严格按照论文公式10实现
        论文公式: L_point = Σ(i=1 to N) 1_{p_i≠[0,0]} · L1(ŷ_p^i, y_p^i)
        """
        pred_coords = predictions['point']  # [N, 2] - ŷ_p^i
        gt_coords = targets['point_coords'] # [N, 2] - y_p^i
        point_exists = targets.get('point_exists', torch.ones(pred_coords.shape[0], 1))  # [N, 1]
        
        # 实现指示函数 1_{p_i≠[0,0]} - 只对存在点的样本计算损失
        valid_mask = point_exists.squeeze(-1) > 0.5  # [N]
        
        if valid_mask.sum() > 0:
            valid_pred = pred_coords[valid_mask]  # [N_valid, 2]
            valid_gt = gt_coords[valid_mask]      # [N_valid, 2]
            
            # L1 Loss (使用Smooth L1作为L1的稳定版本)
            point_loss = self.smooth_l1_loss(valid_pred, valid_gt)
            
            # 移除热力图损失 (原论文中没有)
            # if 'heatmap' in predictions:
            #     heatmap_loss = self._compute_heatmap_loss(predictions, targets, valid_mask)
            #     point_loss += heatmap_loss
            
            return point_loss
        else:
            return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
    
    def _compute_score_loss(self, predictions, targets):
        """计算置信度损失 - Binary Cross-Entropy Loss"""
        pred_score = predictions['point_score']  # [N, 1]
        gt_exists = targets.get('point_exists', torch.ones_like(pred_score))  # [N, 1]
        
        # Binary Cross-Entropy Loss
        return self.bce_loss(pred_score, gt_exists.float())
    
    # 移除的方法 (原论文中没有的损失组件):
    # - _compute_flow_loss: 光流损失
    # - _compute_guidance_loss: 跨分支引导损失  
    # - _compute_heatmap_loss: 热力图损失
    
    def _generate_edge_ground_truth(self, gt_mask):
        """生成边缘真实标签"""
        device = gt_mask.device
        batch_size = gt_mask.shape[0]
        edge_maps = []
        
        for i in range(batch_size):
            mask = gt_mask[i, 0].cpu().numpy().astype(np.uint8)
            
            # Canny边缘检测
            edges = cv2.Canny(mask * 255, 50, 150)
            
            # 形态学膨胀
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            edge_tensor = torch.from_numpy(edges / 255.0).float().unsqueeze(0)
            edge_maps.append(edge_tensor)
        
        return torch.stack(edge_maps, dim=0).to(device)
    
    # 移除的辅助方法 (原论文中没有的功能):
    # - _generate_gaussian_heatmap: 高斯热力图生成
    # - _has_flow_predictions: 光流预测检查
    # - _has_guidance_predictions: 引导预测检查

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡问题"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        pred: [N, 1, H, W] logits
        target: [N, 1, H, W] binary mask
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算pt
        pt = torch.exp(-bce_loss)
        
        # 计算alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss - 衡量分割重叠度"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: [N, 1, H, W] probabilities (after sigmoid)
        target: [N, 1, H, W] binary mask
        """
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        return 1.0 - dice

# 移除的复杂损失类 (原论文中没有):
# - ComprehensiveFlowLoss: 综合光流损失 (包含时序一致性、平滑性等)
# - CrossBranchGuidanceLoss: 跨分支引导损失
# - AdaptiveLossWeights: 自适应权重学习

# 只保留论文中明确提及的基础损失函数: FocalLoss 和 DiceLoss
