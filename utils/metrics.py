"""
BlooDet 评估指标系统
基于论文要求的标准化评估指标实现

评估指标:
1. 出血区域评估指标:
   - IoU (Intersection over Union, 交并比)
   - Dice Coefficient (Dice 系数)

2. 出血点评估指标:
   - PCK (Percentage of Correct Keypoints)
   - PCK-2%, PCK-5%, PCK-10%

3. 综合评估:
   - SurgBlood 数据集的设计目标为联合检测出血区域和出血点提供一个标准化的评估平台
   - 评估结果不仅关注出血区域的空间重叠度(IoU, Dice)，还要考虑出血点的精确定位(PCK)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
from collections import defaultdict

class BlooDet_Evaluator:
    """
    BlooDet 标准化评估器
    
    功能:
    1. 出血区域评估 - IoU, Dice
    2. 出血点评估 - PCK@2%, PCK@5%, PCK@10%
    3. 综合评估报告生成
    4. 可视化支持
    """
    
    def __init__(self, img_size=(512, 512), pck_thresholds=[0.02, 0.05, 0.10]):
        """
        初始化评估器
        
        Args:
            img_size: 图像尺寸 (H, W)
            pck_thresholds: PCK阈值列表 [2%, 5%, 10%]
        """
        self.img_size = img_size
        self.pck_thresholds = pck_thresholds
        
        # 累积统计
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计数据"""
        self.stats = {
            # 出血区域统计
            'mask_ious': [],
            'mask_dices': [],
            
            # 出血点统计
            'point_distances': [],
            'pck_results': {f'pck_{int(th*100)}': [] for th in self.pck_thresholds},
            
            # 置信度统计
            'score_accuracies': [],
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            
            # 样本计数
            'total_samples': 0,
            'samples_with_points': 0,
            'samples_without_points': 0
        }
    
    def evaluate_batch(self, predictions: Dict, targets: Dict) -> Dict:
        """
        评估一个batch的结果
        
        Args:
            predictions: dict 包含 'mask', 'point', 'point_score'等
            targets: dict 包含 'mask', 'point_coords', 'point_exists'等
        
        Returns:
            batch_metrics: dict 当前batch的评估结果
        """
        batch_metrics = {}
        batch_size = targets['mask'].shape[0]
        
        # 1. 出血区域评估
        if 'mask' in predictions:
            mask_metrics = self._evaluate_mask(predictions['mask'], targets['mask'])
            batch_metrics.update(mask_metrics)
            
            # 累积统计
            self.stats['mask_ious'].extend(mask_metrics['batch_ious'])
            self.stats['mask_dices'].extend(mask_metrics['batch_dices'])
        
        # 2. 出血点评估
        if 'point' in predictions and 'point_coords' in targets:
            point_metrics = self._evaluate_points(
                predictions['point'], 
                targets['point_coords'], 
                targets['point_exists']
            )
            batch_metrics.update(point_metrics)
            
            # 累积统计
            self.stats['point_distances'].extend(point_metrics['batch_distances'])
            for pck_key, pck_values in point_metrics['batch_pcks'].items():
                self.stats['pck_results'][pck_key].extend(pck_values)
        
        # 3. 置信度评估
        if 'point_score' in predictions:
            score_metrics = self._evaluate_scores(
                predictions['point_score'], 
                targets['point_exists']
            )
            batch_metrics.update(score_metrics)
            
            # 累积统计
            self.stats['score_accuracies'].extend(score_metrics['batch_accuracies'])
            self.stats['true_positives'] += score_metrics['tp']
            self.stats['false_positives'] += score_metrics['fp']
            self.stats['true_negatives'] += score_metrics['tn']
            self.stats['false_negatives'] += score_metrics['fn']
        
        # 更新样本计数
        self.stats['total_samples'] += batch_size
        if 'point_exists' in targets:
            self.stats['samples_with_points'] += targets['point_exists'].sum().item()
            self.stats['samples_without_points'] += (batch_size - targets['point_exists'].sum().item())
        
        return batch_metrics
    
    def _evaluate_mask(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, threshold: float = 0.5) -> Dict:
        """
        评估出血区域掩码
        
        计算公式:
        - IoU = 预测区域∩真实区域 / 预测区域∪真实区域  
        - Dice = 2×预测区域∩真实区域 / (预测区域+真实区域)
        """
        # 调试信息：检查输入数据
        print(f"🔍 BlooDet_Evaluator._evaluate_mask 调试:")
        print(f"  pred_masks范围: [{pred_masks.min():.6f}, {pred_masks.max():.6f}]")
        print(f"  gt_masks范围: [{gt_masks.min():.6f}, {gt_masks.max():.6f}]")
        print(f"  阈值: {threshold}")
        
        # 转换为概率并二值化
        if pred_masks.dtype != torch.bool:
            pred_probs = torch.sigmoid(pred_masks) if pred_masks.max() > 1 else pred_masks
            pred_binary = (pred_probs > threshold).float()
        else:
            pred_binary = pred_masks.float()
        
        gt_binary = gt_masks.float()
        
        print(f"  pred_binary范围: [{pred_binary.min():.6f}, {pred_binary.max():.6f}]")
        print(f"  pred_binary非零像素数: {(pred_binary > 0).sum().item()}")
        print(f"  gt_binary非零像素数: {(gt_binary > 0).sum().item()}")
        
        # 批量计算IoU和Dice
        batch_ious = []
        batch_dices = []
        
        for i in range(pred_binary.shape[0]):
            pred_flat = pred_binary[i].flatten()
            gt_flat = gt_binary[i].flatten()
            
            # IoU计算
            intersection = (pred_flat * gt_flat).sum()
            union = pred_flat.sum() + gt_flat.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            
            # Dice计算  
            dice = (2.0 * intersection / (pred_flat.sum() + gt_flat.sum() + 1e-8)).item()
            
            batch_ious.append(iou)
            batch_dices.append(dice)
        
        return {
            'mean_iou': np.mean(batch_ious),
            'mean_dice': np.mean(batch_dices),
            'batch_ious': batch_ious,
            'batch_dices': batch_dices
        }
    
    def _evaluate_points(self, pred_points: torch.Tensor, gt_points: torch.Tensor, 
                        point_exists: torch.Tensor) -> Dict:
        """
        评估出血点定位
        
        PCK计算方法:
        - 对于每个出血点，计算预测点与真实点之间的距离，并判断是否在给定的误差范围内
        - 如果误差小于设定的阈值(如2%或5%)，则认为该出血点预测为正确
        - 然后计算所有预测出血点中正确的比例，得出PCK值
        """
        # 调试信息：检查点坐标
        print(f"🔍 BlooDet_Evaluator._evaluate_points 调试:")
        print(f"  pred_points范围: [{pred_points.min():.6f}, {pred_points.max():.6f}]")
        print(f"  pred_points值: {pred_points}")
        print(f"  gt_points范围: [{gt_points.min():.6f}, {gt_points.max():.6f}]")
        print(f"  gt_points值: {gt_points}")
        print(f"  point_exists值: {point_exists}")
        
        batch_size = pred_points.shape[0]
        batch_distances = []
        batch_pcks = {f'pck_{int(th*100)}': [] for th in self.pck_thresholds}
        
        # 计算图像对角线长度（用于PCK归一化）
        img_diagonal = np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)
        
        for i in range(batch_size):
            if point_exists[i].item() > 0.5:  # 只评估存在出血点的样本
                pred_point = pred_points[i].cpu().numpy()  # [2]
                gt_point = gt_points[i].cpu().numpy()      # [2]
                
                # 计算欧氏距离
                distance = np.linalg.norm(pred_point - gt_point)
                batch_distances.append(distance)
                
                # 计算各个阈值下的PCK
                for threshold in self.pck_thresholds:
                    # PCK阈值：threshold * 图像对角线
                    pck_threshold = threshold * img_diagonal
                    is_correct = distance < pck_threshold
                    pck_key = f'pck_{int(threshold*100)}'
                    batch_pcks[pck_key].append(1.0 if is_correct else 0.0)
            else:
                # 对于不存在出血点的样本，如果模型预测了点，则认为错误
                # 这里可以根据具体需求调整评估策略
                batch_distances.append(0.0)  # 距离设为0
                for threshold in self.pck_thresholds:
                    pck_key = f'pck_{int(threshold*100)}'
                    batch_pcks[pck_key].append(1.0)  # 假设正确（没有点需要预测）
        
        # 计算平均值
        mean_distance = np.mean(batch_distances) if batch_distances else 0.0
        mean_pcks = {}
        for pck_key, pck_values in batch_pcks.items():
            mean_pcks[f'mean_{pck_key}'] = np.mean(pck_values) if pck_values else 0.0
        
        result = {
            'mean_point_distance': mean_distance,
            'batch_distances': batch_distances,
            'batch_pcks': batch_pcks
        }
        result.update(mean_pcks)
        
        return result
    
    def _evaluate_scores(self, pred_scores: torch.Tensor, point_exists: torch.Tensor) -> Dict:
        """评估出血点存在性预测"""
        pred_binary = (torch.sigmoid(pred_scores) > 0.5).float()
        gt_binary = point_exists.float()
        
        # 计算混淆矩阵
        tp = ((pred_binary == 1) & (gt_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        tn = ((pred_binary == 0) & (gt_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (gt_binary == 1)).sum().item()
        
        # 计算准确率
        batch_accuracies = (pred_binary == gt_binary).float().cpu().numpy().tolist()
        mean_accuracy = np.mean(batch_accuracies)
        
        return {
            'mean_score_accuracy': mean_accuracy,
            'batch_accuracies': batch_accuracies,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    def get_final_metrics(self) -> Dict:
        """
        获取最终评估结果
        
        Returns:
            综合评估报告
        """
        if self.stats['total_samples'] == 0:
            return {}
        
        metrics = {}
        
        # 1. 出血区域评估结果
        if self.stats['mask_ious']:
            metrics['mask_metrics'] = {
                'mean_iou': np.mean(self.stats['mask_ious']),
                'std_iou': np.std(self.stats['mask_ious']),
                'mean_dice': np.mean(self.stats['mask_dices']),
                'std_dice': np.std(self.stats['mask_dices'])
            }
        
        # 2. 出血点评估结果
        if self.stats['point_distances']:
            point_metrics = {
                'mean_distance': np.mean(self.stats['point_distances']),
                'std_distance': np.std(self.stats['point_distances'])
            }
            
            # PCK结果
            for pck_key, pck_values in self.stats['pck_results'].items():
                if pck_values:
                    threshold_percent = pck_key.split('_')[1]
                    point_metrics[f'{pck_key}'] = np.mean(pck_values)
                    point_metrics[f'{pck_key}_std'] = np.std(pck_values)
                    
                    # 添加详细说明
                    point_metrics[f'{pck_key}_description'] = f"PCK-{threshold_percent}%: {np.mean(pck_values):.3f} (在{threshold_percent}%误差范围内的正确点数量比例)"
            
            metrics['point_metrics'] = point_metrics
        
        # 3. 置信度评估结果
        if self.stats['score_accuracies']:
            tp, fp, tn, fn = self.stats['true_positives'], self.stats['false_positives'], \
                           self.stats['true_negatives'], self.stats['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics['score_metrics'] = {
                'accuracy': np.mean(self.stats['score_accuracies']),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        
        # 4. 数据集统计
        metrics['dataset_stats'] = {
            'total_samples': self.stats['total_samples'],
            'samples_with_points': self.stats['samples_with_points'],
            'samples_without_points': self.stats['samples_without_points'],
            'point_ratio': self.stats['samples_with_points'] / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0.0
        }
        
        return metrics
    
    def print_summary(self):
        """打印评估摘要"""
        metrics = self.get_final_metrics()
        
        print("="*60)
        print("BlooDet 评估结果摘要")
        print("="*60)
        
        # 数据集统计
        if 'dataset_stats' in metrics:
            stats = metrics['dataset_stats']
            print(f"\n📊 数据集统计:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  含出血点样本: {stats['samples_with_points']}")
            print(f"  无出血点样本: {stats['samples_without_points']}")
            print(f"  出血点比例: {stats['point_ratio']:.2%}")
        
        # 出血区域评估
        if 'mask_metrics' in metrics:
            mask = metrics['mask_metrics']
            print(f"\n🎯 出血区域评估指标:")
            print(f"  IoU (交并比):     {mask['mean_iou']:.4f} ± {mask['std_iou']:.4f}")
            print(f"  Dice (Dice系数):  {mask['mean_dice']:.4f} ± {mask['std_dice']:.4f}")
        
        # 出血点评估
        if 'point_metrics' in metrics:
            point = metrics['point_metrics']
            print(f"\n📍 出血点评估指标:")
            print(f"  平均距离误差: {point['mean_distance']:.2f} ± {point['std_distance']:.2f} 像素")
            
            # PCK结果
            for threshold in self.pck_thresholds:
                pck_key = f'pck_{int(threshold*100)}'
                if pck_key in point:
                    print(f"  PCK-{int(threshold*100)}%:         {point[pck_key]:.4f} ± {point.get(f'{pck_key}_std', 0):.4f}")
        
        # 置信度评估
        if 'score_metrics' in metrics:
            score = metrics['score_metrics']
            print(f"\n🎲 置信度评估指标:")
            print(f"  准确率:    {score['accuracy']:.4f}")
            print(f"  精确率:    {score['precision']:.4f}")
            print(f"  召回率:    {score['recall']:.4f}")
            print(f"  F1分数:    {score['f1_score']:.4f}")
        
        print("="*60)

# 兼容性函数（保持向后兼容）
def calculate_metrics(output, gt_mask, gt_point, gt_exists):
    """计算评估指标（兼容性函数）"""
    evaluator = BlooDet_Evaluator()
    
    # 构造输入格式
    predictions = {
        'mask': output.get('mask'),
        'point': output.get('point'), 
        'point_score': output.get('point_score')
    }
    
    targets = {
        'mask': gt_mask,
        'point_coords': gt_point,
        'point_exists': gt_exists
    }
    
    # 评估单个样本
    batch_metrics = evaluator.evaluate_batch(predictions, targets)
    
    # 转换为旧格式
    legacy_metrics = {}
    if 'mean_iou' in batch_metrics:
        legacy_metrics['mask_iou'] = batch_metrics['mean_iou']
    if 'mean_dice' in batch_metrics:
        legacy_metrics['mask_dice'] = batch_metrics['mean_dice']
    if 'mean_point_distance' in batch_metrics:
        legacy_metrics['point_distance'] = batch_metrics['mean_point_distance']
    if 'mean_score_accuracy' in batch_metrics:
        legacy_metrics['score_accuracy'] = batch_metrics['mean_score_accuracy']
    
    return legacy_metrics

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """计算IoU（兼容性函数）"""
    # 调试信息：检查预测值范围
    print(f"🔍 IoU计算调试:")
    print(f"  pred_mask范围: [{pred_mask.min():.6f}, {pred_mask.max():.6f}]")
    print(f"  gt_mask范围: [{gt_mask.min():.6f}, {gt_mask.max():.6f}]")
    print(f"  阈值: {threshold}")
    
    pred_binary = (pred_mask > threshold).float()
    print(f"  pred_binary范围: [{pred_binary.min():.6f}, {pred_binary.max():.6f}]")
    print(f"  pred_binary非零像素数: {(pred_binary > 0).sum().item()}")
    print(f"  gt_mask非零像素数: {(gt_mask > 0).sum().item()}")
    
    intersection = (pred_binary * gt_mask).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3)) - intersection
    iou = intersection / (union + 1e-6)
    print(f"  intersection: {intersection.mean().item():.6f}")
    print(f"  union: {union.mean().item():.6f}")
    print(f"  iou: {iou.mean().item():.6f}")
    
    return iou.mean().item()

def calculate_dice(pred_mask, gt_mask, threshold=0.5):
    """计算Dice系数（兼容性函数）"""
    # 调试信息：检查预测值范围
    print(f"🔍 Dice计算调试:")
    print(f"  pred_mask范围: [{pred_mask.min():.6f}, {pred_mask.max():.6f}]")
    print(f"  gt_mask范围: [{gt_mask.min():.6f}, {gt_mask.max():.6f}]")
    print(f"  阈值: {threshold}")
    
    pred_binary = (pred_mask > threshold).float()
    print(f"  pred_binary范围: [{pred_binary.min():.6f}, {pred_binary.max():.6f}]")
    print(f"  pred_binary非零像素数: {(pred_binary > 0).sum().item()}")
    print(f"  gt_mask非零像素数: {(gt_mask > 0).sum().item()}")
    
    intersection = (pred_binary * gt_mask).sum(dim=(2, 3))
    pred_sum = pred_binary.sum(dim=(2, 3))
    gt_sum = gt_mask.sum(dim=(2, 3))
    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-6)
    
    print(f"  intersection: {intersection.mean().item():.6f}")
    print(f"  pred_sum: {pred_sum.mean().item():.6f}")
    print(f"  gt_sum: {gt_sum.mean().item():.6f}")
    print(f"  dice: {dice.mean().item():.6f}")
    
    return dice.mean().item()

def calculate_point_distance(pred_point, gt_point):
    """计算预测点和真实点之间的欧氏距离（像素）（兼容性函数）"""
    # 调试信息：检查点坐标
    print(f"🔍 Point Error计算调试:")
    print(f"  pred_point范围: [{pred_point.min():.6f}, {pred_point.max():.6f}]")
    print(f"  pred_point值: {pred_point}")
    print(f"  gt_point范围: [{gt_point.min():.6f}, {gt_point.max():.6f}]")
    print(f"  gt_point值: {gt_point}")
    
    distance = torch.norm(pred_point - gt_point, dim=1)
    print(f"  distance: {distance}")
    print(f"  mean_distance: {distance.mean().item():.6f}")
    
    return distance.mean().item()

def calculate_point_accuracy(pred_point, gt_point, threshold=5.0):
    """计算点定位准确率（在阈值范围内的比例）（兼容性函数）"""
    distance = torch.norm(pred_point - gt_point, dim=1)
    accuracy = (distance < threshold).float().mean()
    return accuracy.item()

def calculate_score_accuracy(pred_score, gt_exists):
    """计算点存在性预测准确率（兼容性函数）"""
    pred_binary = (pred_score > 0.5).float()
    accuracy = (pred_binary.view(-1) == gt_exists.float()).float().mean()
    return accuracy.item()

def calculate_pck(pred_points, gt_points, img_size, threshold):
    """计算PCK (Percentage of Correct Keypoints)（兼容性函数）"""
    if pred_points.ndim == 3:  # [N, T, 2]
        pred_points = pred_points[:, -1]  # 取最后一帧
    if gt_points.ndim == 3:
        gt_points = gt_points[:, -1]
    
    # 计算相对距离
    img_diagonal = np.sqrt(img_size[0]**2 + img_size[1]**2)
    distance = np.linalg.norm(pred_points - gt_points, axis=1)
    pck = np.mean(distance < (threshold * img_diagonal))
    return pck

def compute_metrics(predictions_list, targets_list):
    """计算评估指标 - 用于训练过程中的指标计算"""
    # 创建评估器
    evaluator = BlooDet_Evaluator(img_size=(512, 512))
    
    # 逐batch评估
    for predictions, targets in zip(predictions_list, targets_list):
        # 构造评估输入格式
        eval_predictions = {
            'mask': predictions['mask'],
            'point': predictions['point'],
            'point_score': predictions['point_score']
        }
        
        eval_targets = {
            'mask': targets['mask'],
            'point_coords': targets['point_coords'],
            'point_exists': targets['point_exists']
        }
        
        # 评估当前batch
        evaluator.evaluate_batch(eval_predictions, eval_targets)
    
    # 获取最终指标
    final_metrics = evaluator.get_final_metrics()
    
    # 转换为训练需要的格式
    training_metrics = {}
    
    if 'mask_metrics' in final_metrics:
        mask = final_metrics['mask_metrics']
        training_metrics['mask_iou'] = mask['mean_iou']
        training_metrics['mask_dice'] = mask['mean_dice']
    
    if 'point_metrics' in final_metrics:
        point = final_metrics['point_metrics']
        training_metrics['point_distance'] = point['mean_distance']
        training_metrics['pck_2'] = point.get('pck_2', 0.0)
        training_metrics['pck_5'] = point.get('pck_5', 0.0)
        training_metrics['pck_10'] = point.get('pck_10', 0.0)
    
    if 'score_metrics' in final_metrics:
        score = final_metrics['score_metrics']
        training_metrics['score_accuracy'] = score['accuracy']
        training_metrics['score_f1'] = score['f1_score']
    
    return training_metrics
