"""
BlooDet 训练脚本
全模型端到端训练，使用固定损失权重
"""

import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from tqdm import tqdm
import numpy as np

from utils.losses import BlooDet_Loss
from data.dataset import SurgBloodDataset
from modeling.blood_det import BlooDet
from utils.metrics import compute_metrics
from utils.visualization import save_training_visualizations

def load_cfg(config_path='configs/default.yaml'):
    """加载配置文件"""
    with open(config_path) as f: 
        return yaml.safe_load(f)

class Cfg: 
    pass

def to_obj(d):
    """字典转对象"""
    o = Cfg()
    for k, v in d.items():
        setattr(o, k, to_obj(v) if isinstance(v, dict) else v)
    return o

def setup_training(cfg):
    """设置训练环境"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    train_dataset = SurgBloodDataset(cfg, split='train')
    val_dataset = SurgBloodDataset(cfg, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size,
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时使用batch_size=1
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    # 模型
    model = BlooDet(cfg)
    model = model.to(device)
    
    # 损失函数
    criterion = BlooDet_Loss(cfg).to(device)
    
    # 优化器
    optimizer = setup_optimizer(model, cfg)
    
    # 学习率调度器
    scheduler = setup_scheduler(optimizer, cfg)
    
    # TensorBoard
    writer = None
    if cfg.logging.tensorboard:
        log_dir = os.path.join(cfg.logging.log_dir, cfg.experiment.name)
        writer = SummaryWriter(log_dir)
    
    return model, criterion, optimizer, scheduler, train_loader, val_loader, writer, device

def setup_optimizer(model, cfg):
    """设置优化器 - 按论文实现细节配置差分学习率"""
    # 获取优化器配置
    opt_cfg = cfg.train.optimizer_config
    
    # 按论文要求设置差分学习率
    # 图像编码器使用5e-6的较小学习率
    # 其他网络部分使用5e-4的学习率
    
    encoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'sam2_model' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组
    param_groups = [
        {
            'params': encoder_params,
            'lr': float(opt_cfg.image_encoder_lr),  # 确保是浮点数
            'name': 'image_encoder'
        },
        {
            'params': other_params, 
            'lr': float(opt_cfg.other_parts_lr),   # 确保是浮点数
            'name': 'other_parts'
        }
    ]
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam(
        param_groups,
        betas=list(opt_cfg.betas),  # 确保betas是列表
        eps=float(opt_cfg.eps),  # 确保eps是浮点数
        weight_decay=float(opt_cfg.weight_decay)  # 确保weight_decay是浮点数
    )
    
    print(f"✅ 优化器配置完成:")
    print(f"   - 图像编码器参数: {len(encoder_params)} 个, 学习率: {opt_cfg.image_encoder_lr}")
    print(f"   - 其他网络参数: {len(other_params)} 个, 学习率: {opt_cfg.other_parts_lr}")
    print(f"   - 优化器类型: Adam")
    
    return optimizer

def setup_scheduler(optimizer, cfg):
    """设置学习率调度器 - 按论文实现：热身+线性衰减"""
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    
    sched_cfg = cfg.train.scheduler_config
    
    # 按论文要求：热身(warm-up) + 线性衰减(linear decay)
    # 热身阶段
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,  # 从10%开始
        end_factor=1.0,    # 热身到100%
        total_iters=int(sched_cfg.warmup_epochs)  # 确保是整数
    )
    
    # 线性衰减阶段 - 使用其他部分的学习率作为基准
    base_lr = float(cfg.train.optimizer_config.other_parts_lr)  # 从optimizer_config获取
    decay_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=float(sched_cfg.eta_min) / base_lr,  # 确保是浮点数
        total_iters=int(sched_cfg.total_epochs) - int(sched_cfg.warmup_epochs)  # 确保是整数
    )
    
    # 组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[int(sched_cfg.warmup_epochs)]  # 确保是整数列表
    )
    
    print(f"✅ 学习率调度器配置完成:")
    print(f"   - 热身周期: {sched_cfg.warmup_epochs} epochs")
    print(f"   - 总训练周期: {sched_cfg.total_epochs} epochs") 
    print(f"   - 最小学习率: {sched_cfg.eta_min}")
    
    return scheduler

def train_epoch(model, criterion, optimizer, train_loader, device, cfg, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    total_losses = {}
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.train.epochs}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 数据转移到设备
        frames_seq = batch['frames'].to(device)  # [N, T, 3, H, W] frames_seq: torch.Size([1, 8, 3, 512, 512])
        # print('###################################')
        # print(f"frames_seq: {frames_seq.shape}")
        # target 图片序列最后一帧对应的标签
        targets = {
            'mask': batch['mask'].to(device),                    # [N, 1, H, W]
            'point_coords': batch['point_coords'].to(device),    # [N, 2]
            'point_exists': batch['point_exists'].to(device),    # [N, 1]
            'frames_seq': frames_seq
        }
        
        # 前向传播
        optimizer.zero_grad()
        
        try:
            with torch.cuda.amp.autocast(enabled=getattr(cfg.train, 'mixed_precision', False)):
                predictions = model(frames_seq)
                # print('###################################')
                # print(f"predictions: {predictions['mask'].shape}")predictions: torch.Size([1, 1, 512, 512])
                losses = criterion(predictions, targets)
        except Exception as e:
            print(f"❌ 训练前向传播错误 (batch {batch_idx}): {e}")
            print(f"   输入形状: {frames_seq.shape}")
            print(f"   目标键: {list(targets.keys())}")
            raise e
        
        # 反向传播
        total_loss = losses['total_loss']
        total_loss.backward()
        
        # 梯度裁剪
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
        
        optimizer.step()
        
        # 累积损失
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Total': f'{total_loss.item():.4f}',
            'Mask': f'{losses.get("mask_loss", 0):.4f}',
            'Point': f'{losses.get("point_loss", 0):.4f}',
            'Edge': f'{losses.get("edge_loss", 0):.4f}',
            'Score': f'{losses.get("score_loss", 0):.4f}'
        })
        
        # TensorBoard记录
        print_freq = getattr(cfg.logging, 'print_freq', 10) if hasattr(cfg, 'logging') else 10
        if writer and batch_idx % print_freq == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in losses.items():
                writer.add_scalar(f'Train_Batch/{key}', value.item(), global_step)
    
    # 计算平均损失
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    # 记录epoch级别的损失
    if writer:
        for key, value in avg_losses.items():
            writer.add_scalar(f'Train_Epoch/{key}', value, epoch)
        
        # 记录学习率
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)
    
    return avg_losses

def validate_epoch(model, criterion, val_loader, device, cfg, epoch, writer=None):
    """验证一个epoch"""
    model.eval()
    total_losses = {}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            frames_seq = batch['frames'].to(device)
            targets = {
                'mask': batch['mask'].to(device),
                'point_coords': batch['point_coords'].to(device),
                'point_exists': batch['point_exists'].to(device),
                'frames_seq': frames_seq
            }
            
            # 前向传播
            try:
                predictions = model(frames_seq)
                # print('###################################')
                # print(f"predictions: {predictions['mask'].shape}")
                losses = criterion(predictions, targets)
            except Exception as e:
                print(f"❌ 验证前向传播错误 (batch {batch_idx}): {e}")
                print(f"   输入形状: {frames_seq.shape}")
                print(f"   目标键: {list(targets.keys())}")
                raise e
            
            # 累积损失
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            # 收集预测和目标用于指标计算
            # 注意: 模型输出可能已经是sigmoid后的结果，需要检查
            pred_mask = predictions.get('mask', torch.zeros_like(targets['mask']))
            pred_point_score = predictions.get('point_score', torch.zeros_like(targets['point_exists']))
            
            # 检查是否需要sigmoid处理
            if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
                pred_mask = torch.sigmoid(pred_mask)
            if pred_point_score.max() > 1.0 or pred_point_score.min() < 0.0:
                pred_point_score = torch.sigmoid(pred_point_score)
            
            # 构建用于指标计算的预测结果
            batch_predictions = {
                'mask': pred_mask,
                'point': predictions.get('point', torch.zeros_like(targets['point_coords'])),
                'point_score': pred_point_score
            }
            print('###################################')
            print(f"mask_predictions: {batch_predictions['mask'].shape}")
            print(f"point_predictions: {batch_predictions['point'].shape}")
            print(f"point_score_predictions: {batch_predictions['point_score'].shape}")
            
            
            # 构建用于指标计算的目标结果（移除不需要的键）
            batch_targets = {
                'mask': targets['mask'],
                'point_coords': targets['point_coords'],
                'point_exists': targets['point_exists']
            }
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f"mask_targets: {targets['mask'].shape}")
            print(f"point_coords_targets: {targets['point_coords'].shape}")
            print(f"point_exists_targets: {targets['point_exists'].shape}")
            
            all_predictions.append(batch_predictions)
            all_targets.append(batch_targets)
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    # 计算指标
    metrics = compute_metrics(all_predictions, all_targets)
    
    # TensorBoard记录
    if writer:
        for key, value in avg_losses.items():
            writer.add_scalar(f'Val_Epoch/{key}', value, epoch)
        
        for key, value in metrics.items():
            writer.add_scalar(f'Val_Metrics/{key}', value, epoch)
        
        # 保存可视化结果
        if getattr(cfg.logging.visualization, 'enabled', False):
            save_training_visualizations(
                all_predictions[-1], all_targets[-1], 
                epoch, cfg.logging.log_dir, writer
            )
    
    return avg_losses, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, cfg, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'config': cfg
    }
    
    # 保存常规检查点
    if epoch % cfg.train.save_interval == 0:
        save_path = os.path.join(cfg.logging.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(cfg.logging.save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")

def main():
    parser = argparse.ArgumentParser(description='BlooDet Enhanced Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 加载配置
    cfg_dict = load_cfg(args.config)
    cfg = to_obj(cfg_dict)
    
    # 创建保存目录
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    
    # 设置训练
    model, criterion, optimizer, scheduler, train_loader, val_loader, writer, device = setup_training(cfg)
    
    # 打印模型信息
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Model info: {model_info}")
    else:
        # 手动计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # 检查模型初始化状态
    if hasattr(model, '_is_initialized') and not model._is_initialized:
        print("⚠️  警告: 模型未正确初始化")
    else:
        print("✅ 模型初始化正常")
    
    # 恢复训练
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # 训练循环
    print("Starting training...")
    print(f"训练配置:")
    print(f"  - 总epochs: {cfg.train.epochs}")
    print(f"  - 批次大小: {cfg.train.batch_size}")
    print(f"  - 学习率: {cfg.train.optimizer_config.other_parts_lr}")
    print(f"  - 图像编码器学习率: {cfg.train.optimizer_config.image_encoder_lr}")
    print(f"  - 验证间隔: {cfg.train.eval_interval}")
    
    for epoch in range(start_epoch, cfg.train.epochs):
        # 训练
        train_losses = train_epoch(model, criterion, optimizer, train_loader, device, cfg, epoch, writer)
        
        # 验证
        if epoch % cfg.train.eval_interval == 0:
            val_losses, val_metrics = validate_epoch(model, criterion, val_loader, device, cfg, epoch, writer)
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{cfg.train.epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f} (Mask: {train_losses.get('mask_loss', 0):.4f}, Point: {train_losses.get('point_loss', 0):.4f}, Edge: {train_losses.get('edge_loss', 0):.4f}, Score: {train_losses.get('score_loss', 0):.4f})")
            print(f"Val Loss: {val_losses['total_loss']:.4f} (Mask: {val_losses.get('mask_loss', 0):.4f}, Point: {val_losses.get('point_loss', 0):.4f}, Edge: {val_losses.get('edge_loss', 0):.4f}, Score: {val_losses.get('score_loss', 0):.4f})")
            print(f"Val IoU: {val_metrics.get('mask_iou', 0):.4f}")
            print(f"Val Dice: {val_metrics.get('mask_dice', 0):.4f}")
            print(f"Point Error: {val_metrics.get('point_distance', 0):.4f}")
            print(f"PCK@5%: {val_metrics.get('pck_5', 0):.4f}")
            print(f"Score Acc: {val_metrics.get('score_accuracy', 0):.4f}")
            
            # 保存最佳模型
            current_metric = val_metrics.get('mask_dice', 0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
            
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, cfg, is_best)
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if getattr(cfg.validation.early_stopping, 'enabled', False):
            # 这里可以添加早停逻辑
            pass
    
    print("Training completed!")
    
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
