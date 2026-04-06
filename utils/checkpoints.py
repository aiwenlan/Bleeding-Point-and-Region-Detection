import os
import torch
import shutil
from pathlib import Path

def save_checkpoint(epoch, model, optimizer, scheduler, filepath):
    """保存检查点"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from epoch {epoch}")
    
    return epoch, model, optimizer, scheduler

def get_latest_checkpoint(checkpoint_dir):
    """获取最新的检查点文件"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    
    # 按文件名中的epoch数排序
    def extract_epoch(filename):
        try:
            return int(filename.stem.split('_')[-1])
        except:
            return 0
    
    checkpoint_files.sort(key=extract_epoch, reverse=True)
    return str(checkpoint_files[0])

def cleanup_old_checkpoints(checkpoint_dir, max_keep=5):
    """清理旧的检查点文件，只保留最新的几个"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if len(checkpoint_files) <= max_keep:
        return
    
    # 按文件名中的epoch数排序
    def extract_epoch(filename):
        try:
            return int(filename.stem.split('_')[-1])
        except:
            return 0
    
    checkpoint_files.sort(key=extract_epoch, reverse=True)
    
    # 删除多余的检查点
    for checkpoint_file in checkpoint_files[max_keep:]:
        try:
            checkpoint_file.unlink()
            print(f"Removed old checkpoint: {checkpoint_file}")
        except Exception as e:
            print(f"Failed to remove {checkpoint_file}: {e}")

def save_best_model(epoch, model, optimizer, scheduler, filepath, is_best=False):
    """保存最佳模型"""
    save_checkpoint(epoch, model, optimizer, scheduler, filepath)
    
    if is_best:
        # 创建best_model.pth的副本
        best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        shutil.copy2(filepath, best_path)
        print(f"Best model saved to {best_path}")
