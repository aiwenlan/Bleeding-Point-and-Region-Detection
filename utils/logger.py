import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(log_dir, name='BlooDet', level=logging.INFO):
    """设置日志记录器"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录启动信息
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger

def get_logger(name='BlooDet'):
    """获取已存在的日志记录器"""
    return logging.getLogger(name)

def log_training_info(logger, cfg):
    """记录训练配置信息"""
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    
    # 数据配置
    logger.info("Data Configuration:")
    logger.info(f"  Root: {cfg.data.root}")
    logger.info(f"  Image Size: {cfg.data.img_size}")
    logger.info(f"  Window Size: {cfg.data.window_size}")
    logger.info(f"  Train Split: {cfg.data.train_split}")
    logger.info(f"  Test Split: {cfg.data.test_split}")
    
    # 模型配置
    logger.info("Model Configuration:")
    logger.info(f"  SAM2 Checkpoint: {cfg.model.sam2_ckpt}")
    logger.info(f"  PWCNet Checkpoint: {cfg.model.pwcnet_ckpt}")
    logger.info(f"  Mask Memory Length: {cfg.model.mask_memory_len}")
    logger.info(f"  Point Memory Length: {cfg.model.point_memory_len}")
    
    # 训练配置
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {cfg.train.epochs}")
    logger.info(f"  Batch Size: {cfg.train.batch_size}")
    logger.info(f"  Learning Rate (Encoder): {cfg.train.lr_encoder}")
    logger.info(f"  Learning Rate (Others): {cfg.train.lr_others}")
    logger.info(f"  Optimizer: {cfg.train.optimizer}")
    logger.info(f"  Scheduler: {cfg.train.scheduler}")
    
    # 损失配置
    logger.info("Loss Configuration:")
    logger.info(f"  Lambda Mask: {cfg.loss.lambda_mask}")
    logger.info(f"  Lambda Point: {cfg.loss.lambda_point}")
    logger.info(f"  Lambda Score: {cfg.loss.lambda_score}")
    
    logger.info("=" * 50)

def log_epoch_info(logger, epoch, train_loss, val_loss=None, lr=None):
    """记录每个epoch的信息"""
    log_msg = f"Epoch {epoch:3d} - Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        log_msg += f" - Val Loss: {val_loss:.4f}"
    if lr is not None:
        log_msg += f" - LR: {lr:.2e}"
    
    logger.info(log_msg)

def log_metrics(logger, metrics, prefix=""):
    """记录评估指标"""
    logger.info(f"{prefix} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

def log_model_info(logger, model):
    """记录模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
