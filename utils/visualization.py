"""
BlooDet可视化工具
用于训练过程监控、结果展示和调试
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as patches

def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    将光流转换为颜色编码的可视化图像
    flow: [H, W, 2] numpy数组
    返回: [H, W, 3] RGB图像
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # 计算角度和幅度
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    
    if max_flow is None:
        max_flow = np.max(v)
    
    # 创建HSV图像
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (ang / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = np.minimum(v / max_flow * 255, 255).astype(np.uint8)  # Value
    
    # 转换为RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def create_flow_wheel() -> np.ndarray:
    """创建光流颜色轮用于图例"""
    h, w = 151, 151
    center = (w // 2, h // 2)
    
    # 创建坐标网格
    y, x = np.mgrid[0:h, 0:w]
    x = x - center[0]
    y = y - center[1]
    
    # 计算角度和距离
    angle = np.arctan2(y, x) + np.pi
    radius = np.sqrt(x**2 + y**2)
    
    # 创建HSV图像
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle / (2 * np.pi) * 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.minimum(radius / (w // 2) * 255, 255).astype(np.uint8)
    
    # 创建圆形掩码
    mask = radius <= (w // 2)
    hsv[~mask] = 0
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def visualize_prediction_results(frames: torch.Tensor,
                               predictions: Dict[str, torch.Tensor],
                               ground_truth: Dict[str, torch.Tensor],
                               save_path: Optional[str] = None,
                               show_flow: bool = True) -> None:
    """
    可视化预测结果
    frames: [T, 3, H, W] 输入帧序列
    predictions: 模型预测结果
    ground_truth: 真实标签
    """
    T, C, H, W = frames.shape
    
    # 设置图像布局
    if show_flow and 'flows' in predictions:
        fig, axes = plt.subplots(3, T, figsize=(T * 4, 12))
    else:
        fig, axes = plt.subplots(2, T, figsize=(T * 4, 8))
    
    if T == 1:
        axes = axes.reshape(-1, 1)
    
    # 转换数据格式
    frames_np = frames.cpu().numpy().transpose(0, 2, 3, 1)  # [T, H, W, 3]
    pred_mask = torch.sigmoid(predictions['mask']).cpu().numpy()  # [1, H, W]
    gt_mask = ground_truth['mask'].cpu().numpy()  # [1, H, W]
    
    for t in range(T):
        # 显示原始帧
        axes[0, t].imshow(frames_np[t])
        axes[0, t].set_title(f'Frame {t}')
        axes[0, t].axis('off')
        
        # 叠加预测和真实掩码
        overlay = frames_np[t].copy()
        
        # 预测掩码（绿色）
        if t == T - 1:  # 只在最后一帧显示掩码
            pred_mask_resized = cv2.resize(pred_mask[0], (W, H))
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred_mask_resized * 0.7)
            
            # 真实掩码（红色轮廓）
            gt_mask_resized = cv2.resize(gt_mask[0], (W, H))
            contours, _ = cv2.findContours((gt_mask_resized > 0.5).astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, (1, 0, 0), 2)
        
        axes[1, t].imshow(overlay)
        axes[1, t].set_title(f'Prediction (Green) vs GT (Red)')
        axes[1, t].axis('off')
        
        # 显示光流（如果有）
        if show_flow and 'flows' in predictions and t < T - 1:
            flow = predictions['flows'][0, t].cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
            flow_color = flow_to_color(flow)
            axes[2, t].imshow(flow_color)
            axes[2, t].set_title(f'Optical Flow {t}->{t+1}')
            axes[2, t].axis('off')
        elif show_flow:
            axes[2, t].axis('off')
    
    # 添加点预测可视化
    if 'point' in predictions and 'point' in ground_truth:
        pred_point = predictions['point'].cpu().numpy()[0]  # [2]
        gt_point = ground_truth['point'].cpu().numpy()[0]  # [2]
        gt_exists = ground_truth['exists'].cpu().numpy()[0]  # [1]
        
        if gt_exists > 0.5:
            # 在最后一帧上标记点
            axes[1, -1].add_patch(Circle(pred_point, 3, color='yellow', fill=False, linewidth=2))
            axes[1, -1].add_patch(Circle(gt_point, 3, color='blue', fill=False, linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(log_dir: str, save_path: Optional[str] = None) -> None:
    """
    绘制训练曲线
    log_dir: TensorBoard日志目录
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not available for plotting training curves")
        return
    
    # 读取TensorBoard日志
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 获取可用的标量标签
    scalar_tags = event_acc.Tags()['scalars']
    
    # 设置图像布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制总损失
    if 'Epoch/Total_Loss' in scalar_tags:
        steps, values = zip(*[(s.step, s.value) for s in event_acc.Scalars('Epoch/Total_Loss')])
        axes[0, 0].plot(steps, values, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # 绘制各组件损失
    loss_components = ['Mask_Loss', 'Point_Loss', 'Score_Loss', 'Flow_Loss']
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (component, color) in enumerate(zip(loss_components, colors)):
        tag = f'Epoch/{component}'
        if tag in scalar_tags:
            steps, values = zip(*[(s.step, s.value) for s in event_acc.Scalars(tag)])
            axes[0, 1].plot(steps, values, color=color, label=component, linewidth=2)
    
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 绘制学习率
    if 'Learning_Rate/Encoder' in scalar_tags:
        steps, values = zip(*[(s.step, s.value) for s in event_acc.Scalars('Learning_Rate/Encoder')])
        axes[1, 0].plot(steps, values, 'g-', linewidth=2, label='Encoder')
    
    if 'Learning_Rate/Others' in scalar_tags:
        steps, values = zip(*[(s.step, s.value) for s in event_acc.Scalars('Learning_Rate/Others')])
        axes[1, 0].plot(steps, values, 'r-', linewidth=2, label='Others')
    
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 绘制光流损失组件
    flow_components = ['temporal_consistency', 'flow_smoothness', 'motion_consistency']
    for i, component in enumerate(flow_components):
        tag = f'FlowLoss/{component}'
        if tag in scalar_tags:
            steps, values = zip(*[(s.step, s.value) for s in event_acc.Scalars(tag)])
            axes[1, 1].plot(steps, values, label=component, linewidth=2)
    
    axes[1, 1].set_title('Flow Loss Components')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_demo_visualization(model, sample_data: Dict, save_dir: str) -> None:
    """
    创建演示可视化
    model: 训练好的模型
    sample_data: 样本数据
    save_dir: 保存目录
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        frames = sample_data['frames']  # [T, 3, H, W]
        gt_mask = sample_data['mask']   # [1, H, W]
        gt_point = sample_data['point'] # [2]
        gt_exists = sample_data['exists'] # [1]
        
        # 模型预测
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)  # [1, T, 3, H, W]
        
        predictions = model(frames)
        
        # 准备数据
        frames_vis = frames[0]  # [T, 3, H, W]
        ground_truth = {
            'mask': gt_mask,
            'point': gt_point,
            'exists': gt_exists
        }
        
        # 创建综合可视化
        visualize_prediction_results(
            frames_vis, predictions, ground_truth,
            save_path=save_dir / 'prediction_results.png',
            show_flow=True
        )
        
        # 创建光流轮图例
        if 'flows' in predictions:
            flow_wheel = create_flow_wheel()
            plt.figure(figsize=(6, 6))
            plt.imshow(flow_wheel)
            plt.title('Optical Flow Color Coding')
            plt.axis('off')
            plt.savefig(save_dir / 'flow_color_wheel.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Demo visualization saved to {save_dir}")

def analyze_model_performance(predictions: List[Dict], 
                            ground_truths: List[Dict],
                            save_path: Optional[str] = None) -> Dict:
    """
    分析模型性能并生成报告
    predictions: 预测结果列表
    ground_truths: 真实标签列表
    """
    from utils.metrics import calculate_metrics
    
    all_metrics = []
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = calculate_metrics(pred, gt['mask'], gt['point'], gt['exists'])
        all_metrics.append(metrics)
    
    # 计算统计信息
    stats = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    # 创建性能分析图
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IoU分布
        iou_values = [m['mask_iou'] for m in all_metrics]
        axes[0, 0].hist(iou_values, bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(stats['mask_iou']['mean'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mask_iou"]["mean"]:.3f}')
        axes[0, 0].set_title('Mask IoU Distribution')
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 点定位误差分布
        point_errors = [m['point_distance'] for m in all_metrics]
        axes[0, 1].hist(point_errors, bins=30, alpha=0.7, color='green')
        axes[0, 1].axvline(stats['point_distance']['mean'], color='red', linestyle='--',
                          label=f'Mean: {stats["point_distance"]["mean"]:.3f}')
        axes[0, 1].set_title('Point Localization Error Distribution')
        axes[0, 1].set_xlabel('Distance (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 性能指标雷达图
        metrics_names = ['mask_iou', 'mask_dice', 'point_accuracy', 'score_accuracy']
        metrics_values = [stats[name]['mean'] for name in metrics_names]
        
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 闭合雷达图
        angles += angles[:1]
        
        axes[1, 0] = plt.subplot(2, 2, 3, projection='polar')
        axes[1, 0].plot(angles, metrics_values, 'o-', linewidth=2)
        axes[1, 0].fill(angles, metrics_values, alpha=0.25)
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(metrics_names)
        axes[1, 0].set_title('Performance Radar Chart')
        
        # 性能对比表
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = []
        for key, stat in stats.items():
            table_data.append([key, f"{stat['mean']:.4f}", f"{stat['std']:.4f}"])
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Metric', 'Mean', 'Std'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved to {save_path}")
    
    return stats

# 使用示例函数
def save_training_visualizations(predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor],
                                epoch: int, 
                                log_dir: str, 
                                writer) -> None:
    """
    保存训练过程中的可视化结果
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        epoch: 当前epoch
        log_dir: 日志目录
        writer: TensorBoard writer
    """
    try:
        # 创建可视化目录
        vis_dir = Path(log_dir) / 'visualizations' / f'epoch_{epoch:03d}'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取最后一帧进行可视化
        if 'frames' in predictions:
            frames = predictions['frames'][-1]  # [T, 3, H, W] -> [3, H, W]
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)  # [1, 3, H, W]
        else:
            return
        
        # 准备数据
        frames_np = frames[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        
        # 创建图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(frames_np)
        axes[0, 0].set_title('Input Frame')
        axes[0, 0].axis('off')
        
        # 预测掩码
        if 'mask' in predictions:
            pred_mask = torch.sigmoid(predictions['mask']).cpu().numpy()[0]  # [H, W]
            axes[0, 1].imshow(pred_mask, cmap='hot')
            axes[0, 1].set_title('Predicted Mask')
            axes[0, 1].axis('off')
        
        # 真实掩码
        if 'mask' in targets:
            gt_mask = targets['mask'].cpu().numpy()[0]  # [H, W]
            axes[0, 2].imshow(gt_mask, cmap='hot')
            axes[0, 2].set_title('Ground Truth Mask')
            axes[0, 2].axis('off')
        
        # 预测vs真实对比
        if 'mask' in predictions and 'mask' in targets:
            pred_mask = torch.sigmoid(predictions['mask']).cpu().numpy()[0]
            gt_mask = targets['mask'].cpu().numpy()[0]
            
            # 创建对比图
            comparison = np.zeros((*pred_mask.shape, 3))
            comparison[:, :, 0] = gt_mask  # 红色通道：真实掩码
            comparison[:, :, 1] = pred_mask  # 绿色通道：预测掩码
            comparison[:, :, 2] = 0  # 蓝色通道：背景
            
            axes[1, 0].imshow(comparison)
            axes[1, 0].set_title('GT(Red) vs Pred(Green)')
            axes[1, 0].axis('off')
        
        # 点预测可视化
        if 'point_coords' in predictions and 'point_coords' in targets:
            pred_point = predictions['point_coords'].cpu().numpy()[0]  # [2]
            gt_point = targets['point_coords'].cpu().numpy()[0]  # [2]
            gt_exists = targets['point_exists'].cpu().numpy()[0]  # [1]
            
            # 在原始图像上标记点
            axes[1, 1].imshow(frames_np)
            if gt_exists > 0.5:
                axes[1, 1].add_patch(Circle(gt_point, 5, color='red', fill=False, linewidth=2, label='GT'))
            axes[1, 1].add_patch(Circle(pred_point, 5, color='yellow', fill=False, linewidth=2, label='Pred'))
            axes[1, 1].set_title('Point Localization')
            axes[1, 1].legend()
            axes[1, 1].axis('off')
        
        # 光流可视化（如果有）
        if 'flows' in predictions:
            flow = predictions['flows'][0, 0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
            flow_color = flow_to_color(flow)
            axes[1, 2].imshow(flow_color)
            axes[1, 2].set_title('Optical Flow')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = vis_dir / 'training_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 添加到TensorBoard
        if writer is not None:
            # 读取保存的图像并添加到TensorBoard
            img_array = plt.imread(save_path)
            writer.add_image('Training/Visualization', img_array, epoch, dataformats='HWC')
        
        print(f"Training visualization saved to {save_path}")
        
    except Exception as e:
        print(f"Error in save_training_visualizations: {e}")
        if 'fig' in locals():
            plt.close(fig)

def create_visualization_demo():
    """创建可视化演示的示例函数"""
    print("BlooDet Visualization Tools")
    print("Available functions:")
    print("1. visualize_prediction_results() - 可视化预测结果")
    print("2. plot_training_curves() - 绘制训练曲线")
    print("3. create_demo_visualization() - 创建演示可视化")
    print("4. analyze_model_performance() - 分析模型性能")
    print("5. save_training_visualizations() - 保存训练可视化")
    print("\nUsage examples:")
    print("# 可视化预测结果")
    print("visualize_prediction_results(frames, predictions, ground_truth)")
    print("\n# 绘制训练曲线")
    print("plot_training_curves('./logs/run_20241201_120000')")
    print("\n# 创建演示")
    print("create_demo_visualization(model, sample_data, './demo_output')")
    print("\n# 保存训练可视化")
    print("save_training_visualizations(predictions, targets, epoch, log_dir, writer)")

if __name__ == "__main__":
    create_visualization_demo()