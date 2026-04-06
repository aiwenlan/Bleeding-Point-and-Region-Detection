"""
BlooDet 评估结果可视化工具
基于论文要求的评估指标可视化

功能:
1. IoU, Dice 分布直方图
2. PCK-2%, PCK-5%, PCK-10% 结果展示
3. 出血点定位误差分析
4. 预测结果对比可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2
import torch
from typing import Dict, List, Tuple, Optional
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BlooDet_Visualizer:
    """BlooDet 评估结果可视化器"""
    
    def __init__(self, output_dir: str = 'visualization_results'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def plot_metrics_distribution(self, metrics: Dict, save_path: Optional[str] = None):
        """
        绘制评估指标分布图
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BlooDet 评估指标分布', fontsize=16, fontweight='bold')
        
        # 1. IoU分布
        if 'mask_metrics' in metrics:
            mask = metrics['mask_metrics']
            
            # IoU
            ax = axes[0, 0]
            ax.bar(['IoU'], [mask['mean_iou']], yerr=[mask['std_iou']], 
                   color='skyblue', capsize=5, alpha=0.7)
            ax.set_ylabel('IoU 值')
            ax.set_title('出血区域 IoU (交并比)')
            ax.set_ylim(0, 1)
            ax.text(0, mask['mean_iou'] + mask['std_iou'] + 0.05, 
                   f"{mask['mean_iou']:.3f} ± {mask['std_iou']:.3f}", 
                   ha='center', fontweight='bold')
            
            # Dice
            ax = axes[0, 1]
            ax.bar(['Dice'], [mask['mean_dice']], yerr=[mask['std_dice']], 
                   color='lightgreen', capsize=5, alpha=0.7)
            ax.set_ylabel('Dice 值')
            ax.set_title('出血区域 Dice 系数')
            ax.set_ylim(0, 1)
            ax.text(0, mask['mean_dice'] + mask['std_dice'] + 0.05, 
                   f"{mask['mean_dice']:.3f} ± {mask['std_dice']:.3f}", 
                   ha='center', fontweight='bold')
        
        # 2. PCK结果
        if 'point_metrics' in metrics:
            point = metrics['point_metrics']
            
            # PCK柱状图
            ax = axes[0, 2]
            pck_names = []
            pck_values = []
            pck_stds = []
            
            for threshold in [2, 5, 10]:
                pck_key = f'pck_{threshold}'
                if pck_key in point:
                    pck_names.append(f'PCK-{threshold}%')
                    pck_values.append(point[pck_key])
                    pck_stds.append(point.get(f'{pck_key}_std', 0))
            
            if pck_names:
                bars = ax.bar(pck_names, pck_values, yerr=pck_stds, 
                             color=['coral', 'gold', 'lightcoral'], capsize=5, alpha=0.7)
                ax.set_ylabel('PCK 值')
                ax.set_title('出血点定位 PCK 指标')
                ax.set_ylim(0, 1)
                
                # 添加数值标签
                for i, (bar, value, std) in enumerate(zip(bars, pck_values, pck_stds)):
                    ax.text(bar.get_x() + bar.get_width()/2, value + std + 0.05, 
                           f'{value:.3f}', ha='center', fontweight='bold')
            
            # 距离误差
            ax = axes[1, 0]
            ax.bar(['距离误差'], [point['mean_distance']], yerr=[point['std_distance']], 
                   color='salmon', capsize=5, alpha=0.7)
            ax.set_ylabel('像素距离')
            ax.set_title('出血点定位距离误差')
            ax.text(0, point['mean_distance'] + point['std_distance'] + 1, 
                   f"{point['mean_distance']:.1f} ± {point['std_distance']:.1f} px", 
                   ha='center', fontweight='bold')
        
        # 3. 置信度指标
        if 'score_metrics' in metrics:
            score = metrics['score_metrics']
            
            ax = axes[1, 1]
            score_names = ['准确率', '精确率', '召回率', 'F1分数']
            score_values = [score['accuracy'], score['precision'], score['recall'], score['f1_score']]
            
            bars = ax.bar(score_names, score_values, 
                         color=['mediumpurple', 'mediumseagreen', 'orange', 'deeppink'], 
                         alpha=0.7)
            ax.set_ylabel('分数')
            ax.set_title('出血点存在性预测')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, value in zip(bars, score_values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                       f'{value:.3f}', ha='center', fontweight='bold')
        
        # 4. 数据集统计
        if 'dataset_stats' in metrics:
            stats = metrics['dataset_stats']
            
            ax = axes[1, 2]
            labels = ['含出血点', '无出血点']
            sizes = [stats['samples_with_points'], stats['samples_without_points']]
            colors = ['lightblue', 'lightcoral']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax.set_title(f'数据集分布 (总计: {stats["total_samples"]} 样本)')
            
            # 美化饼图
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 指标分布图已保存: {save_path}")
    
    def plot_pck_analysis(self, predictions_data: List[Dict], save_path: Optional[str] = None):
        """
        绘制PCK分析图
        
        Args:
            predictions_data: 预测数据列表
            save_path: 保存路径
        """
        # 收集所有距离数据
        all_distances = []
        all_exists = []
        
        for pred_data in predictions_data:
            if 'metrics' in pred_data and 'batch_distances' in pred_data['metrics']:
                distances = pred_data['metrics']['batch_distances']
                all_distances.extend(distances)
                
                # 获取存在性信息
                if 'targets' in pred_data and 'point_exists' in pred_data['targets']:
                    exists = pred_data['targets']['point_exists']
                    if hasattr(exists, 'tolist'):
                        exists = exists.tolist()
                    all_exists.extend(exists)
        
        if not all_distances:
            print("⚠️ 没有找到距离数据，跳过PCK分析")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('PCK 分析', fontsize=16, fontweight='bold')
        
        # 1. 距离分布直方图
        ax = axes[0]
        ax.hist(all_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('预测距离误差 (像素)')
        ax.set_ylabel('频次')
        ax.set_title('出血点定位距离误差分布')
        ax.axvline(np.mean(all_distances), color='red', linestyle='--', 
                  label=f'均值: {np.mean(all_distances):.2f}px')
        ax.legend()
        
        # 2. PCK曲线
        ax = axes[1]
        img_diagonal = np.sqrt(512**2 + 512**2)  # 假设图像尺寸512x512
        thresholds = np.linspace(0.01, 0.15, 50)
        pck_values = []
        
        for threshold in thresholds:
            pck_threshold_px = threshold * img_diagonal
            pck = np.mean(np.array(all_distances) < pck_threshold_px)
            pck_values.append(pck)
        
        ax.plot(thresholds * 100, pck_values, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('PCK 阈值 (%)')
        ax.set_ylabel('PCK 值')
        ax.set_title('PCK 曲线')
        ax.grid(True, alpha=0.3)
        
        # 标记关键点
        for key_threshold in [0.02, 0.05, 0.10]:
            key_pck = np.interp(key_threshold, thresholds, pck_values)
            ax.axvline(key_threshold * 100, color='red', linestyle='--', alpha=0.7)
            ax.text(key_threshold * 100, key_pck + 0.05, 
                   f'PCK-{int(key_threshold*100)}%\n{key_pck:.3f}', 
                   ha='center', fontsize=10, fontweight='bold')
        
        # 3. 误差累积分布
        ax = axes[2]
        sorted_distances = np.sort(all_distances)
        cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        
        ax.plot(sorted_distances, cumulative, 'g-', linewidth=2)
        ax.set_xlabel('距离误差 (像素)')
        ax.set_ylabel('累积分布函数')
        ax.set_title('误差累积分布')
        ax.grid(True, alpha=0.3)
        
        # 标记关键百分位数
        for percentile in [50, 75, 90, 95]:
            value = np.percentile(all_distances, percentile)
            ax.axvline(value, color='red', linestyle='--', alpha=0.7)
            ax.text(value, percentile/100 - 0.1, 
                   f'P{percentile}: {value:.1f}px', 
                   rotation=90, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pck_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 PCK分析图已保存: {save_path}")
    
    def visualize_predictions(self, predictions_data: List[Dict], 
                            num_samples: int = 8, save_path: Optional[str] = None):
        """
        可视化预测结果对比
        
        Args:
            predictions_data: 预测数据列表
            num_samples: 显示样本数
            save_path: 保存路径
        """
        if not predictions_data:
            print("⚠️ 没有预测数据可视化")
            return
        
        # 选择样本
        selected_samples = []
        for pred_data in predictions_data[:num_samples]:
            if 'predictions' in pred_data and 'targets' in pred_data:
                selected_samples.append(pred_data)
            if len(selected_samples) >= num_samples:
                break
        
        if not selected_samples:
            print("⚠️ 没有有效的预测数据")
            return
        
        fig, axes = plt.subplots(len(selected_samples), 4, figsize=(16, 4*len(selected_samples)))
        fig.suptitle('BlooDet 预测结果可视化', fontsize=16, fontweight='bold')
        
        if len(selected_samples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_data in enumerate(selected_samples):
            predictions = sample_data['predictions']
            targets = sample_data['targets']
            
            # 获取第一个样本的数据
            if 'mask' in predictions and len(predictions['mask']) > 0:
                pred_mask = predictions['mask'][0]  # [C, H, W]
                if isinstance(pred_mask, list):
                    pred_mask = np.array(pred_mask)
                elif hasattr(pred_mask, 'numpy'):
                    pred_mask = pred_mask.numpy()
                
                if len(pred_mask.shape) == 3:
                    pred_mask = pred_mask[0]  # 取第一个通道
                
                gt_mask = targets['mask'][0]
                if isinstance(gt_mask, list):
                    gt_mask = np.array(gt_mask)
                elif hasattr(gt_mask, 'numpy'):
                    gt_mask = gt_mask.numpy()
                
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[0]
                
                # 1. 真实掩码
                axes[i, 0].imshow(gt_mask, cmap='Reds', alpha=0.8)
                axes[i, 0].set_title('真实出血区域')
                axes[i, 0].axis('off')
                
                # 2. 预测掩码
                axes[i, 1].imshow(pred_mask, cmap='Blues', alpha=0.8)
                axes[i, 1].set_title('预测出血区域')
                axes[i, 1].axis('off')
                
                # 3. 重叠对比
                overlap = np.zeros((*gt_mask.shape, 3))
                overlap[:, :, 0] = gt_mask  # 红色：真实
                overlap[:, :, 2] = pred_mask  # 蓝色：预测
                overlap[:, :, 1] = gt_mask * pred_mask  # 绿色：重叠
                
                axes[i, 2].imshow(overlap)
                axes[i, 2].set_title('重叠对比 (红:GT, 蓝:预测, 绿:重叠)')
                axes[i, 2].axis('off')
                
                # 4. 出血点对比
                axes[i, 3].imshow(gt_mask, cmap='gray', alpha=0.3)
                
                # 绘制真实点
                if 'point_coords' in targets and len(targets['point_coords']) > 0:
                    gt_point = targets['point_coords'][0]
                    if isinstance(gt_point, list):
                        gt_point = np.array(gt_point)
                    elif hasattr(gt_point, 'numpy'):
                        gt_point = gt_point.numpy()
                    
                    if len(gt_point) >= 2:
                        axes[i, 3].plot(gt_point[0], gt_point[1], 'ro', markersize=10, 
                                       label='真实点', markeredgecolor='white', markeredgewidth=2)
                
                # 绘制预测点
                if 'point' in predictions and len(predictions['point']) > 0:
                    pred_point = predictions['point'][0]
                    if isinstance(pred_point, list):
                        pred_point = np.array(pred_point)
                    elif hasattr(pred_point, 'numpy'):
                        pred_point = pred_point.numpy()
                    
                    if len(pred_point) >= 2:
                        axes[i, 3].plot(pred_point[0], pred_point[1], 'bx', markersize=12, 
                                       markeredgewidth=3, label='预测点')
                        
                        # 绘制连接线
                        if 'point_coords' in targets and len(targets['point_coords']) > 0:
                            gt_point = targets['point_coords'][0]
                            if hasattr(gt_point, 'numpy'):
                                gt_point = gt_point.numpy()
                            if len(gt_point) >= 2:
                                axes[i, 3].plot([gt_point[0], pred_point[0]], 
                                               [gt_point[1], pred_point[1]], 
                                               'g--', alpha=0.7, linewidth=2)
                
                axes[i, 3].set_title('出血点定位对比')
                axes[i, 3].axis('off')
                axes[i, 3].legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🖼️ 预测结果对比图已保存: {save_path}")
    
    def create_evaluation_report(self, metrics: Dict, predictions_data: List[Dict] = None):
        """
        创建完整的评估报告
        
        Args:
            metrics: 评估指标
            predictions_data: 预测数据（可选）
        """
        print("📋 生成评估报告...")
        
        # 1. 指标分布图
        self.plot_metrics_distribution(metrics)
        
        # 2. PCK分析图
        if predictions_data:
            self.plot_pck_analysis(predictions_data)
            
            # 3. 预测结果可视化
            self.visualize_predictions(predictions_data, num_samples=6)
        
        # 4. 生成HTML报告
        self._generate_html_report(metrics, predictions_data)
        
        print(f"✅ 评估报告已生成完成，保存在: {self.output_dir}")
    
    def _generate_html_report(self, metrics: Dict, predictions_data: List[Dict] = None):
        """生成HTML格式的评估报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BlooDet 评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 30px 0; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .image-card {{ text-align: center; }}
                .image-card img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🩸 BlooDet 评估报告</h1>
                <p>基于论文要求的标准化评估平台</p>
            </div>
        """
        
        # 数据集统计
        if 'dataset_stats' in metrics:
            stats = metrics['dataset_stats']
            html_content += f"""
            <div class="section">
                <h2>📊 数据集统计</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{stats['total_samples']}</div>
                        <div class="metric-label">总样本数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['samples_with_points']}</div>
                        <div class="metric-label">含出血点样本</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['point_ratio']:.2%}</div>
                        <div class="metric-label">出血点比例</div>
                    </div>
                </div>
            </div>
            """
        
        # 出血区域评估
        if 'mask_metrics' in metrics:
            mask = metrics['mask_metrics']
            html_content += f"""
            <div class="section">
                <h2>🎯 出血区域评估指标</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{mask['mean_iou']:.4f}</div>
                        <div class="metric-label">IoU (交并比)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{mask['mean_dice']:.4f}</div>
                        <div class="metric-label">Dice 系数</div>
                    </div>
                </div>
            </div>
            """
        
        # 出血点评估
        if 'point_metrics' in metrics:
            point = metrics['point_metrics']
            html_content += f"""
            <div class="section">
                <h2>📍 出血点评估指标</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{point['mean_distance']:.2f}px</div>
                        <div class="metric-label">平均距离误差</div>
                    </div>
            """
            
            # PCK结果
            for threshold in [2, 5, 10]:
                pck_key = f'pck_{threshold}'
                if pck_key in point:
                    html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{point[pck_key]:.4f}</div>
                        <div class="metric-label">PCK-{threshold}%</div>
                    </div>
                    """
            
            html_content += "</div></div>"
        
        # 图表展示
        html_content += """
        <div class="section">
            <h2>📈 可视化结果</h2>
            <div class="image-grid">
                <div class="image-card">
                    <h3>指标分布</h3>
                    <img src="metrics_distribution.png" alt="指标分布图">
                </div>
        """
        
        if predictions_data:
            html_content += """
                <div class="image-card">
                    <h3>PCK 分析</h3>
                    <img src="pck_analysis.png" alt="PCK分析图">
                </div>
                <div class="image-card">
                    <h3>预测结果对比</h3>
                    <img src="prediction_comparison.png" alt="预测结果对比">
                </div>
            """
        
        html_content += """
            </div>
        </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        html_path = os.path.join(self.output_dir, 'evaluation_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML报告已保存: {html_path}")

def visualize_evaluation_results(metrics_file: str, predictions_file: str = None, 
                               output_dir: str = 'visualization_results'):
    """
    可视化评估结果的便捷函数
    
    Args:
        metrics_file: 评估指标JSON文件路径
        predictions_file: 预测结果JSON文件路径（可选）
        output_dir: 输出目录
    """
    # 加载指标数据
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # 加载预测数据（如果有）
    predictions_data = None
    if predictions_file and os.path.exists(predictions_file):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
    
    # 创建可视化器
    visualizer = BlooDet_Visualizer(output_dir)
    
    # 生成完整报告
    visualizer.create_evaluation_report(metrics, predictions_data)

if __name__ == '__main__':
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='BlooDet 评估结果可视化')
    parser.add_argument('--metrics', type=str, required=True, help='评估指标JSON文件')
    parser.add_argument('--predictions', type=str, help='预测结果JSON文件')
    parser.add_argument('--output', type=str, default='visualization_results', help='输出目录')
    
    args = parser.parse_args()
    
    visualize_evaluation_results(args.metrics, args.predictions, args.output)
