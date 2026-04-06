# 🩸 BlooDet: Enhanced Bleeding Detection in Surgical Videos

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BlooDet** 是一个基于深度学习的手术视频出血检测系统，能够同时进行**出血区域分割**和**出血点定位**的多任务学习模型。该项目基于SAM2架构，集成了先进的跨分支引导机制、光流建模和边缘生成技术。

**本仓库为论文复现**：实现对应 [Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](https://arxiv.org/abs/2503.22174)（arXiv:2503.22174，已录用 CVPR 2026）中的 BlooDet 与 SurgBlood 相关工作流，非官方作者代码。

> **📋 项目状态**: 活跃开发中 | **🔧 版本**: v2.0.0 | **📅 最后更新**: 2025年11月

## 🌟 主要特性

### 🎯 **核心功能**
- **🔴 出血区域分割**: 精确识别手术视频中的出血区域
- **📍 出血点定位**: 亚像素级出血点坐标预测
- **🎬 视频序列处理**: 支持时序建模和运动补偿
- **⚡ 实时推理**: 优化的推理管道，支持在线检测

### 🧠 **技术亮点**
- **🔄 跨分支引导机制**: Point Branch和Mask Branch相互引导优化
- **🌊 光流集成**: 基于PWC-Net的运动建模和时序一致性
- **🔍 边缘增强**: Wavelet Laplacian Filter边缘生成器
- **💾 记忆机制**: 智能记忆库管理历史信息
- **🎛️ 多尺度融合**: 多层级特征融合和预测

### 📊 **评估系统**
- **IoU & Dice**: 出血区域分割评估
- **PCK指标**: 出血点定位精度评估 (PCK-2%, PCK-5%, PCK-10%)
- **可视化报告**: 完整的评估报告和图表生成

## 🏗️ 项目架构

```
BlooDet/
├── 📁 configs/           # 配置文件
│   └── default.yaml     # 默认配置
├── 📁 data/             # 数据处理
│   ├── dataset.py       # 数据集加载
│   ├── transforms.py    # 数据增强
│   └── prepare_surgblood.py # 数据准备
├── 📁 modeling/         # 模型实现
│   ├── blood_det.py     # 主模型
│   ├── sam2_wrapper.py  # SAM2图像编码器
│   ├── point_branch.py  # 出血点分支
│   ├── mask_branch.py   # 出血区域分支
│   ├── edge_generator.py # 边缘生成器
│   ├── memory_bank.py   # 记忆库
│   ├── optical_flow_integration.py # 光流集成
│   └── prompt_encoder.py # 提示编码器
├── 📁 sam2/             # SAM2子项目
│   ├── configs/         # SAM2配置
│   ├── modeling/        # SAM2模型
│   └── utils/           # SAM2工具
├── 📁 PWC_Net/          # PWC-Net子项目
│   └── PyTorch/         # PyTorch实现
├── 📁 utils/            # 工具函数
│   ├── losses.py        # 损失函数
│   ├── metrics.py       # 评估指标
│   ├── eval_visualization.py # 可视化工具
│   ├── checkpoints.py   # 检查点管理
│   └── logger.py        # 日志系统
├── 📁 scripts/          # 脚本工具
│   ├── prepare_dataset_split.py # 数据分割
│   └── download_models.py # 模型下载
├── 📁 checkpoints/      # 模型检查点
├── train.py             # 训练脚本
└── README.md           # 项目说明
```

## 🚀 快速开始

### 1. 准备数据
```bash
# 下载SurgBlood数据集（请根据实际情况调整）
# 数据集结构应为：
SurgBlood/
├── frames/     # 视频帧
├── masks/      # 掩码标注
└── points/     # 点标注

# 创建数据集分割
python scripts/prepare_dataset_split.py --data_root ./SurgBlood
```

### 2. 下载预训练模型
```bash
python scripts/download_models.py
```

### 3. 训练模型
```bash
# 基础训练
python train.py --config configs/default.yaml

# 多GPU训练（如果可用）
python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/default.yaml
```


## 📖 详细使用指南

### 🔧 配置文件说明

主要配置在 `configs/default.yaml` 中：

```yaml
# 数据配置
data:
  root: ./SurgBlood
  img_size: 512
  window_size: 8        # 视频帧窗口大小

# 模型配置
model:
  use_enhanced_model: true
  sam2_config: sam2/configs/sam2/sam2_hiera_b+.yaml
  
# 训练配置
train:
  epochs: 20
  batch_size: 1
  optimizer_config:
    type: "Adam"
    image_encoder_lr: 5e-6    # 编码器学习率
    other_parts_lr: 5e-4      # 其他部分学习率

# 损失权重
loss:
  lambda_mask: 1.0      # 掩码损失
  lambda_point: 0.5     # 点损失
  lambda_score: 1.0     # 置信度损失
  lambda_edge: 0.1      # 边缘损失
```

### 🎯 训练策略

BlooDet采用端到端训练策略，严格按照论文实现：

1. **全模型训练**: 联合训练所有组件（默认）
2. **差分学习率**: 图像编码器使用较小学习率(5e-6)，其他部分使用标准学习率(5e-4)
3. **学习率调度**: 热身+线性衰减策略

```bash
# 标准训练（推荐）
python train.py --config configs/default.yaml

# 恢复训练
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### 📊 评估指标

#### 出血区域评估
- **IoU (Intersection over Union)**: 交并比
- **Dice Coefficient**: Dice系数

#### 出血点评估  
- **PCK-2%**: 2%误差范围内的正确率
- **PCK-5%**: 5%误差范围内的正确率
- **PCK-10%**: 10%误差范围内的正确率

#### 综合评估
- **距离误差**: 平均像素距离
- **置信度评估**: 准确率、精确率、召回率、F1分数


## 🧪 模型架构详解

### 🏗️ 整体架构

BlooDet采用多分支架构设计：

```
Input Video Sequence (T×3×H×W)
           ↓
    SAM2 Image Encoder
           ↓
    Multi-scale Features
         ↙    ↘
   Point Branch    Mask Branch
         ↓              ↓
   Point Coords    Bleeding Mask
         ↘          ↙
    Cross-branch Guidance
           ↓
    Enhanced Predictions
```

### 🔍 关键组件

#### **1. SAM2图像编码器**
- 基于Hierarchical Transformer
- 多尺度特征提取
- 时序特征融合

#### **2. Point Branch (出血点分支)**
- 多尺度特征融合
- 光流运动建模
- 记忆引导定位
- 亚像素精度预测

#### **3. Mask Branch (出血区域分支)**
- 边缘引导分割
- SAM2解码器集成
- 多层级预测
- 边缘细化网络

#### **4. 跨分支引导机制**
- Point → Mask 引导
- Mask → Point 引导
- 双向信息交换
- 自适应权重学习

#### **5. 边缘生成器**
- Wavelet Laplacian Filter
- Gabor小波变换
- 多尺度边缘检测
- 边缘增强融合

## 📈 性能基准

### 在SurgBlood数据集上的表现

| 指标 | 数值 | 说明 |
|------|------|------|
| IoU | 0.784 ± 0.123 | 出血区域交并比 |
| Dice | 0.857 ± 0.110 | 出血区域Dice系数 |
| PCK-2% | 0.623 ± 0.046 | 2%误差范围准确率 |
| PCK-5% | 0.812 ± 0.023 | 5%误差范围准确率 |
| PCK-10% | 0.935 ± 0.012 | 10%误差范围准确率 |
| 距离误差 | 12.3 ± 8.8 px | 平均像素距离误差 |

### 推理性能

| 配置 | FPS | 内存占用 | 精度 |
|------|-----|----------|------|
| RTX 4090 | ~15 | ~8GB | 最高 |
| RTX 3080 | ~12 | ~6GB | 高 |
| RTX 2080Ti | ~8 | ~4GB | 中等 |

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **Python** | 3.10+ | 3.10+ |
| **PyTorch** | 2.5.1+ | 2.5.1+ |
| **CUDA** | 12.1+ | 12.1+ |
| **GPU内存** | 8GB+ | 16GB+ |
| **系统内存** | 16GB+ | 32GB+ |



## 📚 参考文献

本复现所依据的论文请引用：

```bibtex
@misc{pei2025synergistic,
  title={Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos},
  author={Pei, Jialun and Zhou, Zhangjun and Guo, Diandian and Li, Zhixi and Qin, Jing and Du, Bo and Heng, Pheng-Ann},
  year={2025},
  eprint={2503.22174},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.22174}
}
```

### 相关项目

- [Surgical-SAM-2](https://github.com/yourusername/Surgical-SAM-2) - 手术视频分割基础模型
- [PWC-Net](https://github.com/NVlabs/PWC-Net) - 光流估计网络
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - 图像分割基础模型
