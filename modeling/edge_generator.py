# BlooDet edge generator
# 基于 Wavelet Laplacian Filter（小波拉普拉斯滤波器）的边缘生成器
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_gabor_wavelets(kernel_size=15, num_orientations=8, num_scales=3, 
                           sigma_x=2.0, sigma_y=2.0, lambda_min=4.0, lambda_max=12.0):
    """
    生成多尺度、多方向的Gabor小波滤波器组
    
    参数:
    - kernel_size: 滤波器核大小
    - num_orientations: 方向数量（通常8个方向）
    - num_scales: 尺度数量
    - sigma_x, sigma_y: 高斯包络的标准差
    - lambda_min, lambda_max: 波长范围
    """
    wavelets = []
    
    # 创建坐标网格
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    y = np.arange(kernel_size) - center
    X, Y = np.meshgrid(x, y)
    
    # 生成不同尺度的波长
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num_scales)
    
    for scale_idx, lambda_val in enumerate(lambdas):
        for orient_idx in range(num_orientations):
            # 方向角度
            theta = orient_idx * np.pi / num_orientations
            
            # 旋转坐标
            x_theta = X * np.cos(theta) + Y * np.sin(theta)
            y_theta = -X * np.sin(theta) + Y * np.cos(theta)
            
            # Gabor小波公式
            # G(x,y) = exp(-(x'^2/σx^2 + y'^2/σy^2)/2) * exp(i*(2π*x'/λ + ψ))
            gaussian = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
            sinusoid = np.cos(2 * np.pi * x_theta / lambda_val)  # 使用余弦部分
            
            gabor = gaussian * sinusoid
            
            # 归一化
            gabor = gabor - gabor.mean()
            gabor = gabor / (np.abs(gabor).sum() + 1e-8)
            
            wavelets.append(gabor)
    
    return np.array(wavelets)

def generate_laplacian_kernels():
    """生成不同的拉普拉斯算子"""
    # 标准拉普拉斯核
    laplacian_4 = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    
    # 8连通拉普拉斯核
    laplacian_8 = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)
    
    # 增强的拉普拉斯核
    laplacian_enhanced = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]], dtype=np.float32)
    
    return [laplacian_4, laplacian_8, laplacian_enhanced]

class GaborWaveletBank(nn.Module):
    """Gabor小波滤波器组"""
    
    def __init__(self, kernel_size=15, num_orientations=8, num_scales=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # 生成Gabor小波滤波器
        wavelets = generate_gabor_wavelets(kernel_size, num_orientations, num_scales)
        self.num_filters = len(wavelets)
        
        # 创建卷积层
        self.gabor_conv = nn.Conv2d(1, self.num_filters, kernel_size, 
                                   padding=kernel_size//2, bias=False)
        
        # 设置权重为Gabor小波（不可训练）
        with torch.no_grad():
            for i, wavelet in enumerate(wavelets):
                self.gabor_conv.weight[i, 0] = torch.from_numpy(wavelet).float()
        
        # 冻结参数
        self.gabor_conv.weight.requires_grad = False
    
    def forward(self, x):
        """
        x: [N, 1, H, W]
        返回: [N, num_filters, H, W] 多尺度多方向的小波响应
        """
        return self.gabor_conv(x)

class LaplacianOperator(nn.Module):
    """拉普拉斯算子组"""
    
    def __init__(self):
        super().__init__()
        
        # 生成拉普拉斯核
        laplacian_kernels = generate_laplacian_kernels()
        self.num_kernels = len(laplacian_kernels)
        
        # 创建卷积层
        self.laplacian_convs = nn.ModuleList()
        for kernel in laplacian_kernels:
            conv = nn.Conv2d(1, 1, kernel.shape[0], 
                           padding=kernel.shape[0]//2, bias=False)
            with torch.no_grad():
                conv.weight[0, 0] = torch.from_numpy(kernel).float()
            conv.weight.requires_grad = False
            self.laplacian_convs.append(conv)
    
    def forward(self, x):
        """
        x: [N, C, H, W]
        返回: [N, C*num_kernels, H, W]
        """
        results = []
        for conv in self.laplacian_convs:
            if x.size(1) == 1:
                result = conv(x)
            else:
                # 对每个通道分别应用拉普拉斯算子
                channel_results = []
                for c in range(x.size(1)):
                    channel_result = conv(x[:, c:c+1])
                    channel_results.append(channel_result)
                result = torch.cat(channel_results, dim=1)
            results.append(result)
        
        return torch.cat(results, dim=1)

class GaborWaveletLaplacianFilter(nn.Module):
    """Gabor Wavelet Laplacian Filter - 实现论文公式 (5), (6), (7)"""
    
    def __init__(self, kernel_size=15, num_orientations=8, num_scales=3, 
                 sigma=1.0, gamma=0.5, psi=0.0, lambda_min=4.0, lambda_max=12.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.sigma = sigma
        self.gamma = gamma
        self.psi = psi
        
        # 生成Gabor小波核
        self.gabor_kernels = self._generate_gabor_kernels(lambda_min, lambda_max)
        self.num_filters = len(self.gabor_kernels)
        
        # 创建卷积层
        self.gabor_conv = nn.Conv2d(1, self.num_filters, kernel_size, 
                                   padding=kernel_size//2, bias=False)
        
        # 设置权重为Gabor小波核（不可训练）
        with torch.no_grad():
            for i, kernel in enumerate(self.gabor_kernels):
                self.gabor_conv.weight[i, 0] = torch.from_numpy(kernel).float()
        
        # 冻结参数
        self.gabor_conv.weight.requires_grad = False
        
        # 拉普拉斯算子 - 实现公式 (7)
        self.laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.num_filters, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
    
    def _generate_gabor_kernels(self, lambda_min, lambda_max):
        """生成Gabor小波核 - 实现公式 (5)"""
        kernels = []
        
        # 创建坐标网格
        center = self.kernel_size // 2
        x = np.arange(self.kernel_size) - center
        y = np.arange(self.kernel_size) - center
        X, Y = np.meshgrid(x, y)
        
        # 生成不同尺度的波长
        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), self.num_scales)
        
        for scale_idx, lambda_val in enumerate(lambdas):
            for orient_idx in range(self.num_orientations):
                # 方向角度
                theta = orient_idx * np.pi / self.num_orientations
                
                # 旋转坐标 - 公式 (5) 中的 x', y'
                x_theta = X * np.cos(theta) + Y * np.sin(theta)
                y_theta = -X * np.sin(theta) + Y * np.cos(theta)
                
                # Gabor小波公式 (5)
                # G(x,y) = exp(-(x'^2 + γ^2y'^2) / (2σ^2)) * exp(i(2π/λ * x' + ψ))
                
                # 第一个exp: 高斯包络
                gaussian = np.exp(-(x_theta**2 + self.gamma**2 * y_theta**2) / (2 * self.sigma**2))
                
                # 第二个exp: 复正弦平面波 exp(i(2π/λ * x' + ψ))
                # 这里使用欧拉公式: exp(iθ) = cos(θ) + i*sin(θ)
                # 我们取实部: cos(2π/λ * x' + ψ)
                phase = 2 * np.pi * x_theta / lambda_val + self.psi  # 使用相位偏移ψ
                complex_sinusoid = np.cos(phase)  # 取实部
                
                # 两个exp的乘积
                gabor = gaussian * complex_sinusoid
                
                # 归一化
                gabor = gabor - gabor.mean()
                gabor = gabor / (np.abs(gabor).sum() + 1e-8)
                
                kernels.append(gabor)
        
        return np.array(kernels)
    
    def forward(self, x):
        """
        实现公式 (6): Lg(x,y) = Δf(x,y) · G(x,y)
        x: [N, 1, H, W] 输入图像
        返回: [N, 1, H, W] Gabor-Laplacian滤波结果
        """
        # Step 1: Gabor小波变换 - 公式 (5)
        gabor_responses = self.gabor_conv(x)  # [N, num_filters, H, W]
        
        # Step 2: 对每个Gabor响应应用拉普拉斯算子 - 公式 (6)
        laplacian_responses = []
        for i in range(self.num_filters):
            # 对每个Gabor响应应用拉普拉斯算子
            gabor_single = gabor_responses[:, i:i+1]  # [N, 1, H, W]
            laplacian_kernel = self.laplacian_kernel.to(x.device)
            laplacian_single = F.conv2d(gabor_single, laplacian_kernel, padding=1)
            laplacian_responses.append(laplacian_single)
        
        # 拼接所有拉普拉斯响应
        laplacian_concat = torch.cat(laplacian_responses, dim=1)  # [N, num_filters, H, W]
        
        # Step 3: 特征融合
        edge_map = self.feature_fusion(laplacian_concat)  # [N, 1, H, W]
        
        return edge_map
    
    def get_filter_kernel(self):
        """
        获取Lg(x,y)滤波器核，用于实现公式(8)中的卷积操作
        返回: [1, 1, kernel_size, kernel_size] 滤波器核
        """
        # 使用拉普拉斯核作为Lg(x,y)的简化表示
        # 在实际应用中，这里应该是完整的Gabor Wavelet Laplacian Filter
        return self.laplacian_kernel  # [1, 1, 3, 3]

class EdgeGenerator(nn.Module):
    """
    基于Wavelet Laplacian Filter的边缘生成器
    
    工作原理:
    1. 使用Gabor Wavelet进行多尺度小波变换，获取不同尺度的图像信息
    2. 利用拉普拉斯算子检测图像中的边缘，抑制背景噪声，保留出血区域的边缘细节
    3. 将增强后的边缘特征与高分辨率特征进行融合，以更好地捕捉细节信息
    4. 通过自适应提示机制，生成用于Mask Branch的边缘引导信息
    """
    
    def __init__(self, in_ch=256):
        super().__init__()
        
        # 特征降维
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        # 高分辨率特征处理 - 修改为接受256通道输入
        self.hi_res_proj = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding=1),  # 修改为256通道输入
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # 核心：Gabor Wavelet Laplacian滤波器 - 实现公式 (5), (6), (7)
        self.gabor_wavelet_laplacian = GaborWaveletLaplacianFilter(
            kernel_size=15, 
            num_orientations=8,  # 8个方向
            num_scales=3,        # 3个尺度
            sigma=1.0,           # 高斯标准差
            gamma=0.5,           # 纵横比
            psi=0.0,             # 相位偏移
            lambda_min=4.0,      # 最小波长
            lambda_max=12.0      # 最大波长
        )
        
        # 多尺度边缘增强 - 使用Conv2d Trans (转置卷积)
        # 根据图片结构，只有1个转置卷积：顶部路径
        # 使用 output_padding=1 确保输出尺寸精确为输入尺寸的2倍
        self.top_conv_trans = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 第二次转置卷积：顶部+中间路径融合后使用
        self.second_conv_trans = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 底部路径不使用转置卷积，直接使用普通卷积
        self.bottom_enhance = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 根据图片结构，final_add就是F'_mask，不需要额外的edge_integration
        
        # 最终上采样层：将 128x128 放大到 512x512 (4倍)
        # 使用两次 stride=2 的转置卷积，更稳定
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 边缘细化
        self.edge_refinement = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, feat_mask, f2_feats, f1_feats):
        """
        feat_mask: [N, C, H, W] Fmask - 来自Mask Memory Modeling的时空特征
        f2_feats: [N, C2, H2, W2] F2 - 中分辨率特征
        f1_feats: [N, C1, H1, W1] F1 - 高分辨率特征
        
        返回: [N, 1, H, W] 增强的边缘特征图
        """
        N, _, H, W = feat_mask.shape
        # print(f"=== Edge Generator Debug ===")
        # print(f"输入 feat_mask shape: {feat_mask.shape}")
        # print(f"输入 f2_feats shape: {f2_feats.shape}")
        # print(f"输入 f1_feats shape: {f1_feats.shape}")
        # print(f"原始尺寸 H, W: {H}, {W}")
        
        # 根据架构图实现三条路径
        
        # ===== 顶部路径: Fmask -> Gabor Wavelet Laplacian Filter -> ReLU -> Conv2d Trans =====
        # Step 1: Fmask 特征降维
        base_feature = self.feature_reduction(feat_mask)  # [N, 1, H, W]
        # print(f"顶部路径 base_feature shape: {base_feature.shape}")
        
        # Step 2: 获取Lg(x,y)滤波器核 - 实现公式 (5), (6), (7)
        lg_kernel = self.gabor_wavelet_laplacian.get_filter_kernel()  # [1, 1, 3, 3] - Lg(x,y)
        
        # Step 3: 实现图片结构: ReLU(Fmask) ⊙ WL_output
        # 其中 ⊙ 表示元素级乘法，* 表示卷积操作
        
        # 实现 Lg(x,y) * Fmask 的卷积操作
        # Lg(x,y) * Fmask: 使用Lg(x,y)滤波器对Fmask进行卷积
        # 确保lg_kernel与base_feature的数据类型和设备一致
        lg_kernel = lg_kernel.to(dtype=base_feature.dtype, device=base_feature.device)
        wl_output = F.conv2d(base_feature, lg_kernel, padding=lg_kernel.size(-1)//2)  # [N, 1, H, W] - WL_output
        
        # 实现图片结构: ReLU(Fmask) ⊙ WL_output
        top_relu = F.relu(base_feature)  # ReLU(Fmask)
        top_conv = top_relu * wl_output  # ReLU(Fmask) ⊙ WL_output
        
        # Step 4: Conv2d Trans (stride=2 会将特征图放大约2倍)
        top_conv_trans = self.top_conv_trans(top_conv)  # [N, 1, H*2, W*2] (stride=2 放大2倍)
        # print(f"顶部路径 top_conv_trans shape: {top_conv_trans.shape}")
        
        # ===== 中间路径: Fmask (Up×2) + F2 -> Gabor Wavelet Laplacian Filter -> ReLU -> Conv2d Trans =====
        # Step 1: Fmask 上采样2倍
        fmask_2x = F.interpolate(feat_mask, scale_factor=2, mode='bilinear', align_corners=False)  # [N, C, H*2, W*2]
        fmask_2x_proj = self.feature_reduction(fmask_2x)  # [N, 1, H*2, W*2]
        # print(f"中间路径 fmask_2x shape: {fmask_2x.shape}")
        # print(f"中间路径 fmask_2x_proj shape: {fmask_2x_proj.shape}")
        
        # Step 2: F2 特征处理
        # if f2_feats.shape[-2:] != (H*2, W*2):
        #     f2_feats_resized = F.interpolate(f2_feats, size=(H*2, W*2),mode='bilinear', align_corners=False)
        #     print(f"中间路径 f2_feats_resized shape: {f2_feats_resized.shape}")
        # else:
        #     f2_feats_resized = f2_feats
        f2_feats_resized = f2_feats
        f2_proj = self.hi_res_proj(f2_feats_resized)  # [N, 1, H*2, W*2]
        # print(f"中间路径 f2_proj shape: {f2_proj.shape}")
        # Step 3: Fmask + F2 融合，多余实现
        # mid_fused = fmask_2x_proj + f2_proj  # [N, 1, H*2, W*2]
        # print(f"中间路径 mid_fused shape: {mid_fused.shape}")
        
        # Step 4: 获取Lg(x,y)滤波器核 - 实现公式 (5), (6), (7)
        lg_kernel = self.gabor_wavelet_laplacian.get_filter_kernel()  # [1, 1, 3, 3] - Lg(x,y)
        
        # Step 5: 实现图片结构: ReLU(Fmask+F2) ⊙ WL_output
        # 实现 Lg(x,y) * Fmask 的卷积操作
        # 确保lg_kernel与mid_fused的数据类型和设备一致
        lg_kernel = lg_kernel.to(dtype=fmask_2x_proj.dtype, device=fmask_2x_proj.device)
        wl_output = F.conv2d(fmask_2x_proj, lg_kernel, padding=lg_kernel.size(-1)//2)  # [N, 1, H*2, W*2] - WL_output
        # print(f"中间路径 wl_output shape: {wl_output.shape}")
        # 实现图片结构: ReLU(Fmask+F2) ⊙ WL_output
        mid_relu = F.relu(f2_proj)  # ReLU(Fmask+F2)
        # print(f"中间路径 mid_relu shape: {mid_relu.shape}")
        mid_conv = mid_relu * wl_output  # ReLU(Fmask+F2) ⊙ WL_output
        
        # Step 6: 中间路径直接输出，不经过转置卷积（根据图片结构）
        mid_conv_trans = mid_conv  # [N, 1, H*2, W*2] - 直接输出
        # print(f"中间路径 mid_conv_trans shape: {mid_conv_trans.shape}")
        
        # ===== 底部路径: Fmask (Up×4) + F1 -> Gabor Wavelet Laplacian Filter -> ReLU -> Conv2d Trans =====
        # Step 1: Fmask 上采样4倍
        fmask_4x = F.interpolate(feat_mask, scale_factor=4, mode='bilinear', align_corners=False)  # [N, C, H*4, W*4]
        fmask_4x_proj = self.feature_reduction(fmask_4x)  # [N, 1, H*4, W*4]
        # print(f"底部路径 fmask_4x shape: {fmask_4x.shape}")
        # print(f"底部路径 fmask_4x_proj shape: {fmask_4x_proj.shape}")
        
        # Step 2: F1 特征处理
        # if f1_feats.shape[-2:] != (H*4, W*4):
        #     f1_feats_resized = F.interpolate(f1_feats, size=(H*4, W*4), 
        #                                    mode='bilinear', align_corners=False)
        # else:
        #     f1_feats_resized = f1_feats
        f1_feats_resized = f1_feats
        f1_proj = self.hi_res_proj(f1_feats_resized)  # [N, 1, H*4, W*4]
        # print(f"底部路径 f1_proj shape: {f1_proj.shape}")
        # Step 3: Fmask + F1 融合,多余实现
        # bot_fused = fmask_4x_proj + f1_proj  # [N, 1, H*4, W*4]
        # print(f"底部路径 bot_fused shape: {bot_fused.shape}")
        
        # Step 4: 获取Lg(x,y)滤波器核 - 实现公式 (5), (6), (7)
        lg_kernel = self.gabor_wavelet_laplacian.get_filter_kernel()  # [1, 1, 3, 3] - Lg(x,y)
        
        # Step 5: 实现图片结构: ReLU(Fmask+F1) ⊙ WL_output
        # 实现 Lg(x,y) * Fmask 的卷积操作
        # 确保lg_kernel与bot_fused的数据类型和设备一致
        lg_kernel = lg_kernel.to(dtype=fmask_4x_proj.dtype, device=fmask_4x_proj.device)
        wl_output = F.conv2d(fmask_4x_proj, lg_kernel, padding=lg_kernel.size(-1)//2)  # [N, 1, H*4, W*4] - WL_output
        # print(f"底部路径 wl_output shape: {wl_output.shape}")
        # 实现图片结构: ReLU(Fmask+F1) ⊙ WL_output
        bot_relu = F.relu(f1_proj)  # ReLU(Fmask+F1)
        # print(f"底部路径 bot_relu shape: {bot_relu.shape}")
        bot_conv = bot_relu * wl_output  # ReLU(Fmask+F1) ⊙ WL_output
        
        # Step 6: 底部路径直接输出，不经过任何卷积（根据图片结构）
        bot_conv_trans = bot_conv  # [N, 1, H*4, W*4] - 直接输出
        # print(f"底部路径 bot_conv_trans shape: {bot_conv_trans.shape}")
        
        # ===== 路径融合 =====
        # 按照图片结构进行融合：
        # 1. 顶部转置卷积输出 + 中间转置卷积输出
        # 2. 结果进行第二次转置卷积
        # 3. 第二次转置卷积输出 + 底部元素乘法输出
        
        # Step 1: 顶部转置卷积输出 + 中间路径直接输出
        # 将顶部路径输出上采样到中间路径的尺寸，不需要，直接利用转置卷积。
        # top_resized = F.interpolate(top_conv_trans, size=(H*2, W*2), mode='bilinear', align_corners=False)  # [N, 1, H*2, W*2]
        # 将中间路径输出保持1通道
        mid_conv_trans_1ch = mid_conv_trans  # [N, 1, H*2, W*2]
        # print(f"融合 Step 1 - top_conv_trans shape: {top_conv_trans.shape}")
        # print(f"融合 Step 1 - mid_conv_trans_1ch shape: {mid_conv_trans_1ch.shape}")
        first_add = top_conv_trans + mid_conv_trans_1ch  # [N, 1, H*2, W*2]
        
        # print(f"融合 Step 1 - first_add shape: {first_add.shape}")
        
        # Step 2: 第一次融合结果进行第二次转置卷积
        second_conv_trans = self.second_conv_trans(first_add)  # [N, 1, H*2, W*2]
        # print(f"融合 Step 2 - second_conv_trans shape: {second_conv_trans.shape}")
        
        # Step 3: 第二次转置卷积输出 + 底部元素乘法输出
        # 将第二次转置卷积输出上采样到目标尺寸 (512x512)
        # target_size = (512, 512)  # 目标输出尺寸
        # second_resized = F.interpolate(second_conv_trans, size=target_size, mode='bilinear', align_corners=False)
        # 将底部路径输出保持1通道并上采样到目标尺寸
        bot_conv_trans_1ch = bot_conv_trans  # [N, 1, H*4, W*4]
        # bot_resized = F.interpolate(bot_conv_trans_1ch, size=target_size, mode='bilinear', align_corners=False)
        final_add = second_conv_trans + bot_conv_trans_1ch  # [N, 1, 128, 128]
        # print(f"融合 Step 3 - second_conv_trans shape: {second_conv_trans.shape}")
        # print(f"融合 Step 3 - bot_conv_trans_1ch shape: {bot_conv_trans_1ch.shape}")
        # print(f"融合 Step 3 - final_add shape: {final_add.shape}")
        
        # 应用转置卷积将 128x128 放大到 512x512 (4倍)
        F_prime_mask = self.final_upsample(final_add)  # [N, 1, 512, 512]
        # print(f"最终输出 F_prime_mask shape: {F_prime_mask.shape}")
        # print(f"=== Edge Generator Debug 结束 ===")
        
        return F_prime_mask.clamp(min=-1, max=1)  # 限制在[-1, 1]范围内，更常见的激活范围

# 辅助函数：可视化Gabor滤波器（用于调试）
def visualize_gabor_filters(save_path=None):
    """可视化生成的Gabor滤波器"""
    wavelets = generate_gabor_wavelets(kernel_size=15, num_orientations=8, num_scales=3)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    idx = 0
    for scale in range(3):
        for orient in range(8):
            ax = axes[scale, orient]
            ax.imshow(wavelets[idx], cmap='gray')
            ax.set_title(f'Scale {scale}, Orient {orient*22.5:.0f}°')
            ax.axis('off')
            idx += 1
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # 测试代码
    # print("Testing Wavelet Laplacian Edge Generator...")
    
    # 创建测试数据
    feat_mask = torch.randn(2, 256, 64, 64)
    hi_res_feats = torch.randn(2, 64, 64, 64)
    
    # 创建边缘生成器
    edge_gen = EdgeGenerator(in_ch=256, hi_ch=64)
    
    # 前向传播
    with torch.no_grad():
        edge_output = edge_gen(feat_mask, hi_res_feats)
    
    print(f"Input feature shape: {feat_mask.shape}")
    print(f"Hi-res feature shape: {hi_res_feats.shape}")
    print(f"Output edge shape: {edge_output.shape}")
    print(f"Output range: [{edge_output.min().item():.3f}, {edge_output.max().item():.3f}]")
    
    # 可视化Gabor滤波器
    print("Generating Gabor filter visualization...")
    visualize_gabor_filters("gabor_filters.png")
