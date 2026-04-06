import torch
import torch.nn as nn
import torch.nn.functional as F
from .optical_flow_integration import OpticalFlowEstimator
from .memory_bank import SAM2MemoryBank


# 导入SAM2组件
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam import TwoWayTransformer

class SAM2PointMemoryModeling(nn.Module):
    """
    Point Memory Modeling - 严格按照论文4.2节实现
    
    两个步骤:
    1) 结合光流和区域图补偿相机视点偏移
    2) 与掩码记忆特征交互获得点记忆特征
    """
    
    def __init__(self, feature_dim=256, pwcnet_ckpt=None):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Step 1: 光流估计器 - frozen PWC-Net
        self.optical_flow_estimator = OpticalFlowEstimator(pwcnet_ckpt)
        
        # Step 1: 全局偏移MLP - 将2D偏移转换为64维特征（与memory_dim一致）
        self.global_offset_mlp = nn.Sequential(
            nn.Linear(2, 32),           # 平均偏移2维 -> 32维
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)  # -> 64维特征，与memory_dim一致
        )
        
        # 注意：多尺度特征融合已移除，直接使用f3特征
        
        # Step 2: 点记忆特征聚合 - 聚合点记忆库中的Mp
        # 注意：memory features 使用 64 维，不需要通道适配器
        
        
        # Step 2: 掩码引导修正 - 连接Mp、Õi、Mm得到F̃kref
        # concat flow_point_features (64维) 和 Mm_features (64维) → 128维，然后投影到 64维
        self.memory_proj = nn.Conv2d(128, 64, 1)  # 将 concat 后的 128 维投影到 64 维
        
        # 使用已导入的SAM2组件
        
        # 创建SAM2风格的MemoryAttentionLayer
        memory_layer = MemoryAttentionLayer(
            activation="relu",
            cross_attention=RoPEAttention(
                embedding_dim=feature_dim,
                kv_in_dim=64,  # ✅ key 和 value 的输入维度（memory 的维度）
                num_heads=8,
                downsample_rate=1,
                dropout=0.1,
                rope_theta=10000.0,
                feat_sizes=[64, 64],  # 假设特征图尺寸
                rope_k_repeat=True  # 必须设置为True，因为cross-attention中query和key的序列长度不同
            ),
            d_model=feature_dim,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            self_attention=RoPEAttention(
                embedding_dim=feature_dim,
                num_heads=8,
                downsample_rate=1,
                dropout=0.1,
                rope_theta=10000.0,
                feat_sizes=[64, 64]
            )
        )
        
        # 创建SAM2的MemoryAttention
        self.memory_attention = MemoryAttention(
            d_model=feature_dim,
            pos_enc_at_input=True,
            layer=memory_layer,
            num_layers=4,
            batch_first=True
        )
        
        # 位置编码 - 使用SAM2官方配置
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=feature_dim,  # 256
            temperature=10000,
            normalize=True,
            scale=None
        )
        
        # 时间位置编码（用于对象指针）
        # 根据 SAM2：时间位置编码使用 mem_dim（64）生成，不需要投影（已经是 64 维）
        self.add_tpos_enc_to_obj_ptrs = True  # 是否添加时间位置编码
        
        # SAM2风格的"无memory"embedding（用于第一帧）
        # 参考sam2_base.py的实现
        from torch.nn.init import trunc_normal_
        # no_mem_embed 和 no_mem_pos_enc 应该是 64 维（memory_dim），与 Fk_ref_flat 维度一致
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 64))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, 64))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
    
    def forward(self, feats_seq, frames_seq, prev_mask_feats, prev_point_feats, prev_mask_maps, obj_ptrs_tokens=None, t_diffs=None):
        """
        严格按照论文4.2节实现Point Memory Modeling
        
        Args:
            obj_ptrs_tokens: [T*4, N, 64] 或 None - 对象指针 tokens（来自 Point Memory Bank）
            t_diffs: [T] 或 None - 时间距离列表（用于时间位置编码）
            feats_seq: List[Dict] - 多帧SAM2特征序列 [I_{k-7}, ..., I_k]
            frames_seq: [N, T, 3, H, W] - 输入帧序列
            prev_mask_feats: [N, K, C, H, W] - 掩码记忆特征 {M_q^m}_{q=k-7}^{k-1}
            prev_point_feats: [N, K, C, H, W] - 点记忆特征 {M_q^p}_{q=k-7}^{k-1}
            prev_mask_maps: [N, K, 1, H, W] - 掩码预测图 {M_q}_{q=k-7}^{k-1}
        """
        # print(f"🔍 SAM2PointMemoryModeling 输入特征:")
        # print(f"   feats_seq长度: {len(feats_seq)}")
        # print(f"   frames_seq尺寸: {frames_seq.shape if frames_seq is not None else 'None'}")
        # print(f"   prev_mask_feats尺寸: {prev_mask_feats.shape if prev_mask_feats is not None else 'None'}")
        # print(f"   prev_point_feats尺寸: {prev_point_feats.shape if prev_point_feats is not None else 'None'}")
        # print(f"   prev_mask_maps尺寸: {prev_mask_maps.shape if prev_mask_maps is not None else 'None'}")
        
        N, T, C_img, H_img, W_img = frames_seq.shape
        # 获取当前帧特征 Fk - 直接使用f3特征
        current_features = feats_seq[-1]  # 最后一帧特征
        Fk = current_features["f3"]  # [N, 256, H, W] - 直接使用f3
        
        # === Step 1: 视点偏移补偿 ===
        # T 总是大于 2，所以总是执行光流估计，但只在有足够的 mask maps 时才进行加权
        # print(f"🔍 视点偏移补偿:")
        # 1.1 光流估计: 计算相邻帧对的光流，然后复制到7对
        optical_flows = []
        for i in range(T - 1):  # 实际可用的帧对数
            frame_pair = frames_seq[:, [i, i+1]]  # [I_i, I_{i+1}]
            # 确保输入为float32
            frame_pair = frame_pair.float()
            Oi = self.optical_flow_estimator(frame_pair)  # [N, 2, H, W]
            optical_flows.append(Oi)
        
        # 堆叠实际光流: [N, 7, 2, H, W] 其中 7 = T-1（输入固定8帧）
        optical_flows = torch.stack(optical_flows, dim=1)  # [N, 7, 2, H, W]
        K_flow = 7  # 输入固定8帧，光流总是7个
        
        # 1.2 计算平均视点偏移 (论文公式4)
        # 检查是否有足够的 mask maps（7个）进行加权
        use_mask_weighting = (prev_mask_maps is not None and prev_mask_maps.size(1) >= 7)
        
        if use_mask_weighting:
            # 为每个光流使用对应帧的掩码图 Mi 进行加权
            weighted_flows = []
            for i in range(optical_flows.shape[1]):  # 遍历每个光流
                Oi = optical_flows[:, i]  # [N, 2, H, W] - 第i个光流
                
                # 使用对应帧的掩码图 Mi（确保有7个）
                if i < prev_mask_maps.shape[1]:
                    Mi = prev_mask_maps[:, i]  # [N, 1, H, W] - 第i帧的掩码图
                    # print(f"   使用第{i}帧掩码图: {Mi.shape}")
                else:
                    # 如果掩码图不足，使用最后一帧
                    Mi = prev_mask_maps[:, -1]
                    # print(f"   掩码图不足，使用最后一帧掩码图: {Mi.shape}")
                
                # 调整掩码图尺寸到光流尺寸
                if Mi.shape[-2:] != Oi.shape[-2:]:
                    Mi = F.interpolate(Mi, size=Oi.shape[-2:], mode='bilinear', align_corners=False)
                
                # 计算背景区域权重 (1 - Mi)
                background_weight = (1 - Mi)  # [N, 1, H, W]
                
                # 加权光流 - 不需要unsqueeze，PyTorch会自动广播
                weighted_Oi = Oi * background_weight  # [N, 2, H, W]
                weighted_flows.append(weighted_Oi)
            
            # 堆叠所有加权光流
            weighted_flows = torch.stack(weighted_flows, dim=1)  # [N, K_flow, 2, H, W]
            # 计算每个光流的平均偏移 - 保留K_flow个独立的偏移
            H_flow, W_flow = weighted_flows.shape[-2:]  # 使用weighted_flows的尺寸
            Oi_avg = weighted_flows.sum(dim=(3, 4)) / (H_flow * W_flow)  # [N, K_flow, 2]
        else:
            # 不使用 mask maps 加权，直接计算平均偏移
            # print(f"   不使用 mask maps 加权（prev_mask_maps 为 None 或少于 7 个）")
            H_flow, W_flow = optical_flows.shape[-2:]  # 使用 optical_flows 的尺寸
            Oi_avg = optical_flows.sum(dim=(3, 4)) / (H_flow * W_flow)  # [N, K_flow, 2]
        # print(f"   {K_flow}个光流平均偏移: {Oi_avg.shape}")
        
        # 1.3 通过MLP为每个光流生成特征
        # 输入：K_flow个光流的平均偏移 [N, K_flow, 2] - 每个光流独立处理
        # 输出：K_flow个光流特征 [N, K_flow, 64]
        flow_features = []
        for i in range(K_flow):
            flow_feat = self.global_offset_mlp(Oi_avg[:, i])  # [N, 2] -> [N, 64]
            flow_features.append(flow_feat)
        flow_features = torch.stack(flow_features, dim=1)  # [N, K_flow, 64]
        # print(f"   flow_features ({K_flow}个光流特征): {flow_features.shape}")
        
        # === Step 2: 掩码记忆特征交互 ===
        
        # 2.1 处理点记忆特征 Mp - 保持K个特征（K <= 7）
        # 检查是否有足够的点记忆特征（7个）用于 element-wise 加法
        Mp_features = None
        has_sufficient_point_feats = (prev_point_feats is not None and prev_point_feats.size(1) >= 7)
        
        if has_sufficient_point_feats:
            # 使用所有可用的点记忆特征（按照架构图要求）
            # prev_point_feats: [N, K, C, H, W] - K帧的点记忆特征（K >= 7）
            K = prev_point_feats.size(1)
            # 只使用最新的7个
            if K > 7:
                prev_point_feats = prev_point_feats[:, -7:]
                K = 7
            # print(f"   使用{K}帧点记忆特征（满足条件，将用于 element-wise 加法）")
            
            # 调整尺寸到当前帧特征尺寸
            if prev_point_feats.shape[-2:] != Fk.shape[-2:]:
                # 调整所有帧的尺寸
                adjusted_feats = []
                for t in range(K):
                    feat_t = prev_point_feats[:, t]  # [N, C, H, W]
                    feat_t = F.interpolate(feat_t, size=Fk.shape[-2:], mode='bilinear', align_corners=False)
                    adjusted_feats.append(feat_t)
                prev_point_feats = torch.stack(adjusted_feats, dim=1)  # [N, K, C, H, W]
            
            # 直接使用 64 维的 memory features，不需要通道适配
            # 保持K个特征，不进行聚合
            Mp_features = prev_point_feats  # [N, 7, 64, H, W]
            # print(f"   Mp_features (7帧点记忆特征): {Mp_features.shape}")
        else:
            # 如果没有足够的点记忆特征（None 或少于 7 个），不进行 element-wise 加法
            # 设置为 None，后续不会参与计算
            Mp_features = None
            # print(f"   Mp_features 为 None（prev_point_feats 为 None 或少于 7 个，不进行 element-wise 加法）")
        
        # 2.2 获取掩码记忆特征 Mm - 保持K个特征（K <= 7）
        # 检查是否有足够的掩码记忆特征（7个）用于 concat
        Mm_features = None
        has_sufficient_mask_feats = (prev_mask_feats is not None and prev_mask_feats.size(1) >= 7)
        
        if has_sufficient_mask_feats:
            # 使用所有可用的掩码记忆特征（按照架构图要求）
            # prev_mask_feats: [N, K, C, H, W] - K帧的掩码记忆特征（K >= 7）
            K = prev_mask_feats.size(1)
            # 只使用最新的7个
            if K > 7:
                prev_mask_feats = prev_mask_feats[:, -7:]
                K = 7
            # print(f"   使用{K}帧掩码记忆特征（满足条件，将用于 concat）")
            
            # 调整尺寸到当前帧特征尺寸
            # print(f"🔍 掩码记忆特征是否需要调整:")
            # print(f"   prev_mask_feats shape: {prev_mask_feats.shape}")
            # print(f"   Fk shape: {Fk.shape}")
            if prev_mask_feats.shape[-2:] != Fk.shape[-2:]:
                
                # 调整所有帧的尺寸
                adjusted_feats = []
                for t in range(K):
                    feat_t = prev_mask_feats[:, t]  # [N, C, H, W]
                    feat_t = F.interpolate(feat_t, size=Fk.shape[-2:], mode='bilinear', align_corners=False)
                    adjusted_feats.append(feat_t)
                prev_mask_feats = torch.stack(adjusted_feats, dim=1)  # [N, K, C, H, W]
            
            # 保持K个特征，不进行聚合
            Mm_features = prev_mask_feats  # [N, 7, C, H, W]
            # print(f"   Mm_features (7帧掩码记忆特征): {Mm_features.shape}")
        else:
            # 如果没有足够的掩码记忆特征（None 或少于 7 个），不进行 concat
            # 设置为 None，后续不会参与计算
            Mm_features = None
            # print(f"   Mm_features 为 None（prev_mask_feats 为 None 或少于 7 个，不进行 concat）")
        
        # 2.3 构建掩码引导的修正点特征 F̃kref - flow_features 总是有值（T > 2）
        # 如果 Mp_features 有值，进行 element-wise 加法；否则只用 flow_features
        # 如果 Mm_features 有值，进行 concat；否则不 concat
        # flow_features 总是 7 帧
        K = 7
        K_flow = flow_features.shape[1]
        assert K_flow == 7, f"flow_features 应该有 7 帧，但得到 {K_flow}"
        
        # 将光流特征扩展到空间维度
        # flow_features: [N, 7, 64] -> [N, 7, 64, H, W]
        flow_features_spatial = flow_features.unsqueeze(-1).unsqueeze(-1)  # [N, 7, 64, 1, 1]
        flow_features_spatial = flow_features_spatial.expand(-1, -1, -1, Fk.shape[2], Fk.shape[3])  # [N, 7, 64, H, W]
        
        # 步骤1: 如果有 Mp_features，进行 element-wise 加法；否则只用 flow_features
        if Mp_features is not None:
            assert Mp_features.shape[1] == 7, f"Mp_features 应该有 7 帧，但得到 {Mp_features.shape[1]}"
            # 光流MLP后特征 + 点memory bank特征 → add (保持7个特征，都是64维)
            flow_point_features = Mp_features + flow_features_spatial  # [N, 7, 64, H, W]
            # print(f"   flow_point_features (光流特征 + 点记忆特征): {flow_point_features.shape}")
        else:
            # 没有点记忆特征，只用光流特征
            flow_point_features = flow_features_spatial  # [N, 7, 64, H, W]
            # print(f"   flow_point_features (只用光流特征，未加点记忆特征): {flow_point_features.shape}")
        
        # 步骤2: 如果有 Mm_features，进行 concat；否则直接使用 flow_point_features
        # concat 后投影到 64 维，Fk_ref_features 始终是 64 维
        if Mm_features is not None:
            assert Mm_features.shape[1] == 7, f"Mm_features 应该有 7 帧，但得到 {Mm_features.shape[1]}"
            # concat flow_point_features (64维) 和 Mm_features (64维) → 128维
            # print(f"flow_point_features shape: {flow_point_features.shape}")
            # print(f"Mm_features shape: {Mm_features.shape}")
            combined_features = torch.cat([flow_point_features, Mm_features], dim=2)  # [N, 7, 128, H, W]
            # 投影到 64 维
            N, T, C, H, W = combined_features.shape
            combined_features_reshaped = combined_features.reshape(N * T, C, H, W)  # [N*7, 128, H, W]
            Fk_ref_features_proj = self.memory_proj(combined_features_reshaped)  # [N*7, 64, H, W]
            Fk_ref_features = Fk_ref_features_proj.reshape(N, T, 64, H, W)  # [N, 7, 64, H, W]
        else:
            # 没有掩码记忆特征，直接使用 flow_point_features（64 通道）
            Fk_ref_features = flow_point_features  # [N, 7, 64, H, W]
            # print(f"Fk_ref_features shape: {Fk_ref_features.shape}")
        # 2.4 使用SAM2的MemoryAttention进行self-attention和cross-attention
        H, W = Fk.shape[-2:]
        Fk_flat = Fk.flatten(2).transpose(1, 2)  # [N, H*W, 256]
        Fk_pos = self.position_encoding(Fk).flatten(2).transpose(1, 2)  # [N, H*W, 256]
        
        # 准备记忆特征 - 使用K个特征（K <= 7）
        if Fk_ref_features is not None:
            # 将K个特征展平并连接
            # Fk_ref_features: [N, K, 64, H, W] → [N, K*H*W, 64]
            N, T, _, H, W = Fk_ref_features.shape
            assert Fk_ref_features.shape[2] == 64, f"Fk_ref_features 应该是 64 维，但得到 {Fk_ref_features.shape[2]}"
            Fk_ref_flat = Fk_ref_features.permute(0, 1, 3, 4, 2).reshape(N, T*H*W, 64)  # [N, K*H*W, 64]
            # print(f"Fk_ref_flat shape: {Fk_ref_flat.shape}")
            
            # 为所有K帧计算位置编码（64 维）
            if not hasattr(self, 'memory_pos_enc_64'):
                from sam2.modeling.position_encoding import PositionEmbeddingSine
                self.memory_pos_enc_64 = PositionEmbeddingSine(
                    num_pos_feats=64,
                    temperature=10000,
                    normalize=True,
                    scale=None
                ).to(device=Fk_ref_features.device, dtype=Fk_ref_features.dtype)
            
            Fk_ref_pos_list = []
            for t in range(T):
                feat_t = Fk_ref_features[:, t]  # [N, 64, H, W]
                pos_t = self.memory_pos_enc_64(feat_t).flatten(2).transpose(1, 2)  # [N, H*W, 64]
                Fk_ref_pos_list.append(pos_t)
            Fk_ref_pos = torch.cat(Fk_ref_pos_list, dim=1)  # [N, K*H*W, 64]
        else:
            # 如果没有Fk_ref_features（即Mp_features、Mm_features或flow_features为None）
            # 使用1个可学习的dummy token作为memory（参考SAM2的实现）
            # 一直有，不会创建no_mem_embed和no_mem_pos_enc
            # print("########################################")
            N = Fk_flat.shape[0]
            Fk_ref_flat = self.no_mem_embed.expand(N, 1, 64)  # [N, 1, 64]
            Fk_ref_pos = self.no_mem_pos_enc.expand(N, 1, 64)  # [N, 1, 64]
        
        # 注意：SAM2的MemoryAttention支持不同序列长度和不同维度的curr和memory
        # curr (Fk_flat): [N, H*W, 256] - 当前帧特征（序列长度 H*W，维度 256）
        # memory (Fk_ref_flat): [N, K*H*W, 64] - K帧历史特征（序列长度 K*H*W，维度 64，K <= 7）
        # 这是正常的，Cross-Attention机制支持query和key/value的序列长度和维度不同
        # 不需要进行线性映射对齐，直接使用即可（符合SAM2的设计）
        
        # 处理对象指针（如果提供，与Mask分支一致）
        # 根据 SAM2：对象指针保持 64 维（mem_dim），不转换为 256 维
        num_obj_ptr_tokens = 0
        if obj_ptrs_tokens is not None:
            # 转换格式: [T*4, N, 64] -> [N, T*4, 64]
            # 注意：对象指针保持 64 维，不转换为 256 维（与 SAM2 一致）
            obj_ptrs_tokens = obj_ptrs_tokens.transpose(0, 1)  # [N, T*4, 64]
            
            # 添加到 memory 末尾
            # Fk_ref_flat 是 64 维，obj_ptrs_tokens 也是 64 维，直接 concat
            Fk_ref_flat = torch.cat([Fk_ref_flat, obj_ptrs_tokens], dim=1)  # [N, K*H*W + T*4, 64]
            
            # 为对象指针创建位置编码（时间位置编码）
            # 根据 SAM2：使用 mem_dim（64）生成时间位置编码
            if self.add_tpos_enc_to_obj_ptrs and t_diffs is not None:
                # 计算时间位置编码
                # t_diffs: [T] - 每个对象指针与当前帧的时间距离
                T = len(t_diffs)  # 历史帧数
                num_tokens_per_ptr = obj_ptrs_tokens.shape[1] // T  # 每个对象指针的token数（通常是4）
                
                # 归一化时间距离
                t_diff_max = max(1.0, float(t_diffs.max().item()))  # 避免除零
                t_diffs_norm = t_diffs / t_diff_max  # [T]
                
                # 生成1D正弦位置编码 - 使用 mem_dim（64）生成（与 SAM2 一致）
                from sam2.modeling.sam2_utils import get_1d_sine_pe
                mem_dim = 64  # memory_dim
                obj_pos_1d = get_1d_sine_pe(t_diffs_norm, dim=mem_dim, temperature=10000)  # [T, 64]
                
                # 扩展到每个分割的token
                obj_pos_1d = obj_pos_1d.unsqueeze(1).expand(-1, num_tokens_per_ptr, -1)  # [T, num_tokens_per_ptr, 64]
                obj_ptrs_pos = obj_pos_1d.reshape(-1, mem_dim).unsqueeze(0).expand(N, -1, -1)  # [N, T*num_tokens_per_ptr, 64]
            else:
                # 全零位置编码（不使用时间位置编码）
                obj_ptrs_pos = torch.zeros_like(obj_ptrs_tokens)  # [N, T*4, 64]
            
            Fk_ref_pos = torch.cat([Fk_ref_pos, obj_ptrs_pos], dim=1)  # [N, K*H*W + T*4, 64]
            
            num_obj_ptr_tokens = obj_ptrs_tokens.shape[1]  # T*4
        
        # SAM2的MemoryAttention在batch_first=True时，期望输入格式是[SeqLen, Batch, C]（序列优先）
        # 然后内部会转换为[Batch, SeqLen, C]进行处理
        # 我们需要将当前格式[N, SeqLen, C]转换为[SeqLen, N, C]
        # 注意：curr和memory的序列长度和维度可以不同，这是SAM2的正常设计
        # - curr (query): [H*W, N, 256] - 当前帧特征，序列长度 H*W，维度 256
        # - memory (key/value): [K*H*W + T*4, N, 64] - K帧历史特征 + 对象指针，序列长度 K*H*W + T*4，维度 64
        # Cross-Attention机制支持query和key/value的序列长度和维度不同，不需要对齐
        Fk_flat_seq_first = Fk_flat.transpose(0, 1)  # [H*W, N, 256]
        Fk_ref_flat_seq_first = Fk_ref_flat.transpose(0, 1)  # [K*H*W + T*4, N, 64]
        Fk_pos_seq_first = Fk_pos.transpose(0, 1)  # [H*W, N, 256]
        Fk_ref_pos_seq_first = Fk_ref_pos.transpose(0, 1)  # [K*H*W + T*4, N, 64]
        # print(f"Fk_flat_seq_first shape: {Fk_flat_seq_first.shape}")
        # print(f"Fk_ref_flat_seq_first shape: {Fk_ref_flat_seq_first.shape}")
        # print(f"Fk_pos_seq_first shape: {Fk_pos_seq_first.shape}")
        # print(f"Fk_ref_pos_seq_first shape: {Fk_ref_pos_seq_first.shape}")
        Fpoint_flat_seq_first = self.memory_attention(
            curr=Fk_flat_seq_first,           # [H*W, N, 256] 当前特征作为query
            memory=Fk_ref_flat_seq_first,     # [K*H*W + T*4, N, 64] Fk_ref_features + 对象指针作为key/value
            curr_pos=Fk_pos_seq_first,        # [H*W, N, 256] 当前特征位置编码
            memory_pos=Fk_ref_pos_seq_first,  # [K*H*W + T*4, N, 64] 记忆特征位置编码
            num_obj_ptr_tokens=num_obj_ptr_tokens  # ✅ 传递对象指针数量，排除 RoPE（与Mask分支一致）
        )  # 返回 [H*W, N, 256]
        
        # 转换回batch优先格式
        Fpoint_flat = Fpoint_flat_seq_first.transpose(0, 1)  # [N, H*W, 256]
        
        # 重塑回空间维度
        # 注意：MemoryAttention 输出维度与 curr 相同（256 维）
        Fpoint = Fpoint_flat.transpose(1, 2).reshape(N, -1, H, W)  # [N, 256, H, W]
        # print(f"   Fpoint (点特征): {Fpoint.shape}")
        # 构建输出
        outputs = {
            'point_features': Fpoint,           # F_point - 记忆增强的点特征
            'flow_features': flow_features,     # [N, 7, 256] - 7个光流特征
            'optical_flows': optical_flows,     # [N, 7, 2, H, W] - 7个光流场
            'mask_guided_ref': Fk_ref_features  # [N, 7, 256, H, W] - 7帧掩码引导修正特征
        }
        
        #打印SAM2PointMemoryModeling输出特征
        # print(f"🔍 SAM2PointMemoryModeling 输出特征:")
        # print(f"   point_features (F_point): {outputs['point_features'].shape}")
        # print(f"   flow_features (7个光流特征): {outputs['flow_features'].shape if outputs['flow_features'] is not None else 'None'}")
        # print(f"   optical_flows (7个光流场): {outputs['optical_flows'].shape if outputs['optical_flows'] is not None else 'None'}")
        # print(f"   mask_guided_ref (7帧F̃kref): {outputs['mask_guided_ref'].shape if outputs['mask_guided_ref'] is not None else 'None'}")
        
        return outputs


class PointDecoder(nn.Module):
    """
    Point Decoder - 基于SAM2架构实现
    使用SAM2的TwoWayTransformer和learnable tokens，通过self-attention和cross-attention + MLP预测点坐标和置信度
    """
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 使用已导入的SAM2 TwoWayTransformer
        self.transformer = TwoWayTransformer(
            depth=num_layers,
            embedding_dim=feature_dim,
            mlp_dim=feature_dim * 4,
            num_heads=num_heads,
        )
        
        # 使用SAM2官方位置编码
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=feature_dim,
            temperature=10000,
            normalize=True,
            scale=None
        )
        
        # Learnable tokens - 按SAM2设计
        # 4个输出token：坐标、分数、记忆特征、对象指针（与Mask Decoder一致）
        self.output_tokens = nn.Parameter(torch.randn(1, 4, feature_dim))  # 4个输出token
        self.prompt_tokens = nn.Parameter(torch.randn(1, 2, feature_dim))  # 2个提示token
        
        # 最终预测头 - MLP layers
        self.point_coord_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 2)  # x, y坐标
        )
        
        self.point_score_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1)  # 置信度分数
        )
    
    
    def forward(self, point_features):
        """
        point_features: [N, 256, H, W] - 来自Point Memory Modeling的F_point
        """
        # print(f"🔍 PointDecoder 输入特征:")
        # print(f"   point_features (F_point): {point_features.shape}")
        
        N, C, H, W = point_features.shape
        
        # 准备tokens
        output_tokens = self.output_tokens.expand(N, -1, -1)  # [N, 4, 256]
        prompt_tokens = self.prompt_tokens.expand(N, -1, -1)  # [N, 2, 256]
        tokens = torch.cat([output_tokens, prompt_tokens], dim=1)  # [N, 6, 256]
        
        # 使用SAM2的TwoWayTransformer
        # 直接使用原始特征图，与SAM2官方实现保持一致
        feature_map = point_features  # [N, 256, H, W]
        B, C, H, W = feature_map.shape
        
        # 使用SAM2官方位置编码
        feature_pe = self.position_embedding(feature_map)  # [N, 256, H, W]
        
        # 使用TwoWayTransformer - 注意参数顺序和返回值
        tokens, _ = self.transformer(
            image_embedding=feature_map,  # [N, 256, H, W]
            image_pe=feature_pe,          # [N, 256, H, W]
            point_embedding=tokens        # [N, 6, 256]
        )
        
        # 提取预测tokens (前4个是输出tokens)
        pred_tokens = tokens[:, :4]  # [N, 4, 256]
        
        # 预测坐标和置信度
        coords_token = pred_tokens[:, 0]  # 第1个token用于坐标
        score_token = pred_tokens[:, 1]   # 第2个token用于置信度
        mem_feat_token = pred_tokens[:, 2]  # 第3个token用于记忆特征
        object_pointer_token = pred_tokens[:, 3]  # 第4个token用于对象指针（与Mask Decoder一致）
        
        # MLP预测
        point_coords = self.point_coord_head(coords_token)  # [N, 2]
        point_score = self.point_score_head(score_token)    # [N, 1]
        
        # 对象指针：直接使用token，格式与Mask Decoder一致 [N, 1, 256]
        object_pointer = object_pointer_token.unsqueeze(1)  # [N, 1, 256]
        
        # 构建输出
        outputs = {
            'coords': point_coords,      # [N, 2] 点坐标
            'score': point_score,        # [N, 1] 置信度分数
            'point_mem_feat': mem_feat_token.unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W),  # [N, 256, H, W] 记忆特征
            'object_pointer': object_pointer  # [N, 1, 256] 对象指针（与Mask Decoder一致）
        }
        
        # 打印PointDecoder输出特征
        # print(f"🔍 PointDecoder 输出特征:")
        # print(f"   coords (点坐标): {outputs['coords'].shape,outputs['coords']}")
        # print(f"   score (置信度): {outputs['score'].shape,outputs['score']}")
        # print(f"   point_mem_feat (记忆特征): {outputs['point_mem_feat'].shape}")
        
        return outputs


class PointBranch(nn.Module):
    """
    出血点分支 - 严格按照论文架构实现
    
    论文架构三个主要部分:
    1. Point Memory Modeling - 嵌入光流估计预测位移场
    2. Point Decoder - 使用SAM2架构或自定义实现预测点坐标和置信度
    3. Point Memory Bank - 存储点记忆增强时序建模
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. Point Memory Modeling - 嵌入光流估计
        self.point_memory_modeling = SAM2PointMemoryModeling(
            feature_dim=256,
            pwcnet_ckpt=getattr(cfg.model, 'pwcnet_ckpt', None)
        )
        
        # 2. Point Decoder - SAM2风格的解码器
        self.point_decoder = PointDecoder(
            feature_dim=256,
            num_heads=8,
            num_layers=2
        )
        
        # 3. Point Memory Bank - 在分支内部管理
        self.point_memory_bank = SAM2MemoryBank(
            max_len=getattr(cfg.model, 'point_memory_len', 7),
            feature_dim=getattr(cfg.model, 'feature_dim', 256),
            memory_dim=getattr(cfg.model, 'memory_dim', 64),
            image_size=getattr(cfg.model, 'image_size', 1024),
            backbone_stride=getattr(cfg.model, 'backbone_stride', 16)
        )
        
        # 4. Mask Memory Bank 引用 - 用于跨分支历史信息获取
        self.mask_memory_bank = None  # 将在运行时设置
        

    def _generate_point_map(self, coords, image_size):
        """
        生成点图用于记忆库存储
        
        Args:
            coords: [N, 2] 点坐标
            image_size: (H, W) 图像尺寸
        """
        N = coords.shape[0]
        H, W = image_size
        
        # 创建点图
        point_map = torch.zeros(N, 1, H, W, device=coords.device)
        
        for i in range(N):
            x, y = coords[i]
            
            # 确保坐标在有效范围内
            x = torch.clamp(x, 0, W-1)
            y = torch.clamp(y, 0, H-1)
            
            # 转换为整数坐标
            x_int = int(x.round().item())
            y_int = int(y.round().item())
            
            # 在点位置设置高斯分布
            sigma = 3.0
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=coords.device, dtype=torch.float32),
                torch.arange(W, device=coords.device, dtype=torch.float32),
                indexing='ij'
            )
            
            gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            point_map[i, 0] = gaussian
        
        return point_map

    def forward(self, feats_seq, prev_mask_maps, prev_mask_feats, frames_seq=None, prev_point_feats=None):
        """
        严格按照论文架构的Point Branch前向传播
        
        Args:
            feats_seq: List[Dict] - SAM2编码器特征序列 [I_{k-7}, ..., I_k]
            prev_mask_maps: [N, K, 1, H, W] - 前一帧掩码预测 {M_q}_{q=k-7}^{k-1}
            prev_mask_feats: [N, K, C, H, W] - 前一帧掩码记忆特征 {M_q^m}_{q=k-7}^{k-1}
            frames_seq: [N, T, 3, H, W] - 输入帧序列
            prev_point_feats: [N, K, C, H, W] - 前一帧点记忆特征 {M_q^p}_{q=k-7}^{k-1}
        
        Returns:
            Dict containing:
                - coords: [N, 2] 出血点坐标
                - score: [N, 1] 置信度分数  
                - point_mem_feat: [N, C, H, W] 点记忆特征
                - point_map: [N, 1, H, W] 点图用于记忆库
                - global_offset: [N, 2] 全局偏移坐标 Õi
        """
       
        # Step 1: 从Point Memory Bank获取历史点记忆特征和对象指针
        # 根据论文架构图，Point Memory Bank 存储特征和对象指针（与Mask分支一致）
        current_device = feats_seq[0]['f3'].device if feats_seq else torch.device('cpu')
        # 假设当前帧索引（可以从外部传入，这里使用相对索引）
        current_frame_idx = len(feats_seq) - 1 if feats_seq else None
        _, point_memory_feats, point_obj_ptrs_tokens, point_t_diffs = self.point_memory_bank.fetch(
            device=current_device,
            include_obj_ptrs=True,  # ✅ 获取对象指针（与Mask分支一致）
            current_frame_idx=current_frame_idx  # ✅ 传递当前帧索引用于计算时间距离
        )
        
        # 如果Point Memory Bank有数据，使用Memory Bank的记忆
        if point_memory_feats is not None:
            prev_point_feats = point_memory_feats  # 使用Point Memory Bank的记忆
            # print(f"prev_point_feats (Point Memory Bank有数据): {prev_point_feats.shape}")
        # 否则使用传入的prev_point_feats（第一帧或外部提供）
        
        # Step 1.5: 从Mask Memory Bank获取历史掩码信息（跨分支历史信息）
        if self.mask_memory_bank is not None:
            try:
                # 确保在正确的设备上获取数据
                mask_memory_maps, mask_memory_feats, _, _ = self.mask_memory_bank.fetch(
                    device=current_device
                    # 注意：Point分支使用自己的对象指针，不从Mask Memory Bank获取
                )
                
                if mask_memory_maps is not None:
                    prev_mask_maps = mask_memory_maps
                    # print(f"prev_mask_maps (从Mask Memory Bank获取): {prev_mask_maps.shape}, device: {prev_mask_maps.device}")
                if mask_memory_feats is not None:
                    prev_mask_feats = mask_memory_feats
                    # print(f"prev_mask_feats (从Mask Memory Bank获取): {prev_mask_feats.shape}, device: {prev_mask_feats.device}")
                # print(f"🔍 点分支输入特征:")
                # print(f"   feats_seq 8个特征序列，每个f3特征序列shape: {feats_seq[0]['f3'].shape if feats_seq is not None else 'None'}")
                # print(f"   frames_seq尺寸: {frames_seq.shape if frames_seq is not None else 'None'}")
                # print(f"   prev_mask_maps尺寸: {prev_mask_maps.shape if prev_mask_maps is not None else 'None'}")
                # print(f"  point branch prev_mask_feats尺寸: {prev_mask_feats.shape if prev_mask_feats is not None else 'None'}")
                # print(f"   prev_point_feats尺寸: {prev_point_feats.shape if prev_point_feats is not None else 'None'}")
            except Exception as e:
                # print(f"❌ 从Mask Memory Bank获取历史信息失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ Mask Memory Bank 未设置，无法获取历史掩码信息")
        
        # Step 2: Point Memory Modeling (论文4.2节)
        # 嵌入光流估计预测位移场，整合掩码记忆特征
        # ✅ 使用Point Memory Bank的对象指针（与Mask分支一致）
        memory_outputs = self.point_memory_modeling(
            feats_seq=feats_seq,
            frames_seq=frames_seq,
            prev_mask_feats=prev_mask_feats,
            prev_point_feats=prev_point_feats,
            prev_mask_maps=prev_mask_maps,
            obj_ptrs_tokens=point_obj_ptrs_tokens,  # ✅ 使用Point Memory Bank的对象指针
            t_diffs=point_t_diffs  # ✅ 传递时间距离（用于时间位置编码）
        )
        
        # Step 3: Point Decoder (论文描述)
        # 使用自定义Point Decoder实现预测点坐标和置信度
        decoder_outputs = self.point_decoder(
            point_features=memory_outputs['point_features']  # Fpoint
        )
        
        # Step 4: 构建最终输出
        # 生成点图用于Point Memory Bank存储
        # 使用 resize 后的图像尺寸 512x512
        coords = decoder_outputs['coords']  # [N, 2]
        point_map = self._generate_point_map(
            coords=coords,
            image_size=(512, 512)  # resize 后的尺寸
        )
        
        # Step 5: 更新Point Memory Bank (按照架构图)
        # 根据论文架构图，Point Encoding 只使用 point_map，不需要 F_k 特征
        # 同时存储对象指针（与Mask分支一致）
        object_pointer = decoder_outputs.get('object_pointer')  # [N, 1, 256]
        # print(f"🔧 准备调用update_point: point_map.shape={point_map.shape}")
        self.point_memory_bank.update_point(
            point_map,
            object_pointer=object_pointer  # ✅ 存储对象指针（与Mask分支一致）
        )
        # print("✅ update_point调用完成")
        
        # 整合所有输出 - 只保留外部需要的输出
        # 根据 blood_det.py 的 _build_output 和 mask_branch.py 的使用情况：
        # - coords: 必需（blood_det.py 使用）
        # - score: 必需（blood_det.py 使用）
        # - point_map: 必需（blood_det.py 和 mask_branch.py 使用）
        # 以下输出只在内部使用，不需要输出：
        # - point_mem_feat: 只在 Point Decoder 内部使用
        # - object_pointer: 只在 Memory Bank 内部使用
        # - flow_features: 只在 Point Memory Modeling 内部使用
        # - flows: 只在 Point Memory Modeling 内部使用
        # - mask_guided_ref: 只在 Point Memory Modeling 内部使用
        outputs = {
            'coords': decoder_outputs['coords'],    # [N, 2] 点坐标
            'score': decoder_outputs['score'],      # [N, 1] 置信度分数
            'point_map': point_map,                 # [N, 1, H, W] 点图（用于 mask_branch 提示编码）
        }
        #打印输出值
        # print(f"coords: {coords}")
        # print(f"score: {decoder_outputs['score']}")
        # print(f"point_map最大最小和均值: {point_map.max().item():.6f}, {point_map.min().item():.6f}, {point_map.mean().item():.6f}")
        
        return outputs

