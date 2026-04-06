"""
出血区域分支 - 基于论文架构实现
实现Mask Memory Modeling和Edge-guided区域分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .edge_generator import EdgeGenerator
from .prompt_encoder import PromptEncoder
from .sam2_wrapper import SAM2Backbone
from .memory_bank import SAM2MemoryBank

# 导入SAM2组件
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.position_encoding import PositionEmbeddingSine


class SAM2MaskMemoryModeling(nn.Module):
    """
    Mask Memory Modeling - 使用SAM2原生MemoryAttention
    
    功能:
    1. 使用SAM2的MemoryAttention进行self-attention和cross-attention
    2. 产生时空特征Fmask
    3. 完全基于SAM2的Memory机制
    """
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, image_size=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        # 使用已导入的SAM2组件
        
        # 创建SAM2风格的MemoryAttentionLayer
        memory_layer = MemoryAttentionLayer(
            activation="relu",
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[64, 64],  # 假设特征图尺寸
                kv_in_dim=64,  # ✅ key 和 value 的输入维度（memory 的维度）
                embedding_dim=feature_dim,
                num_heads=num_heads,
                downsample_rate=1,
                dropout=0.1,
                rope_k_repeat=True  # 必须设置为True，因为cross-attention中query和key的序列长度不同
            ),
            d_model=feature_dim,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[64, 64],
                embedding_dim=feature_dim,
                num_heads=num_heads,
                downsample_rate=1,
                dropout=0.1
            )
        )
        
        # 创建SAM2的MemoryAttention
        self.memory_attention = MemoryAttention(
            d_model=feature_dim,
            pos_enc_at_input=True,
            layer=memory_layer,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 位置编码
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=feature_dim,  # 256
            temperature=10000,
            normalize=True,
            scale=None
        )
        
        # 时间位置编码（用于对象指针）
        self.add_tpos_enc_to_obj_ptrs = True  # 是否添加时间位置编码
        
        # SAM2风格的"无memory"embedding（用于第一帧）
        # 参考sam2_base.py的实现
        # no_mem_embed 和 no_mem_pos_enc 应该是 64 维（memory_dim），与 memory_feat 维度一致
        from torch.nn.init import trunc_normal_
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 64))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, 64))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        
        # 为 memory 特征创建 64 维的位置编码器（与 point 分支一致）
        self.memory_pos_enc_64 = PositionEmbeddingSine(
            num_pos_feats=64,
            temperature=10000,
            normalize=True,
            scale=None
        )
        # self.memory_channel_adapter = nn.Conv2d(64, 256, 1)  # 64 -> 256
        self.to_memory_dim = nn.Linear(256, 64)
    
    def forward(self, current_feat, prev_mask_feats, obj_ptrs_tokens=None, t_diffs=None):
        """
        Args:
            current_feat: [N, C, H, W] - 当前帧特征
            prev_mask_feats: [N, K, C, H, W] - 前K帧mask memory features
            obj_ptrs_tokens: [T*4, N, 64] 或 None - 对象指针 tokens（来自 Mask Memory Bank）
            t_diffs: [T] 或 None - 时间距离列表（用于时间位置编码）
        Returns:
            Fmask: [N, C, H, W] - 时空特征
        """
        N, C, H, W = current_feat.shape
        # print(f"############current_feat shape: {current_feat.shape}")
        
        # 准备当前帧特征和位置编码（没问题）
        current_feat_flat = current_feat.flatten(2).transpose(1, 2)  # [N, H*W, C]
        current_pos = self.position_encoding(current_feat).flatten(2).transpose(1, 2)  # [N, H*W, C]
        
        # 准备记忆特征
        # 只有当有 7 个或更多的 mask memory features 时才使用真实记忆特征
        # 否则使用 SAM2 风格的"无memory"embedding（与Point分支一致）
        K = 0  # 初始化K
        has_sufficient_mask_feats = (prev_mask_feats is not None and prev_mask_feats.size(1) >= 7)
        
        if has_sufficient_mask_feats:
            # 只使用最新的 7 个
            if prev_mask_feats.size(1) > 7:
                prev_mask_feats = prev_mask_feats[:, -7:]
            # prev_mask_feats 现在有 7 个特征，形状是 [N, 7, 64, H, W]
            # 直接使用 64 维，不需要通道适配,需要转为256维
            N_actual, K, C_mem, H_actual, W_actual = prev_mask_feats.shape
            # print(f"############N_actual shape: {N_actual}")
            # print(f"############K shape: {K}")
            # print(f"############C_mem shape: {C_mem}")
            # print(f"############H_actual shape: {H_actual}")
            # print(f"############W_actual shape: {W_actual}")
            assert K == 7, f"prev_mask_feats 应该有 7 帧，但得到 {K}"
            
            # 展平为序列格式（使用 64 维）
            # print(f"############prev_mask_feats shape: {prev_mask_feats.shape}")
            memory_feat = prev_mask_feats.contiguous().view(N, K, C_mem, H_actual, W_actual)
            memory_feat = memory_feat.contiguous().view(N, K*H_actual*W_actual, C_mem)  # [N, 7*H*W, 64]
            
            # 为记忆特征生成位置编码 - 为每一帧生成独立的位置编码
            # 符合SAM2定义：每一帧应该有独立的空间位置编码
            # 注意：位置编码的维度需要与 memory_feat 的维度匹配（64 维）
            memory_pos_list = []
            for k in range(K):
                # 为每一帧生成独立的空间位置编码（使用 64 维位置编码器）
                feat_k = prev_mask_feats[:, k]  # [N, 64, H, W]
                pos_k = self.memory_pos_enc_64(feat_k)  # [N, 64, H, W]
                pos_k = pos_k.flatten(2).transpose(1, 2)  # [N, H*W, 64]
                memory_pos_list.append(pos_k)
            # 拼接所有帧的位置编码
            memory_pos = torch.cat(memory_pos_list, dim=1)  # [N, K*H*W, 64]
        else:
            # SAM2方式：如果没有历史特征，使用1个可学习的dummy token（参考sam2_base.py）
            # 使用no_mem_embed作为memory，而不是复制当前特征
            # no_mem_embed 和 no_mem_pos_enc 是 64 维，与 memory_feat 维度一致
            K = 7  # 设置为7，表示复制到7帧
            # print(f"current_feat_flat shape: {current_feat_flat.shape}")
            # print(f"current_pos shape: {current_pos.shape}")
            current_feat_flat_mem = self.to_memory_dim(current_feat_flat)
            current_pos_mem = self.to_memory_dim(current_pos)
            memory_feat = current_feat_flat_mem.unsqueeze(1).repeat(1, 7, 1, 1)  # [N, 7, H*W, C]
            memory_feat = memory_feat.view(N, 7*H*W, 64)  # [N, 7*H*W, C]
            # current_pos: [N, H*W, C] -> [N, 7, H*W, C] -> [N, 7*H*W, C]
            memory_pos = current_pos_mem.unsqueeze(1).repeat(1, 7, 1, 1)  # [N, 7, H*W, C]
            memory_pos = memory_pos.view(N, 7*H*W, 64)  # [N, 7*H*W, C]
            # print(f"no_mem_embed shape: {self.no_mem_embed.shape}")
            # print(f"no_mem_pos_enc shape: {self.no_mem_pos_enc.shape}")
            # memory_feat = self.no_mem_embed.expand(N, 1, 64)  # [N, 1, 64]
            # memory_pos = self.no_mem_pos_enc.expand(N, 1, 64)  # [N, 1, 64]
        
        # 调试信息
        # print(f"Memory attention inputs:")
        # print(f"  current_feat_flat shape: {current_feat_flat.shape}")
        # print(f"  memory_feat shape: {memory_feat.shape}")
        # print(f"  current_pos shape: {current_pos.shape}")
        # print(f"  memory_pos shape: {memory_pos.shape}")
        
        # 注意：MemoryAttention支持不同序列长度和维度的curr和memory
        # curr: [N, H*W, 256] - 当前帧特征，维度 256
        # memory: [N, K*H*W + T*4, 64] - 多帧记忆特征 + 对象指针，序列长度可以不同，维度 64
        # 不需要压缩memory的序列长度，也不需要维度对齐，这是SAM2的正常设计
        
        # 确保所有输入的数据类型与 SAM2 模型一致
        # 获取 SAM2 模型的数据类型
        sam2_dtype = next(self.memory_attention.parameters()).dtype
        # print(f"SAM2 model dtype: {sam2_dtype}")
        
        # 处理对象指针（如果提供，与Point分支一致）
        # 根据 SAM2：对象指针保持 64 维（mem_dim），不转换为 256 维
        num_obj_ptr_tokens = 0
        if obj_ptrs_tokens is not None:
            # 转换格式: [T*4, N, 64] -> [N, T*4, 64]
            # 注意：对象指针保持 64 维，不转换为 256 维（与 SAM2 一致）
            obj_ptrs_tokens = obj_ptrs_tokens.transpose(0, 1)  # [N, T*4, 64]
            # if not hasattr(self, 'obj_ptr_channel_adapter'):
            #     self.obj_ptr_channel_adapter = nn.Linear(64, C).to(
            #         device=obj_ptrs_tokens.device, dtype=obj_ptrs_tokens.dtype
            #     )
            # obj_ptrs_tokens = self.obj_ptr_channel_adapter(obj_ptrs_tokens)  # [N, T*4, C]
            
            # 添加到 memory 末尾
            # 注意：memory_feat 现在是 64 维，obj_ptrs_tokens 也是 64 维，维度匹配
            # print(f"memory_feat shape: {memory_feat.shape}")
            # print(f"obj_ptrs_tokens shape: {obj_ptrs_tokens.shape}")
            memory_feat = torch.cat([memory_feat, obj_ptrs_tokens], dim=1)  # [N, K*H*W + T*4, 64]
            
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
                # 不需要投影，已经是 64 维
                
                # 扩展到每个分割的token
                obj_pos_1d = obj_pos_1d.unsqueeze(1).expand(-1, num_tokens_per_ptr, -1)  # [T, num_tokens_per_ptr, 64]
                obj_ptrs_pos = obj_pos_1d.reshape(-1, mem_dim).unsqueeze(0).expand(N, -1, -1)  # [N, T*num_tokens_per_ptr, 64]
            else:
                # 全零位置编码（不使用时间位置编码）
                obj_ptrs_pos = torch.zeros_like(obj_ptrs_tokens)  # [N, T*4, 64]
            # print(f"memory_pos shape: {memory_pos.shape}")
            # print(f"obj_ptrs_pos shape: {obj_ptrs_pos.shape}")
            memory_pos = torch.cat([memory_pos, obj_ptrs_pos], dim=1)  # [N, K*H*W + T*4, 64]
            
            num_obj_ptr_tokens = obj_ptrs_tokens.shape[1]  # T*4
        
        # 将所有输入转换为 SAM2 模型的数据类型
        current_feat_flat = current_feat_flat.to(dtype=sam2_dtype)
        memory_feat = memory_feat.to(dtype=sam2_dtype)
        current_pos = current_pos.to(dtype=sam2_dtype)
        memory_pos = memory_pos.to(dtype=sam2_dtype)
        
        # 调试信息：打印 batch size 和形状
        # print(f"🔍 SAM2MaskMemoryModeling Debug Info:")
        # print(f"  current_feat shape: {current_feat.shape}")
        # print(f"  current_feat_flat shape: {current_feat_flat.shape}")
        # print(f"  memory_feat shape: {memory_feat.shape}")
        # print(f"  current_pos shape: {current_pos.shape}")
        # print(f"  memory_pos shape: {memory_pos.shape}")
        # print(f"  N (from current_feat): {N}")
        # print(f"  curr batch size (current_feat_flat.shape[0]): {current_feat_flat.shape[0]}")
        # print(f"  memory batch size (memory_feat.shape[0]): {memory_feat.shape[0]}")
        # if prev_mask_feats is not None:
        #     print(f"  prev_mask_feats shape: {prev_mask_feats.shape}")
        # if obj_ptrs_tokens is not None:
        #     # obj_ptrs_tokens 已经在上面处理过了，所以这里可以直接访问
        #     try:
        #         print(f"  obj_ptrs_tokens shape (after processing): {obj_ptrs_tokens.shape}")
        #     except:
        #         print(f"  obj_ptrs_tokens: provided but shape unavailable")
        # print(f"  num_obj_ptr_tokens: {num_obj_ptr_tokens}")
        
        # 注意：batch size 已经在前面修复过了（第114-123行和第213-221行）
        # current_feat_flat 和 memory_feat 的 batch size 应该都是 N，不需要再次检查
        
        # SAM2的MemoryAttention在batch_first=True时，期望输入格式是[SeqLen, Batch, C]（序列优先）
        # 然后内部会转换为[Batch, SeqLen, C]进行处理
        # 我们需要将当前格式[N, SeqLen, C]转换为[SeqLen, N, C]
        # 注意：curr 和 memory 的维度可以不同
        # - curr: [H*W, N, 256] - 当前帧特征，维度 256
        # - memory: [K*H*W + T*4, N, 64] - 记忆特征 + 对象指针，维度 64
        current_feat_seq_first = current_feat_flat.transpose(0, 1)  # [H*W, N, 256]
        # memory_feat可能是[N, K*H*W, 64]（有memory时）或[N, 1, 64]（无memory时，使用no_mem_embed）
        memory_feat_seq_first = memory_feat.transpose(0, 1)  # [K*H*W + T*4, N, 64] 或 [1, N, 64]
        current_pos_seq_first = current_pos.transpose(0, 1)  # [H*W, N, 256]
        # memory_pos可能是[N, K*H*W, 64]（有memory时）或[N, 1, 64]（无memory时）
        memory_pos_seq_first = memory_pos.transpose(0, 1)  # [K*H*W + T*4, N, 64] 或 [1, N, 64]
        # print(f"current_feat_seq_first shape: {current_feat_seq_first.shape}")
        # print(f"memory_feat_seq_first shape: {memory_feat_seq_first.shape}")
        # print(f"current_pos_seq_first shape: {current_pos_seq_first.shape}")
        # print(f"memory_pos_seq_first shape: {memory_pos_seq_first.shape}")
        # print(f"num_obj_ptr_tokens: {num_obj_ptr_tokens}")
        # 使用SAM2的MemoryAttention
        Fmask_flat_seq_first = self.memory_attention(
            curr=current_feat_seq_first,  # [H*W, N, 256] 当前特征作为query
            memory=memory_feat_seq_first,  # [K*H*W + T*4, N, 64] 记忆特征 + 对象指针作为key/value
            curr_pos=current_pos_seq_first,  # [H*W, N, 256] 当前特征位置编码
            memory_pos=memory_pos_seq_first,  # [K*H*W + T*4, N, 64] 记忆特征位置编码
            num_obj_ptr_tokens=num_obj_ptr_tokens  # ✅ 传递对象指针数量，排除 RoPE（与Point分支一致）
        )  # 返回 [H*W, N, 256]（输出维度与 curr 相同）
        
        # 转换回batch优先格式
        Fmask_flat = Fmask_flat_seq_first.transpose(0, 1)  # [N, H*W, C]
        
        # 转换回空间格式
        Fmask = Fmask_flat.transpose(1, 2).reshape(N, C, H, W)
        
        return Fmask



class MaskBranch(nn.Module):
    """
    出血区域分支 - 基于论文架构实现
    
    功能:
    1. Mask Memory Modeling - 与previous frames交互产生Fmask
    2. Edge Generator - 基于Fmask生成增强边缘
    3. Adaptive Prompt Embeddings - 结合edge map和point map
    4. SAM2解码器集成
    5. 多层级输出
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Mask Memory Modeling - 使用SAM2原生MemoryAttention
        self.mask_memory_modeling = SAM2MaskMemoryModeling(
            feature_dim=256,
            num_heads=8,
            num_layers=4,
            image_size=getattr(cfg.model, 'image_size', 512)  # 论文使用 512x512
        )
        
        # Mask Memory Bank - 在分支内部管理
        self.mask_memory_bank = SAM2MemoryBank(
            max_len=getattr(cfg.model, 'mask_memory_len', 7),
            feature_dim=getattr(cfg.model, 'feature_dim', 256),
            memory_dim=getattr(cfg.model, 'memory_dim', 64),
            image_size=getattr(cfg.model, 'image_size', 512),  # 论文使用 512x512
            backbone_stride=getattr(cfg.model, 'backbone_stride', 16)
        )
        
        # Point Memory Bank 引用（将在运行时设置，用于获取 point_map）
        self.point_memory_bank = None
        
        # 边缘生成器 (Wavelet Laplacian Filter) - 输入Fmask
        self.edge_generator = EdgeGenerator(in_ch=256)
        
        # 提示编码器 - 使用论文实现
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            pos_embed_dim=128
        )
        
        # 注意：多尺度特征融合已移除，使用SAM2原生output_upscaling
        
        
        # 注意：掩码细化和多层级预测已移除，使用SAM2原生MaskDecoder
        
        # SAM2解码器 - 由分支自己管理
        self.sam2_decoder = SAM2Backbone(
            sam2_config=getattr(cfg.model, 'sam2_config', 'sam2_hiera_b+.yaml'),
            ckpt_path=getattr(cfg.model, 'sam2_ckpt', None)
        )
        
        # 自适应权重
        self.fusion_weights = nn.Parameter(torch.ones(3))  # edge, point, feature
        self.scale_weights = nn.Parameter(torch.ones(3))   # multi-scale weights

    def _generate_mask_map(self, sam_mask, image_size):
        """
        生成掩码图用于记忆库存储
        类似点分支的_generate_point_map，但用于掩码
        
        Args:
            sam_mask: [N, 1, H, W] SAM2原始掩码输出
            image_size: (H, W) 目标图像尺寸
        """
        N = sam_mask.shape[0]
        H, W = image_size
        
        # 如果sam_mask已经是目标尺寸，直接返回
        if sam_mask.shape[-2:] == (H, W):
            return sam_mask
        
        # 调整到目标尺寸
        mask_map = F.interpolate(sam_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # 可选：对掩码进行后处理，如阈值化
        # mask_map = torch.sigmoid(mask_map)  # 如果原始输出是logits
        # mask_map = (mask_map > 0.5).float()  # 二值化
        
        return mask_map

    def forward(self, feats_seq, point_map=None, prev_mask_feats=None):
        """
        完整的出血区域分支前向传播
        
        Args:
            feats_seq: List[Dict] 多帧特征序列 Fk-N, ..., Fk
            point_map: [N, 1, H, W] 点分支输出的点图（可选，如果为None则使用edge作为point_map）
            prev_mask_feats: [N, T, C, H, W] 前T帧的mask memory features（可选，通常从Memory Bank获取）
        
        Returns:
            dict: 包含mask_map, mask_mem_feat, edge_features等
        """
        # Step 0: 从Mask Memory Bank获取历史记忆和对象指针
        # fetch() 返回: (maps, features, obj_ptrs_tokens, t_diffs)
        # ✅ 获取对象指针和时间距离（与Point分支一致）
        current_device = feats_seq[-1]["f3"].device if feats_seq else torch.device('cpu')
        # 假设当前帧索引（可以从外部传入，这里使用相对索引）
        current_frame_idx = len(feats_seq) - 1 if feats_seq else None
        mask_memory_maps, mask_memory_feats, mask_obj_ptrs_tokens, mask_t_diffs = self.mask_memory_bank.fetch(
            device=current_device,
            include_obj_ptrs=True,  # ✅ 获取对象指针（与Point分支一致）
            current_frame_idx=current_frame_idx  # ✅ 传递当前帧索引用于计算时间距离
        )
        # 直接传递 mask_memory_feats，判断逻辑放在 SAM2MaskMemoryModeling 中（与Point分支一致）
        prev_mask_feats = mask_memory_feats
        
        # Step 1: 获取当前帧特征
        current_features = feats_seq[-1]  # dict: f1, f2, f3
        
        # Step 2: 直接使用f3特征 - 替代多尺度融合
        multiscale_feat = current_features["f3"]  # [N, 256, H, W]
        
        # Step 3: Mask Memory Modeling - 使用SAM2原生MemoryAttention
        # 与previous frames的mask memory features进行self-attention和cross-attention
        # ✅ 使用Mask Memory Bank的对象指针（与Point分支一致）
        # 如果没有历史记忆，mask_memory_modeling 内部会使用 SAM2 风格的"无memory"embedding
        
        Fmask = self.mask_memory_modeling(
            current_feat=multiscale_feat,  # 当前帧特征
            prev_mask_feats=prev_mask_feats,  # 前T帧mask memory features（可能为None）
            obj_ptrs_tokens=mask_obj_ptrs_tokens,  # ✅ 传递对象指针（与Point分支一致）
            t_diffs=mask_t_diffs  # ✅ 传递时间距离（用于时间位置编码）
        )  # [N, 256, H, W] - 时空特征Fmask
        
        # Step 4: Edge Generator - 基于Fmask生成增强边缘
        # 论文: "引入采用了multi-scale Wavelet Laplacian filters的边缘生成器得到对于Fmask的增强的edge"
        # 架构图显示Edge Generator接收Fmask和F1, F2多尺度特征
        edge_features = self.edge_generator(
            Fmask,  # Fmask - 来自Mask Memory Modeling的时空特征
            current_features["f2"],  # F2 - 中分辨率特征
            current_features["f1"]   # F1 - 高分辨率特征
        )  # [N, 1, H, W] - 增强的边缘特征Em
        
        # Step 5: 处理 point_map - 优先从 point_memory_bank 获取，否则使用传入的或 edge
        if point_map is None:
            # 尝试从 point_memory_bank 获取最新的 point_map
            if self.point_memory_bank is not None:
                point_map = self.point_memory_bank.fetch_point_map(device=current_device)
            
            # 如果仍然为 None，使用 edge 作为 point_map
            if point_map is None:
                point_map = edge_features  # [N, 1, H, W]
        
        # Step 6: Adaptive Prompt Embeddings - 论文核心
        # "we form adaptive prompt embeddings by combining the edge map Em from the edge
        # generator with the point map Pm output from point branch"
        edge_tokens, point_tokens = self.prompt_encoder(edge_features, point_map)
        
        # Step 7: SAM2解码 - 使用分支自己的SAM2解码器
        # "We input Fmask into the mask decoder similar to SAM 2"
        # 根据架构图，Mask Decoder应该接收Fmask、edge_tokens、point_tokens，以及F1和F2作为high_res_features
        sam_mask = None
        iou_predictions = None
        object_pointer = None
        occlusion_score = None
        try:
            # 使用SAM2的mask_decode方法
            # 注意：应该传递Fmask而不是原始特征，以及F1和F2用于high_res_features
            sam_output = self.sam2_decoder.mask_decode(
                Fmask, edge_tokens, point_tokens,
                f1=current_features["f1"],  # F1 - 高分辨率特征（符合架构图）
                f2=current_features["f2"]   # F2 - 中分辨率特征（符合架构图）
            )
            # 提取所有输出
            if isinstance(sam_output, dict):
                sam_mask = sam_output.get('masks')
                iou_predictions = sam_output.get('iou_predictions')  # [N, 1] - IoU 预测分数
                object_pointer = sam_output.get('object_pointer')  # [N, 1, 256] - 对象指针
                occlusion_score = sam_output.get('occlusion_score')  # [N, 1] - 遮挡分数
            else:
                sam_mask = sam_output
        except Exception as e:
            print(f"Warning: SAM2 decode failed: {e}")
        
        # Step 8: 将SAM2原始输出上采样到高分辨率
        # SAM2 mask decoder 输出低分辨率 logits，需要上采样到论文的输入尺寸
        image_size = getattr(self.cfg.model, 'image_size', 512)  # 论文使用 512x512
        if sam_mask is not None:
            # 使用双线性插值上采样到 image_size（论文使用 512x512）
            high_res_mask = F.interpolate(
                sam_mask,  # logits [N, 1, H_low, W_low]
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            )  # [N, 1, image_size, image_size]
        else:
            print("SAM2 解码失败")
            # 如果 SAM2 解码失败，创建默认掩码
            high_res_mask = torch.zeros(
                1, 1, image_size, image_size,
                device=Fmask.device, dtype=Fmask.dtype
            )
        
        # Step 9: 生成mask_map用于记忆库存储
        # 类似点分支的_generate_point_map，生成用于记忆库的mask map
        mask_map = self._generate_mask_map(high_res_mask, image_size=(image_size, image_size))
        
        # Step 10: 更新Mask Memory Bank
        # 根据论文架构图，Memory Encoding 使用 F_k（原始图像特征），而不是 F_mask
        current_f3 = current_features["f3"]  # F_k - [N, 256, H, W] 图像编码器原始特征
        self.mask_memory_bank.update_mask(
            current_f3,  # F_k，不是 Fmask
            mask_map,    # 概率值掩码（已应用 sigmoid）
            object_pointer=object_pointer,
            occlusion_score=occlusion_score
        )
        
        # 构建输出
        output = {
            "mask_map": mask_map,                      # [N, 1, image_size, image_size] 最终掩码（概率值）
            "edge_features": edge_features,            # [N, 1, H, W] 边缘特征Em
            "sam_mask": sam_mask,                     # [N, 1, H, W] SAM2预测（logits）
            "iou_predictions": iou_predictions,        # [N, 1] IoU 预测分数
            "object_pointer": object_pointer,         # [N, 1, 256] 对象指针（用于 Memory Bank）
            "occlusion_score": occlusion_score        # [N, 1] 遮挡分数（用于遮挡判断）
        }
        # print(f"🔍 掩码分支输出特征:")
        # print(f"   mask_map (最终掩码): {output['mask_map'].shape}")
        # print(f"   Fmask (时空特征): {Fmask.shape}")
        # print(f"   edge_features (边缘特征): {output['edge_features'].shape}")
        # print(f"   point_tokens (点tokens): {point_tokens.shape}")
        # print(f"   edge_tokens (边缘tokens): {edge_tokens.shape}")
        # print(f"   sam_mask (SAM2预测): {output['sam_mask'].shape if output['sam_mask'] is not None else 'None'}")
        # print("最小最大值和均值:")
        # print(f"   mask_map (最终掩码): {output['mask_map'].min().item():.6f}, {output['mask_map'].max().item():.6f}, {output['mask_map'].mean().item():.6f}")
        # print(f"   Fmask (时空特征): {Fmask.min().item():.6f}, {Fmask.max().item():.6f}, {Fmask.mean().item():.6f}")
        # print(f"   edge_features (边缘特征): {output['edge_features'].min().item():.6f}, {output['edge_features'].max().item():.6f}, {output['edge_features'].mean().item():.6f}")
        # print(f"   point_tokens (点tokens): {point_tokens.min().item():.6f}, {point_tokens.max().item():.6f}, {point_tokens.mean().item():.6f}")
        # print(f"   edge_tokens (边缘tokens): {edge_tokens.min().item():.6f}, {edge_tokens.max().item():.6f}, {edge_tokens.mean().item():.6f}")
        # print(f"   sam_mask (SAM2预测): {output['sam_mask'].min().item():.6f}, {output['sam_mask'].max().item():.6f}, {output['sam_mask'].mean().item():.6f}")
        
        return output



