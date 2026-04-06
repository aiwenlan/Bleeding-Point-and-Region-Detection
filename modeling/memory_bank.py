"""
记忆库 - 基于SAM2的Memory机制实现
使用SAM2的MemoryEncoder和MemoryAttention进行高级记忆管理
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

# 导入SAM2的Memory相关组件
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine


class SAM2MemoryBank(nn.Module):
    """
    基于SAM2的Memory机制的记忆库
    
    功能:
    1. 使用SAM2的MemoryEncoder进行高级记忆编码
    2. 使用SAM2的MemoryAttention进行记忆注意力机制
    3. 支持时序记忆管理和特征融合
    4. 完全兼容SAM2架构
    """
    
    def __init__(self, max_len=7, feature_dim=256, memory_dim=64, 
                 image_size=512, backbone_stride=16):  # 论文使用 512x512
        super().__init__()
        self.max_len = max_len
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        
        # SAM2的MemoryEncoder组件
        self.mask_downsampler = MaskDownSampler(
            embed_dim=feature_dim,  # 使用feature_dim而不是memory_dim
            kernel_size=3,
            stride=2,
            padding=1,
            total_stride=backbone_stride,
            activation=nn.GELU
        )
        
        self.fuser = Fuser(
            layer=CXBlock(
                dim=feature_dim,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True
            ),
            num_layers=2
        )
        
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=memory_dim,
            normalize=True,
            scale=None,
            temperature=10000
        )
        
        # SAM2的MemoryEncoder
        self.memory_encoder = MemoryEncoder(
            out_dim=memory_dim,
            mask_downsampler=self.mask_downsampler,
            fuser=self.fuser,
            position_encoding=self.position_encoding,
            in_dim=feature_dim
        )
        
        
        # 记忆存储
        self.memory_maps = []      # 存储mask/point maps
        self.memory_features = []  # 存储编码后的memory features
        self.object_pointers = []  # 存储对象指针 [N, 1, 256]
        self.frame_indices = []    # 存储帧索引（用于时间位置编码）
        # 对于 Point Memory Bank，额外存储 point_map（用于 mask_branch 获取）
        self.point_maps = []       # 存储 point_map [N, 1, H, W]
        
        # 双队列设计（可选）：区分近 N 帧记忆和被提示帧记忆
        self.recent_memory_maps = []      # 近 N 帧记忆（带时间编码）
        self.recent_memory_features = []  # 近 N 帧记忆特征
        self.recent_object_pointers = []  # 近 N 帧对象指针
        self.recent_frame_indices = []    # 近 N 帧索引
        
        self.prompted_memory_maps = []      # 被提示帧记忆（无时间编码）
        self.prompted_memory_features = []  # 被提示帧记忆特征
        self.prompted_object_pointers = []  # 被提示帧对象指针
        self.prompted_frame_indices = []    # 被提示帧索引
    
    
    def forward(self, x):
        """为了兼容nn.Module，但实际不使用"""
        return x
    
    def encode_memory(self, pixel_features: torch.Tensor, masks: torch.Tensor, skip_mask_sigmoid: bool = False) -> Dict[str, torch.Tensor]:
        """
        使用SAM2的MemoryEncoder编码记忆
        
        Args:
            pixel_features: [N, C, H, W] 像素特征
            masks: [N, 1, H, W] 掩码/点图（概率值，已应用 sigmoid）
            skip_mask_sigmoid: 是否跳过 sigmoid（如果 masks 已经是概率值，应该为 True）
            
        Returns:
            Dict containing:
                - vision_features: [N, memory_dim, H', W'] 编码后的记忆特征
                - vision_pos_enc: [N, memory_dim, H', W'] 位置编码
        """
        return self.memory_encoder(pixel_features, masks, skip_mask_sigmoid=skip_mask_sigmoid)
    
    
    def update_mask(self, features: torch.Tensor, maps: torch.Tensor, 
                    object_pointer=None, occlusion_score=None, is_prompted=False, frame_idx=None):
        """
        更新Mask Memory Bank
        
        Args:
            features: [N, C, H, W] 图像特征（应该是 F_k，不是 Fmask）
            maps: [N, 1, H, W] 掩码图（概率值，已经应用过 sigmoid）
            object_pointer: [N, 1, 256] 对象指针（可选）
            occlusion_score: [N, 1] 遮挡分数（可选）
            is_prompted: bool 是否为被提示帧（可选，用于双队列设计）
            frame_idx: int 帧索引（可选，用于时间位置编码）
        """
        # 调用 memory_encoder，传入 skip_mask_sigmoid=True
        # 因为 maps 已经在 mask_branch 中应用了 sigmoid
        memory_output = self.encode_memory(features, maps, skip_mask_sigmoid=True)
        encoded_features = memory_output['vision_features']
        
        # 存储记忆（保持在同一设备上）
        self.memory_maps.append(maps.detach())
        self.memory_features.append(encoded_features.detach())
        
        # 存储对象指针（如果提供）
        if object_pointer is not None:
            self.object_pointers.append(object_pointer.detach())
        else:
            self.object_pointers.append(None)
        
        # 存储帧索引（用于时间位置编码）
        if frame_idx is not None:
            self.frame_indices.append(frame_idx)
        else:
            # 如果没有提供，使用当前列表长度作为相对索引
            self.frame_indices.append(len(self.memory_maps))
        
        # 双队列设计：区分近 N 帧记忆和被提示帧记忆
        if is_prompted:
            # 被提示帧记忆（无时间编码）
            self.prompted_memory_maps.append(maps.detach())
            self.prompted_memory_features.append(encoded_features.detach())
            if object_pointer is not None:
                self.prompted_object_pointers.append(object_pointer.detach())
            else:
                self.prompted_object_pointers.append(None)
            if frame_idx is not None:
                self.prompted_frame_indices.append(frame_idx)
            else:
                self.prompted_frame_indices.append(len(self.prompted_memory_maps) - 1)
            
            # 维护被提示帧队列大小（通常可以保留更多帧）
            if len(self.prompted_memory_maps) > self.max_len * 2:  # 提示帧可以保留更多
                self.prompted_memory_maps.pop(0)
                self.prompted_memory_features.pop(0)
                self.prompted_object_pointers.pop(0)
                if len(self.prompted_frame_indices) > 0:
                    self.prompted_frame_indices.pop(0)
        else:
            # 近 N 帧记忆（带时间编码）
            self.recent_memory_maps.append(maps.detach())
            self.recent_memory_features.append(encoded_features.detach())
            if object_pointer is not None:
                self.recent_object_pointers.append(object_pointer.detach())
            else:
                self.recent_object_pointers.append(None)
            if frame_idx is not None:
                self.recent_frame_indices.append(frame_idx)
            else:
                self.recent_frame_indices.append(len(self.recent_memory_maps) - 1)
            
            # 维护近 N 帧队列大小
            if len(self.recent_memory_maps) > self.max_len:
                self.recent_memory_maps.pop(0)
                self.recent_memory_features.pop(0)
                self.recent_object_pointers.pop(0)
                if len(self.recent_frame_indices) > 0:
                    self.recent_frame_indices.pop(0)
        
        # 维护统一队列大小（向后兼容）
        if len(self.memory_maps) > self.max_len:
            self.memory_maps.pop(0)
            self.memory_features.pop(0)
            if len(self.object_pointers) > 0:
                self.object_pointers.pop(0)
            if len(self.frame_indices) > 0:
                self.frame_indices.pop(0)
    
    def update_point(self, maps: torch.Tensor, object_pointer=None, frame_idx=None):
        """
        更新Point Memory Bank
        
        根据论文架构图，Point Encoding 只使用 point_map，不需要 F_k 特征
        
        Args:
            maps: [N, 1, H, W] 点图（概率值，已应用 sigmoid）
            object_pointer: [N, 1, 256] 对象指针（可选，与Mask分支一致）
            frame_idx: int 帧索引（可选，用于时间位置编码）
        """
        # print("🚀 update_point被调用了！")
        
        # 根据论文架构图，Point Encoding 只使用 point_map
        # 但 SAM2 的 MemoryEncoder 需要 pix_feat 和 masks 两个输入
        # 因此创建一个全零的虚拟 pix_feat（只使用 point_map 的信息）
        N, _, H, W = maps.shape
        # 计算下采样后的尺寸 (total_stride=16)
        downsampled_H = H // self.backbone_stride
        downsampled_W = W // self.backbone_stride
        
        # 创建全零的虚拟 pix_feat（因为 Point Encoding 只使用 point_map）
        dummy_pix_feat = torch.zeros(
            N, self.feature_dim, downsampled_H, downsampled_W,
            device=maps.device, dtype=maps.dtype
        )
        
        try:
            # print("🔄 开始调用encode_memory...")
            # point_map 已经使用 sigmoid 生成概率值，需要跳过 sigmoid
            # 使用虚拟的 pix_feat（全零），实际编码只使用 point_map
            memory_output = self.encode_memory(dummy_pix_feat, maps, skip_mask_sigmoid=True)
            # print("✅ encode_memory调用成功")
            encoded_features = memory_output['vision_features']
            # print(f"✅ 获取encoded_features: {encoded_features.shape}")
        except Exception as e:
            # print(f"❌ encode_memory调用失败: {e}")
            import traceback
            traceback.print_exc()
            return  # 如果编码失败，直接返回
        
        # 存储编码后的特征（保持在同一设备上）
        self.memory_features.append(encoded_features.detach())
        
        # 存储 point_map（用于 mask_branch 获取）
        self.point_maps.append(maps.detach())
        
        # 存储对象指针（如果提供，与Mask分支一致）
        if object_pointer is not None:
            self.object_pointers.append(object_pointer.detach())
        else:
            self.object_pointers.append(None)
        
        # 存储帧索引（用于时间位置编码）
        if frame_idx is not None:
            self.frame_indices.append(frame_idx)
        else:
            # 如果没有提供，使用当前列表长度作为相对索引
            self.frame_indices.append(len(self.memory_features))
        
        # print(f"✅ update_point完成: memory_features长度={len(self.memory_features)}")
        
        # 维护记忆库大小
        if len(self.memory_features) > self.max_len:
            self.memory_features.pop(0)
            if len(self.point_maps) > 0:
                self.point_maps.pop(0)
            if len(self.object_pointers) > 0:
                self.object_pointers.pop(0)
            if len(self.frame_indices) > 0:
                self.frame_indices.pop(0)
            # print(f"🔄 维护记忆库大小: memory_features长度={len(self.memory_features)}, point_maps长度={len(self.point_maps)}")
    
    def fetch(self, device=None, include_obj_ptrs=False, use_dual_queue=False, current_frame_idx=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取记忆
        
        Args:
            device: 目标设备，如果为None则保持原设备
            include_obj_ptrs: 是否包含对象指针（分割成 tokens）
            use_dual_queue: 是否使用双队列设计（合并近 N 帧和被提示帧）
            current_frame_idx: int 当前帧索引（可选，用于计算时间距离）
        
        Returns:
            maps: [N, T, 1, H, W] 或 None
            features: [N, T, C, H, W] 或 None
            obj_ptrs_tokens: [T*4, N, 64] 或 None（如果 include_obj_ptrs=True，对象指针分割成 4 个 64 维 tokens）
            t_diffs: [T] 或 None - 时间距离列表（用于时间位置编码）
        """
        if use_dual_queue:
            # 使用双队列设计：合并近 N 帧和被提示帧
            all_maps = self.recent_memory_maps + self.prompted_memory_maps
            all_features = self.recent_memory_features + self.prompted_memory_features
            all_obj_ptrs = self.recent_object_pointers + self.prompted_object_pointers
            all_frame_indices = self.recent_frame_indices + self.prompted_frame_indices
        else:
            # 使用统一队列（向后兼容）
            all_maps = self.memory_maps
            all_features = self.memory_features
            all_obj_ptrs = self.object_pointers
            all_frame_indices = self.frame_indices
        
        # 对于 Point Memory Bank，如果没有 features，返回 None
        # 注意：Point Memory Bank 可能不存储 maps，所以检查 features 而不是 maps
        if len(all_features) == 0:
            return None, None, None, None
        
        # 论文要求：记忆固定为7帧
        # - 不足7帧：不填充，直接用可用的帧数（变长）
        # - 刚好7帧：用7帧
        # - 超过7帧：用最新的7帧
        K = len(all_features)
        if K > self.max_len:
            # 超过7帧，只取最新的7帧
            all_features = all_features[-self.max_len:]
            if len(all_maps) > 0:
                all_maps = all_maps[-self.max_len:]
            if len(all_obj_ptrs) > 0:
                all_obj_ptrs = all_obj_ptrs[-self.max_len:]
            if len(all_frame_indices) > 0:
                all_frame_indices = all_frame_indices[-self.max_len:]
            K = self.max_len
        
        # 计算时间距离（用于时间位置编码）
        t_diffs = None
        if current_frame_idx is not None and len(all_frame_indices) > 0:
            # 计算每个历史帧与当前帧的时间距离
            t_diffs = [abs(current_frame_idx - idx) for idx in all_frame_indices]
            t_diffs = torch.tensor(t_diffs, dtype=torch.float32)
            if device is not None:
                t_diffs = t_diffs.to(device)
        elif len(all_frame_indices) > 0:
            # 如果没有提供当前帧索引，使用相对距离（从新到旧：0, 1, 2, ...）
            t_diffs = torch.arange(len(all_frame_indices), dtype=torch.float32)
            if device is not None:
                t_diffs = t_diffs.to(device)
        
        # 将记忆转换为tensor（K可能是1-7之间的任意值）
        features_tensor = torch.stack(all_features, dim=1)  # [N, K, C, H, W]，K <= 7
        
        # 处理 maps：Point Memory Bank 可能不存储 maps
        if len(all_maps) > 0:
            maps_tensor = torch.stack(all_maps, dim=1)  # [N, T, 1, H, W]
        else:
            maps_tensor = None  # Point Memory Bank 不存储 maps
        
        # 如果需要，移动到指定设备
        if device is not None:
            features_tensor = features_tensor.to(device)
            if maps_tensor is not None:
                maps_tensor = maps_tensor.to(device)
        
        # 处理对象指针（如果请求）
        obj_ptrs_tokens = None
        if include_obj_ptrs and len(all_obj_ptrs) > 0:
            # 过滤 None
            valid_ptrs = [ptr for ptr in all_obj_ptrs if ptr is not None]
            if len(valid_ptrs) > 0:
                obj_ptrs = torch.stack(valid_ptrs, dim=0)  # [T, N, 1, 256]
                obj_ptrs = obj_ptrs.squeeze(2)  # [T, N, 256]
                
                # 分割成 4 个 64 维 tokens（根据 SAM2 的设计）
                T, N, C = obj_ptrs.shape  # C = 256
                mem_dim = self.memory_dim  # 64
                num_tokens = C // mem_dim  # 4
                
                obj_ptrs_tokens = obj_ptrs.reshape(T, N, num_tokens, mem_dim)  # [T, N, 4, 64]
                obj_ptrs_tokens = obj_ptrs_tokens.permute(0, 2, 1, 3)  # [T, 4, N, 64]
                obj_ptrs_tokens = obj_ptrs_tokens.flatten(0, 1)  # [T*4, N, 64]
                
                if device is not None:
                    obj_ptrs_tokens = obj_ptrs_tokens.to(device)
        
        return maps_tensor, features_tensor, obj_ptrs_tokens, t_diffs
    
    def fetch_point_map(self, device=None):
        """
        获取最新的 point_map（用于 mask_branch）
        
        Args:
            device: 目标设备，如果为None则保持原设备
        
        Returns:
            point_map: [N, 1, H, W] 或 None - 最新的 point_map
        """
        if len(self.point_maps) == 0:
            return None
        
        # 获取最新的 point_map
        point_map = self.point_maps[-1]  # [N, 1, H, W]
        
        if device is not None:
            point_map = point_map.to(device)
        
        return point_map
    
    def clear(self):
        """清空记忆库"""
        self.memory_maps.clear()
        self.memory_features.clear()
        self.object_pointers.clear()
        self.frame_indices.clear()
        
        # 清空双队列
        self.recent_memory_maps.clear()
        self.recent_memory_features.clear()
        self.recent_object_pointers.clear()
        self.recent_frame_indices.clear()
        self.prompted_memory_maps.clear()
        self.prompted_memory_features.clear()
        self.prompted_object_pointers.clear()
        self.prompted_frame_indices.clear()
        # 清空 point_maps
        self.point_maps.clear()