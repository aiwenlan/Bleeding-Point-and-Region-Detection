"""
论文 Prompt Encoder 实现
实现公式 (1), (2), (3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sam2.modeling.sam2_utils import LayerNorm2d

class PromptEncoder(nn.Module):
    """
    论文 Prompt Encoder 实现
    实现公式 (1), (2), (3)
    """
    
    def __init__(self, embed_dim=256, pos_embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        
        # 公式 (2): Ep = Conv(LN(G(Conv(LN(G(Conv(Em)))))))
        # 使用LayerNorm2d，符合SAM2标准
        self.edge_conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2)  # 2x2 conv
        self.edge_gelu1 = nn.GELU()
        self.edge_ln1 = LayerNorm2d(64)  # 2D LayerNorm，对通道维度归一化
        
        self.edge_conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)  # 2x2 conv
        self.edge_gelu2 = nn.GELU()
        self.edge_ln2 = LayerNorm2d(128)  # 2D LayerNorm，对通道维度归一化
        
        self.edge_conv3 = nn.Conv2d(128, embed_dim, kernel_size=2, stride=2)  # 2x2 conv
        
        # 公式 (3): Pp = C[sin(2π(Po(Pm))), cos(2π(Po(Pm)))] + Le
        # Po: 位置编码（通过 point_proj 实现，不使用 PositionalEncoding 类）
        
        # Le: 学习嵌入
        self.learned_embeddings = nn.Parameter(torch.randn(embed_dim))
        
        # 点图到位置编码的投影
        self.point_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, pos_embed_dim, 3, padding=1)
        )
        
        # 位置编码到最终嵌入的投影
        self.pos_to_embed = nn.Linear(pos_embed_dim * 2, embed_dim)  # sin + cos = 2 * pos_embed_dim
    
    def forward(self, edge_map, point_map):
        """
        实现公式 (1): Ep, Pp = P[Em, Pm]
        
        Args:
            edge_map: [N, 1, H, W] 边缘图 Em
            point_map: [N, 1, H, W] 点图 Pm
            
        Returns:
            edge_tokens: [N, embed_dim] 边缘提示特征 Ep
            point_tokens: [N, embed_dim] 点提示特征 Pp
        """
        # 公式 (2): Ep = Conv(LN(G(Conv(LN(G(Conv(Em)))))))
        # 使用LayerNorm2d，直接处理2D特征图
        x = self.edge_conv1(edge_map)  # [N, 64, H/2, W/2]
        x = self.edge_gelu1(x)
        x = self.edge_ln1(x)  # LayerNorm2d on channel dimension
        
        x = self.edge_conv2(x)  # [N, 128, H/4, W/4]
        x = self.edge_gelu2(x)
        x = self.edge_ln2(x)  # LayerNorm2d on channel dimension
        
        edge_tokens = self.edge_conv3(x)  # [N, embed_dim, H/8, W/8]
        edge_tokens = F.adaptive_avg_pool2d(edge_tokens, (1, 1)).flatten(1)  # [N, embed_dim]
        
        # 公式 (3): Pp = C[sin(2π(Po(Pm))), cos(2π(Po(Pm)))] + Le
        point_tokens = self._encode_point_map(point_map)  # [N, embed_dim]
        
        return edge_tokens, point_tokens
    
    def _encode_point_map(self, point_map):
        """
        实现公式 (3): Pp = C[sin(2π(Po(Pm))), cos(2π(Po(Pm)))] + Le
        """
        N, _, H, W = point_map.shape
        
        # Po(Pm): 位置编码
        pos_encoded = self.point_proj(point_map)  # [N, pos_embed_dim, H, W]
        pos_encoded = F.adaptive_avg_pool2d(pos_encoded, (1, 1)).flatten(1)  # [N, pos_embed_dim]
        
        # 2π * Po(Pm)
        scaled_pos = 2 * math.pi * pos_encoded  # [N, pos_embed_dim]
        
        # sin(2π(Po(Pm))) 和 cos(2π(Po(Pm)))
        sin_pos = torch.sin(scaled_pos)  # [N, pos_embed_dim]
        cos_pos = torch.cos(scaled_pos)  # [N, pos_embed_dim]
        
        # C[sin(...), cos(...)]: 拼接
        concat_pos = torch.cat([sin_pos, cos_pos], dim=1)  # [N, pos_embed_dim * 2]
        
        # 投影到嵌入维度
        point_tokens = self.pos_to_embed(concat_pos)  # [N, embed_dim]
        
        # + Le: 添加学习嵌入
        point_tokens = point_tokens + self.learned_embeddings.unsqueeze(0).expand(N, -1)
        
        return point_tokens


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    edge_map = torch.randn(2, 1, 64, 64)
    point_map = torch.randn(2, 1, 64, 64)
    
    # 创建 Prompt Encoder
    prompt_encoder = PromptEncoder(embed_dim=256, pos_embed_dim=128)
    
    # 前向传播
    edge_tokens, point_tokens = prompt_encoder(edge_map, point_map)
    
    # print(f"Edge map shape: {edge_map.shape}")
    # print(f"Point map shape: {point_map.shape}")
    # print(f"Edge tokens shape: {edge_tokens.shape}")
    # print(f"Point tokens shape: {point_tokens.shape}")
    
    # 验证输出维度
    assert edge_tokens.shape == (2, 256), f"Expected (2, 256), got {edge_tokens.shape}"
    assert point_tokens.shape == (2, 256), f"Expected (2, 256), got {point_tokens.shape}"
    
    # print("✅ Paper Prompt Encoder 测试通过！")
