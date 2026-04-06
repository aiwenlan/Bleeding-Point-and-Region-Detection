import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from sam2.build_sam import build_sam2
SAM2_AVAILABLE = True

class SAM2Backbone(nn.Module):
    """
    基于Surgical-SAM-2的SAM2主干网络包装器
    实现多帧时序编码和特征金字塔提取
    严格按照论文要求，必须使用真正的SAM2模型
    """
    def __init__(self, sam2_config="sam2_hiera_b+.yaml", ckpt_path=None):
        super().__init__()
        self.sam2_config = sam2_config
        self.ckpt_path = ckpt_path
        
        # 特征维度配置
        self.feature_dims = {
            'f1': 64,   # 高分辨率特征 (1/4 scale)
            'f2': 128,  # 中分辨率特征 (1/8 scale)
            'f3': 256   # 低分辨率特征 (1/16 scale)
        }
        
        # 必须使用真实的SAM2模型
        if not ckpt_path:
            raise ValueError("sam2_ckpt is required. Please provide the path to SAM2 weights.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SAM2 weights not found at {ckpt_path}")
        
        try:
            print(f"Loading SAM2 with config: {sam2_config}")
            print(f"Loading SAM2 with checkpoint: {ckpt_path}")
            
            # 直接使用 build_sam2 来构建模型并加载权重
            print("Using build_sam2 to construct model and load weights...")
            self.sam2_model = build_sam2(
                config_file=sam2_config,
                ckpt_path=ckpt_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                mode='train'
            )
            
            self.image_encoder = self.sam2_model.image_encoder
            self.use_real_sam2 = True
            print(f"Successfully loaded SAM2 model from {ckpt_path}")
            
            # 确保模型的所有参数都是相同的数据类型
            model_dtype = next(self.sam2_model.parameters()).dtype
            print(f"SAM2 model dtype: {model_dtype}")
            
            # 检查关键组件是否存在
            print(f"Image encoder: {hasattr(self.sam2_model, 'image_encoder')}")
            print(f"Prompt encoder: {hasattr(self.sam2_model, 'sam_prompt_encoder')}")
            print(f"Mask decoder: {hasattr(self.sam2_model, 'sam_mask_decoder')}")
            
            if hasattr(self.sam2_model, 'sam_prompt_encoder'):
                print(f"Prompt encoder initialized: {self.sam2_model.sam_prompt_encoder is not None}")
            if hasattr(self.sam2_model, 'sam_mask_decoder'):
                print(f"Mask decoder initialized: {self.sam2_model.sam_mask_decoder is not None}")
            
            # 创建转置卷积层用于上采样（替代插值）
            # 用于 image_embeddings 的上采样（2倍）
            self.upsample_2x = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
            # 用于 feat_s1 的上采样（2倍）
            self.upsample_feat_s1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
            # 用于 feat_s0 的上采样（4倍，分两次2倍上采样）
            self.upsample_feat_s0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
            
        except Exception as e:
            print(f"Failed to load SAM2 model: {e}")
            print(f"Config path: {sam2_config}")
            print(f"Checkpoint path: {ckpt_path}")
            raise RuntimeError(f"Cannot load required SAM2 model from {ckpt_path}. Error: {e}")
    
    
    def forward(self, frames_seq):
        """
        前向传播 (使用真实SAM2模型)
        frames_seq: [N, T, 3, H, W] 输入帧序列
        返回: List[Dict[str, Tensor]] 每帧的多尺度特征
        """
        N, T, C, H, W = frames_seq.shape
        features_list = []
        
        for t in range(T):
            frame = frames_seq[:, t]  # [N, 3, H, W]
            
            # SAM2图像编码
            features = self.image_encoder(frame)
            
            # 提取多尺度特征 - 根据SAM2实际输出结构
            if isinstance(features, dict):
                # vision_features: 图像编码器的直接输出 (最低分辨率，用于后续分支)
                # backbone_fpn: 特征金字塔列表 [最高分辨率, ..., 最低分辨率]
                # vision_pos_enc: 位置编码列表
                f1 = features['backbone_fpn'][0]  # 特征金字塔最底层 (最高分辨率)
                f2 = features['backbone_fpn'][1] if len(features['backbone_fpn']) > 1 else features['backbone_fpn'][0]  # 中间层
                f3 = features['vision_features']  # 图像编码器直接输出 (最低分辨率，用于mask/point分支)
            else:
                # 如果是单一特征，创建多尺度
                f3 = features  # 图像编码器的直接输出
                f2 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=False)
                f1 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=False)
            
            features_list.append({
                'f1': f1,  # 特征金字塔最底层 (最高分辨率)
                'f2': f2,  # 特征金字塔中间层 (中分辨率)
                'f3': f3   # 图像编码器直接输出 (最低分辨率，用于mask/point分支)
            })
            
            # 打印每帧的特征图形状
            # print(f"🔍 第{t+1}帧特征图形状:")
            # print(f"   f1 (高分辨率): {f1.shape}")
            # print(f"   f2 (中分辨率): {f2.shape}")
            # print(f"   f3 (低分辨率): {f3.shape}")
        
        return features_list
    
    
    def mask_decode(self, features, edge_tokens, point_tokens, f1=None, f2=None):
        """
        基于SAM2的掩码解码器 (使用真实SAM2模型)
        
        根据论文要求和架构图：
        - features: [N, C, H, W] Fmask特征 (来自Mask Memory Modeling的时空特征)
          数据流: Fk(图像编码器) → Mask Memory Modeling → Fmask(时空特征) → SAM2解码器
        - edge_tokens: [N, C] 边缘提示tokens (来自Edge Generator的Em)
        - point_tokens: [N, C] 点提示tokens (来自Point Branch的Pm)
        - f1: [N, C1, H1, W1] F1高分辨率特征 (来自Image Encoder，用于high_res_features)
        - f2: [N, C2, H2, W2] F2中分辨率特征 (来自Image Encoder，用于high_res_features)
        
        论文: "We input Fmask into the mask decoder similar to SAM 2"
        架构图: Mask Decoder接收Fmask作为image_embeddings，F1和F2作为high_res_features
        
        返回:
        - masks: [N, 1, H, W] - 预测掩码
        - iou_predictions: [N, 1] - IoU 预测分数
        - object_pointer: [N, 1, 256] - 对象指针（mask token，用于 Memory Bank）
        - occlusion_score: [N, 1] - 遮挡分数（用于遮挡判断）
        """
        # 使用SAM2的mask decoder进行解码
        try:
            # 处理特征输入 - 根据论文要求，应该使用Fmask特征
            
            # Fmask特征张量 - 这是论文要求的正确输入
            image_embeddings = features
            # print("Using Fmask features as required by the paper.")
            
            # 检查特征尺寸是否匹配
            # print(f"Image embeddings shape: {image_embeddings.shape}")
            # print(f"Edge tokens shape: {edge_tokens.shape}")
            # print(f"Point tokens shape: {point_tokens.shape}")
            
            # 获取SAM2 prompt_encoder期望的尺寸，确保 image_embeddings (Fmask) 的尺寸与SAM2内部期望的尺寸一致
            expected_pe = self.sam2_model.sam_prompt_encoder.get_dense_pe()
            expected_h, expected_w = expected_pe.shape[-2:]
            # print(f"SAM2 Expected image_embedding_size: {self.sam2_model.sam_prompt_encoder.image_embedding_size}")
            
            # 如果image_embeddings尺寸不匹配，需要上采样到期望尺寸
            
            image_embeddings = self.upsample_2x(image_embeddings)
       
            # print(f"Resized image_embeddings shape: {image_embeddings.shape}")
            
            # 直接使用传入的prompt features (edge_tokens, point_tokens)
            # 根据论文设计，这些是经过PromptEncoder处理的特征
            N = edge_tokens.shape[0]
            # print(f"Using provided prompt features:")
            # print(f"  edge_tokens shape: {edge_tokens.shape}")
            # print(f"  point_tokens shape: {point_tokens.shape}")
            
            # 根据正确的架构理解：
            # sparse_embeddings 应该是 point_tokens (用于自注意力)
            # dense_embeddings 应该是 edge_tokens (用于与Fmask的交叉注意力)
            sparse_embeddings = point_tokens.unsqueeze(1)  # [N, 1, 256] - point_tokens
            
            # 将edge_tokens转换为dense格式 (空间特征图)
            # edge_tokens: [N, 256] -> dense_embeddings: [N, 256, H, W]
            edge_tokens_expanded = edge_tokens.unsqueeze(-1).unsqueeze(-1)  # [N, 256, 1, 1]
            dense_embeddings = edge_tokens_expanded.expand(N, 256, expected_h, expected_w)  # [N, 256, H, W]
            
            # print(f"Generated sparse_embeddings shape: {sparse_embeddings.shape}")
            # print(f"Generated dense_embeddings shape: {dense_embeddings.shape}")
            
            # 获取位置编码
            image_pe = self.sam2_model.sam_prompt_encoder.get_dense_pe()
         
            
            # 调用SAM2的掩码解码器
            # print(f"Final check before SAM2 decoder:")
            # print(f"  image_embeddings: {image_embeddings.shape}")
            # print(f"  image_pe: {image_pe.shape}")
            # print(f"  sparse_prompt_embeddings: {sparse_embeddings.shape}")
            # print(f"  dense_prompt_embeddings: {dense_embeddings.shape}")
            
            # print("Calling SAM2 predict_masks directly...")
            
            # 检查 mask_decoder 的状态
            # print(f"Mask decoder type: {type(self.sam2_model.sam_mask_decoder)}")
            # print(f"Mask decoder training mode: {self.sam2_model.sam_mask_decoder.training}")
            
            # 检查关键属性
            # print(f"  num_mask_tokens: {self.sam2_model.sam_mask_decoder.num_mask_tokens}")
            # print(f"  use_high_res_features: {self.sam2_model.sam_mask_decoder.use_high_res_features}")
            # print(f"  pred_obj_scores: {self.sam2_model.sam_mask_decoder.pred_obj_scores}")
            
            # 准备高分辨率特征（当 use_high_res_features=True 时需要）
            # 根据架构图，应该使用从 Image Encoder 来的 F1 和 F2
            # 使用真实的 F1 和 F2 特征（符合架构图）
            # 通过 conv_s0 和 conv_s1 投影到正确的通道数
            # 然后上采样到期望的尺寸
            # feat_s0: [1, 32, 256, 256] (F1 -> conv_s0 -> 上采样4倍)
            # feat_s1: [1, 64, 128, 128] (F2 -> conv_s1 -> 上采样2倍)
            
            # 先通过卷积投影通道数
            feat_s0 = self.sam2_model.sam_mask_decoder.conv_s0(f1)  # F1 -> [N, 32, H1, W1]
            feat_s1 = self.sam2_model.sam_mask_decoder.conv_s1(f2)  # F2 -> [N, 64, H2, W2]
            
            # 根据 image_embeddings 的尺寸计算期望的高分辨率特征尺寸
            # image_embeddings: [N, 256, H, W] (例如 [1, 256, 64, 64])
            # feat_s1 应该是 [N, 64, H*2, W*2] (例如 [1, 64, 128, 128])
            # feat_s0 应该是 [N, 32, H*4, W*4] (例如 [1, 32, 256, 256])
            
            # 上采样到期望的尺寸（使用转置卷积替代插值）
            
            
            feat_s0 = self.upsample_feat_s0(feat_s0)
            feat_s1 = self.upsample_feat_s1(feat_s1)
            
            high_res_features = [feat_s0, feat_s1]
            # print(f"  Using real F1 and F2 features: feat_s0={feat_s0.shape}, feat_s1={feat_s1.shape}")
            
            
            # 使用 forward 方法 (更简洁，自动选择最佳掩码)
            # print("Calling SAM2 mask_decoder forward...")
    
            decoder_output = self.sam2_model.sam_mask_decoder(
                image_embeddings=image_embeddings,  # 使用Fmask
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,  # 自动返回1个最佳掩码
                repeat_image=False,
                high_res_features=high_res_features,
            )
            # print("SAM2 forward call completed successfully")
            
            # print(f"decoder_output type: {type(decoder_output)}")
            if decoder_output is None:
                # print("ERROR: SAM2 decoder returned None!")
                raise RuntimeError("SAM2 decoder returned None")
            
            # 解包 forward 的返回值
            # SAM2 mask_decoder.forward() 返回 4 个值：
            # 1. masks: [N, num_masks, H, W] - 预测掩码
            # 2. iou_pred: [N, num_masks] - IoU 预测分数
            # 3. sam_tokens_out: [N, 1, 256] - 对象指针（mask token）
            # 4. object_score_logits: [N, 1] - 遮挡分数（logits）
            if len(decoder_output) >= 4:
                masks, iou_predictions, object_pointer, occlusion_score = decoder_output
                # print(f"  masks shape: {masks.shape}")
                # print(f"  iou_predictions shape: {iou_predictions.shape}")
                # print(f"  object_pointer shape: {object_pointer.shape}")
                # print(f"  occlusion_score shape: {occlusion_score.shape}")
            elif len(decoder_output) >= 2:
                # 兼容旧版本（只有 masks 和 iou_predictions）
                masks, iou_predictions = decoder_output[0], decoder_output[1]
                object_pointer = None
                occlusion_score = None
                # print(f"  Warning: SAM2 decoder returned only {len(decoder_output)} values")
            else:
                # print("ERROR: SAM2 decoder returned insufficient values!")
                raise RuntimeError(f"SAM2 decoder returned only {len(decoder_output)} values, expected at least 2")
            
            # 返回完整输出（包含对象指针和遮挡分数）
            return {
                'masks': masks,                    # [N, 1, H, W] - 预测掩码
                'iou_predictions': iou_predictions, # [N, 1] - IoU 预测分数
                'object_pointer': object_pointer,   # [N, 1, 256] - 对象指针（用于 Memory Bank）
                'occlusion_score': occlusion_score  # [N, 1] - 遮挡分数（用于遮挡判断）
            }
            
        except Exception as e:
            # print(f"Error: SAM2 mask decoder failed: {e}")
            # print(f"Features type: {type(features)}")
            # if hasattr(features, 'shape'):
            #     print(f"Features shape: {features.shape}")
            # print(f"Edge tokens shape: {edge_tokens.shape}")
            # print(f"Point tokens shape: {point_tokens.shape}")
            raise RuntimeError(f"SAM2 mask decoder failed: {e}")
    
    
    
    