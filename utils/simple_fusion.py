"""
简化的特征融合模块

这个模块提供了一个简化但稳定的特征融合实现，避免复杂的注意力机制导致的维度问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleFusionModule(nn.Module):
    """简化的特征融合模块"""
    
    def __init__(self, 
                 snn_channels: int,
                 cnn_channels: int,
                 output_channels: int,
                 fusion_strategy: str = 'adaptive'):
        super(SimpleFusionModule, self).__init__()
        
        self.snn_channels = snn_channels
        self.cnn_channels = cnn_channels
        self.output_channels = output_channels
        self.fusion_strategy = fusion_strategy
        
        # 通道对齐层
        self.snn_align = nn.Conv2d(snn_channels, output_channels, 1)
        self.cnn_align = nn.Conv2d(cnn_channels, output_channels, 1)
        
        if fusion_strategy == 'adaptive':
            # 自适应权重生成
            self.weight_gen = nn.Sequential(
                nn.Conv2d(output_channels * 2, output_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, 2, 1),  # 2个权重：SNN和CNN
                nn.Softmax(dim=1)
            )
        elif fusion_strategy == 'concatenation':
            # 简单拼接
            self.concat_conv = nn.Sequential(
                nn.Conv2d(output_channels * 2, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        # element_wise 策略不需要额外的层
        
        # 最终增强层
        self.enhance = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, snn_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        融合SNN和CNN特征
        
        Args:
            snn_features: SNN特征 [B, C_snn, H, W]
            cnn_features: CNN特征 [B, C_cnn, H, W]
            
        Returns:
            融合后的特征 [B, output_channels, H, W]
        """
        # 确保空间维度匹配
        if snn_features.shape[2:] != cnn_features.shape[2:]:
            cnn_features = F.interpolate(
                cnn_features, 
                size=snn_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 通道对齐
        snn_aligned = self.snn_align(snn_features)
        cnn_aligned = self.cnn_align(cnn_features)
        
        # 根据策略进行融合
        if self.fusion_strategy == 'adaptive':
            # 自适应权重融合
            concat_features = torch.cat([snn_aligned, cnn_aligned], dim=1)
            weights = self.weight_gen(concat_features)  # [B, 2, H, W]
            
            snn_weight = weights[:, 0:1, :, :]  # [B, 1, H, W]
            cnn_weight = weights[:, 1:2, :, :]  # [B, 1, H, W]
            
            fused = snn_weight * snn_aligned + cnn_weight * cnn_aligned
            
        elif self.fusion_strategy == 'concatenation':
            # 拼接融合
            concat_features = torch.cat([snn_aligned, cnn_aligned], dim=1)
            fused = self.concat_conv(concat_features)
            
        elif self.fusion_strategy == 'element_wise':
            # 逐元素相加
            fused = snn_aligned + cnn_aligned
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # 特征增强
        fused = self.enhance(fused)
        
        return fused


class UltraSimpleFusion(nn.Module):
    """超简化融合模块，只做基本的特征组合"""
    
    def __init__(self, 
                 snn_channels: int,
                 cnn_channels: int,
                 output_channels: int):
        super(UltraSimpleFusion, self).__init__()
        
        # 简单的1x1卷积对齐通道
        self.snn_proj = nn.Conv2d(snn_channels, output_channels, 1)
        self.cnn_proj = nn.Conv2d(cnn_channels, output_channels, 1)
        
        # 简单的融合层
        self.fusion_conv = nn.Conv2d(output_channels * 2, output_channels, 1)
        
    def forward(self, snn_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        """超简化融合"""
        # 确保空间维度匹配
        if snn_features.shape[2:] != cnn_features.shape[2:]:
            cnn_features = F.interpolate(
                cnn_features, 
                size=snn_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 投影到相同通道数
        snn_proj = self.snn_proj(snn_features)
        cnn_proj = self.cnn_proj(cnn_features)
        
        # 拼接并融合
        concat = torch.cat([snn_proj, cnn_proj], dim=1)
        fused = self.fusion_conv(concat)
        
        return fused


def create_simple_fusion(snn_channels: int,
                        cnn_channels: int,
                        output_channels: int,
                        strategy: str = 'ultra_simple'):
    """
    创建简化的融合模块
    
    Args:
        snn_channels: SNN特征通道数
        cnn_channels: CNN特征通道数
        output_channels: 输出通道数
        strategy: 融合策略 ('ultra_simple', 'simple_adaptive', 'simple_concat', 'simple_element')
        
    Returns:
        融合模块
    """
    if strategy == 'ultra_simple':
        return UltraSimpleFusion(snn_channels, cnn_channels, output_channels)
    elif strategy.startswith('simple_'):
        fusion_type = strategy.replace('simple_', '')
        return SimpleFusionModule(snn_channels, cnn_channels, output_channels, fusion_type)
    else:
        return SimpleFusionModule(snn_channels, cnn_channels, output_channels, strategy)
