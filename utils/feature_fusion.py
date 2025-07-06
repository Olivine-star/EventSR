"""
Feature Fusion Module for Dual-Channel EventSR

This module implements intelligent fusion mechanisms to combine 
SNN temporal features and CNN spatial features for enhanced super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature fusion."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature fusion."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing SNN and CNN features."""
    
    def __init__(self, 
                 snn_channels: int, 
                 cnn_channels: int, 
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super(CrossAttentionFusion, self).__init__()
        
        self.snn_channels = snn_channels
        self.cnn_channels = cnn_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project features to common dimension
        self.snn_proj = nn.Conv2d(snn_channels, hidden_dim, 1)
        self.cnn_proj = nn.Conv2d(cnn_channels, hidden_dim, 1)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, snn_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention fusion of SNN and CNN features.
        
        Args:
            snn_features: SNN features [B, C_snn, H, W]
            cnn_features: CNN features [B, C_cnn, H, W]
            
        Returns:
            Fused features [B, hidden_dim, H, W]
        """
        B, _, H, W = snn_features.shape
        
        # Project to common dimension
        snn_proj = self.snn_proj(snn_features)  # [B, hidden_dim, H, W]
        cnn_proj = self.cnn_proj(cnn_features)  # [B, hidden_dim, H, W]
        
        # Reshape for attention: [B, H*W, hidden_dim]
        snn_flat = snn_proj.view(B, self.hidden_dim, -1).transpose(1, 2)
        cnn_flat = cnn_proj.view(B, self.hidden_dim, -1).transpose(1, 2)
        
        # Cross attention: SNN attends to CNN
        snn_attended, _ = self.multihead_attn(snn_flat, cnn_flat, cnn_flat)
        
        # Cross attention: CNN attends to SNN
        cnn_attended, _ = self.multihead_attn(cnn_flat, snn_flat, snn_flat)
        
        # Concatenate and project
        fused_flat = torch.cat([snn_attended, cnn_attended], dim=-1)  # [B, H*W, 2*hidden_dim]
        
        # Reshape back to spatial
        fused = fused_flat.transpose(1, 2).view(B, 2 * self.hidden_dim, H, W)
        fused = self.output_proj(fused)  # [B, hidden_dim, H, W]
        
        return fused


class AdaptiveFusion(nn.Module):
    """Adaptive fusion with learnable weights."""
    
    def __init__(self, 
                 snn_channels: int, 
                 cnn_channels: int, 
                 output_channels: int):
        super(AdaptiveFusion, self).__init__()
        
        # Channel alignment
        self.snn_align = nn.Conv2d(snn_channels, output_channels, 1)
        self.cnn_align = nn.Conv2d(cnn_channels, output_channels, 1)
        
        # Fusion weight generation
        self.weight_gen = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, 2, 1),  # 2 weights for SNN and CNN
            nn.Softmax(dim=1)
        )
        
        # Feature enhancement
        self.enhance = CBAM(output_channels)
    
    def forward(self, snn_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion with learnable weights.
        
        Args:
            snn_features: SNN features [B, C_snn, H, W]
            cnn_features: CNN features [B, C_cnn, H, W]
            
        Returns:
            Fused features [B, output_channels, H, W]
        """
        # Align channels
        snn_aligned = self.snn_align(snn_features)
        cnn_aligned = self.cnn_align(cnn_features)
        
        # Generate fusion weights
        concat_features = torch.cat([snn_aligned, cnn_aligned], dim=1)
        weights = self.weight_gen(concat_features)  # [B, 2, H, W]
        
        # Weighted fusion
        snn_weight = weights[:, 0:1, :, :]  # [B, 1, H, W]
        cnn_weight = weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        fused = snn_weight * snn_aligned + cnn_weight * cnn_aligned
        
        # Feature enhancement
        fused = self.enhance(fused)
        
        return fused


class FeatureFusionModule(nn.Module):
    """Main feature fusion module with multiple fusion strategies."""
    
    def __init__(self, 
                 snn_channels: int,
                 cnn_channels: int,
                 output_channels: int,
                 fusion_strategy: str = 'adaptive',
                 hidden_dim: int = 256):
        super(FeatureFusionModule, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.snn_channels = snn_channels
        self.cnn_channels = cnn_channels
        self.output_channels = output_channels
        
        if fusion_strategy == 'adaptive':
            self.fusion = AdaptiveFusion(snn_channels, cnn_channels, output_channels)
        elif fusion_strategy == 'cross_attention':
            self.fusion = CrossAttentionFusion(snn_channels, cnn_channels, hidden_dim)
            self.output_proj = nn.Conv2d(hidden_dim, output_channels, 1)
        elif fusion_strategy == 'concatenation':
            self.fusion = nn.Sequential(
                nn.Conv2d(snn_channels + cnn_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                CBAM(output_channels)
            )
        elif fusion_strategy == 'element_wise':
            self.snn_proj = nn.Conv2d(snn_channels, output_channels, 1)
            self.cnn_proj = nn.Conv2d(cnn_channels, output_channels, 1)
            self.fusion = CBAM(output_channels)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(self, snn_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse SNN and CNN features.
        
        Args:
            snn_features: SNN features [B, C_snn, H, W]
            cnn_features: CNN features [B, C_cnn, H, W]
            
        Returns:
            Fused features [B, output_channels, H, W]
        """
        # Ensure spatial dimensions match
        if snn_features.shape[2:] != cnn_features.shape[2:]:
            cnn_features = F.interpolate(
                cnn_features, 
                size=snn_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if self.fusion_strategy == 'adaptive':
            return self.fusion(snn_features, cnn_features)
        
        elif self.fusion_strategy == 'cross_attention':
            fused = self.fusion(snn_features, cnn_features)
            return self.output_proj(fused)
        
        elif self.fusion_strategy == 'concatenation':
            concat_features = torch.cat([snn_features, cnn_features], dim=1)
            return self.fusion(concat_features)
        
        elif self.fusion_strategy == 'element_wise':
            snn_proj = self.snn_proj(snn_features)
            cnn_proj = self.cnn_proj(cnn_features)
            fused = snn_proj + cnn_proj  # Element-wise addition
            return self.fusion(fused)


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion for hierarchical features."""
    
    def __init__(self, 
                 snn_channel_list: List[int],
                 cnn_channel_list: List[int],
                 output_channels: int,
                 fusion_strategy: str = 'adaptive'):
        super(MultiScaleFusion, self).__init__()
        
        assert len(snn_channel_list) == len(cnn_channel_list), \
            "SNN and CNN channel lists must have same length"
        
        self.num_scales = len(snn_channel_list)
        
        # Fusion modules for each scale
        self.fusion_modules = nn.ModuleList()
        for i in range(self.num_scales):
            self.fusion_modules.append(
                FeatureFusionModule(
                    snn_channels=snn_channel_list[i],
                    cnn_channels=cnn_channel_list[i],
                    output_channels=output_channels,
                    fusion_strategy=fusion_strategy
                )
            )
        
        # Multi-scale aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(output_channels * self.num_scales, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            CBAM(output_channels)
        )
    
    def forward(self, 
                snn_feature_list: List[torch.Tensor], 
                cnn_feature_list: List[torch.Tensor],
                target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Multi-scale feature fusion.
        
        Args:
            snn_feature_list: List of SNN features at different scales
            cnn_feature_list: List of CNN features at different scales
            target_size: Target spatial size for output
            
        Returns:
            Fused multi-scale features
        """
        if target_size is None:
            target_size = snn_feature_list[0].shape[2:]
        
        fused_features = []
        
        for i, (snn_feat, cnn_feat) in enumerate(zip(snn_feature_list, cnn_feature_list)):
            # Fuse features at this scale
            fused = self.fusion_modules[i](snn_feat, cnn_feat)
            
            # Resize to target size
            if fused.shape[2:] != target_size:
                fused = F.interpolate(
                    fused, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            fused_features.append(fused)
        
        # Aggregate multi-scale features
        multi_scale_features = torch.cat(fused_features, dim=1)
        output = self.aggregation(multi_scale_features)
        
        return output
