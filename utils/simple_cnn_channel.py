"""
简化的CNN通道实现，专门用于双通道EventSR

这个模块提供了一个简化但稳定的CNN实现，避免复杂的维度匹配问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SimpleCNNChannel(nn.Module):
    """简化的CNN通道，用于处理事件帧"""
    
    def __init__(self, 
                 in_channels: int = 16,  # 2 * 8 frames
                 out_channels: int = 2,
                 base_channels: int = 32,
                 scale_factor: int = 2):
        super(SimpleCNNChannel, self).__init__()
        
        self.scale_factor = scale_factor
        
        # 输入投影层，确保输入通道数正确
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 特征提取层（用于融合）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(base_channels * 4, out_channels, 1),
            nn.AdaptiveAvgPool2d((34, 34))  # 确保输出尺寸匹配SNN输出
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入事件帧 [B, C, H, W, T] 或 [B, C*T, H, W]
            
        Returns:
            Tuple of (output, features)
        """
        # 处理时间维度
        if len(x.shape) == 5:  # [B, C, H, W, T]
            B, C, H, W, T = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, T, H, W]
            x = x.view(B, C * T, H, W)  # 展平时间维度
        
        # 输入投影
        x = self.input_proj(x)
        
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        output = self.decoder(encoded)
        
        # 提取特征用于融合
        features = self.feature_extractor(encoded)
        
        return output, features


class AdaptiveCNNChannel(nn.Module):
    """自适应CNN通道，可以处理任意输入维度"""
    
    def __init__(self, 
                 out_channels: int = 2,
                 base_channels: int = 32,
                 scale_factor: int = 2):
        super(AdaptiveCNNChannel, self).__init__()
        
        self.scale_factor = scale_factor
        self.base_channels = base_channels
        self.out_channels = out_channels
        
        # 这些层将在第一次前向传播时初始化
        self.input_proj = None
        self.encoder = None
        self.decoder = None
        self.feature_extractor = None
        self._initialized = False
    
    def _initialize_layers(self, in_channels: int):
        """根据输入通道数初始化层"""
        if self._initialized:
            return
        
        # 输入投影层
        self.input_proj = nn.Conv2d(in_channels, self.base_channels, 3, padding=1)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.base_channels, self.base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.base_channels, self.out_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.base_channels * 4, self.out_channels, 1),
            nn.AdaptiveAvgPool2d((34, 34))
        )
        
        # 移动到正确的设备
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_proj = self.input_proj.to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.feature_extractor = self.feature_extractor.to(device)
        
        self._initialized = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入事件帧 [B, C, H, W, T] 或 [B, C*T, H, W]
            
        Returns:
            Tuple of (output, features)
        """
        # 处理时间维度
        if len(x.shape) == 5:  # [B, C, H, W, T]
            B, C, H, W, T = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, T, H, W]
            x = x.view(B, C * T, H, W)  # 展平时间维度
        
        # 初始化层（如果需要）
        if not self._initialized:
            self._initialize_layers(x.shape[1])

        # 确保所有层都在正确的设备上
        device = x.device
        if self.input_proj.weight.device != device:
            self.input_proj = self.input_proj.to(device)
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            self.feature_extractor = self.feature_extractor.to(device)
        
        # 前向传播
        x = self.input_proj(x)
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        features = self.feature_extractor(encoded)
        
        return output, features


def create_simple_cnn_channel(in_channels: int = None, 
                             out_channels: int = 2,
                             base_channels: int = 32,
                             scale_factor: int = 2,
                             adaptive: bool = True):
    """
    创建简化的CNN通道
    
    Args:
        in_channels: 输入通道数（如果为None且adaptive=True，则使用自适应模式）
        out_channels: 输出通道数
        base_channels: 基础通道数
        scale_factor: 缩放因子
        adaptive: 是否使用自适应模式
        
    Returns:
        CNN通道模型
    """
    if adaptive or in_channels is None:
        return AdaptiveCNNChannel(
            out_channels=out_channels,
            base_channels=base_channels,
            scale_factor=scale_factor
        )
    else:
        return SimpleCNNChannel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            scale_factor=scale_factor
        )
