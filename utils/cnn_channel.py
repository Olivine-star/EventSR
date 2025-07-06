"""
CNN Channel Architecture for Dual-Channel EventSR

This module implements the CNN pathway for processing event frames,
focusing on spatial feature extraction and super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ResidualBlock(nn.Module):
    """Residual block for CNN feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class AttentionBlock(nn.Module):
    """Spatial attention block for feature enhancement."""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        
        # Attention weights
        attention = self.conv1(avg_pool)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class CNNEncoder(nn.Module):
    """CNN Encoder for spatial feature extraction from event frames."""
    
    def __init__(self, 
                 in_channels: int = 2,  # For positive/negative polarity
                 base_channels: int = 64,
                 num_blocks: int = 3):
        super(CNNEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                     stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with increasing channels
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1  # Downsample after first block
            
            self.encoder_blocks.append(
                ResidualBlock(current_channels, out_channels, stride)
            )
            current_channels = out_channels
        
        # Attention mechanism
        self.attention = AttentionBlock(current_channels)
        
        self.output_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input event frames [B, C, H, W, T] or [B, C*T, H, W]

        Returns:
            List of feature maps at different scales
        """
        # Handle temporal dimension
        if len(x.shape) == 5:  # [B, C, H, W, T]
            B, C, H, W, T = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, T, H, W]
            x = x.view(B, C * T, H, W)  # Flatten temporal dimension

        # Ensure input channels match expected input channels
        if x.shape[1] != self.in_channels:
            # Add a projection layer if needed
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Conv2d(x.shape[1], self.in_channels, 1).to(x.device)
            x = self.input_projection(x)
        
        features = []
        
        # Initial convolution
        x = self.initial_conv(x)
        features.append(x)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        
        # Apply attention to final features
        x = self.attention(x)
        features[-1] = x
        
        return features


class CNNDecoder(nn.Module):
    """CNN Decoder for super-resolution reconstruction."""
    
    def __init__(self, 
                 encoder_channels: List[int],
                 out_channels: int = 2,
                 scale_factor: int = 2):
        super(CNNDecoder, self).__init__()
        
        self.scale_factor = scale_factor
        self.encoder_channels = encoder_channels
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Reverse the encoder channels for decoder
        decoder_channels = encoder_channels[::-1]
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Skip connection doubles the input channels
            if i > 0:
                in_ch += decoder_channels[i]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, 
                                     stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    ResidualBlock(out_ch, out_ch)
                )
            )
        
        # Final output layer
        final_in_ch = decoder_channels[-1]
        if len(decoder_channels) > 1:
            final_in_ch += decoder_channels[-1]  # Skip connection
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_ch, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # Additional 2x upsampling
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output activation
        )
    
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            encoder_features: List of feature maps from encoder
            
        Returns:
            Super-resolved output
        """
        # Start with the deepest features
        x = encoder_features[-1]
        
        # Decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i > 0:
                # Add skip connection from encoder
                skip_idx = len(encoder_features) - 2 - i
                if skip_idx >= 0:
                    skip_features = encoder_features[skip_idx]
                    # Resize if necessary
                    if x.shape[2:] != skip_features.shape[2:]:
                        x = F.interpolate(x, size=skip_features.shape[2:], 
                                        mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip_features], dim=1)
            
            x = decoder_block(x)
        
        # Add final skip connection
        if len(encoder_features) > 1:
            skip_features = encoder_features[0]
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], 
                                mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_features], dim=1)
        
        # Final output
        x = self.final_conv(x)
        
        return x


class CNNChannel(nn.Module):
    """Complete CNN Channel for processing event frames."""
    
    def __init__(self, 
                 in_channels: int = 2,
                 out_channels: int = 2,
                 base_channels: int = 64,
                 num_encoder_blocks: int = 3,
                 scale_factor: int = 2):
        super(CNNChannel, self).__init__()
        
        self.encoder = CNNEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        # Calculate encoder output channels
        encoder_channels = [base_channels]
        for i in range(num_encoder_blocks):
            encoder_channels.append(base_channels * (2 ** i))
        
        self.decoder = CNNDecoder(
            encoder_channels=encoder_channels,
            out_channels=out_channels,
            scale_factor=scale_factor
        )
        
        self.scale_factor = scale_factor
    
    def forward(self, event_frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through CNN channel.
        
        Args:
            event_frames: Input event frames [B, C, H, W, T] or [B, C*T, H, W]
            
        Returns:
            Tuple of (super_resolved_output, encoder_features)
        """
        # Encode
        encoder_features = self.encoder(event_frames)
        
        # Decode
        output = self.decoder(encoder_features)
        
        return output, encoder_features


class TemporalCNNChannel(nn.Module):
    """CNN Channel with explicit temporal processing."""
    
    def __init__(self, 
                 in_channels: int = 2,
                 out_channels: int = 2,
                 base_channels: int = 64,
                 temporal_frames: int = 8,
                 scale_factor: int = 2):
        super(TemporalCNNChannel, self).__init__()
        
        self.temporal_frames = temporal_frames
        
        # Temporal convolution to process frame sequences
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels // 2, 
                     kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels // 2, base_channels, 
                     kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial CNN channel
        self.spatial_cnn = CNNChannel(
            in_channels=base_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            scale_factor=scale_factor
        )
    
    def forward(self, event_frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with temporal processing.
        
        Args:
            event_frames: Input event frames [B, C, H, W, T]
            
        Returns:
            Tuple of (super_resolved_output, encoder_features)
        """
        if len(event_frames.shape) != 5:
            raise ValueError("TemporalCNNChannel expects 5D input [B, C, H, W, T]")
        
        B, C, H, W, T = event_frames.shape
        
        # Temporal processing
        x = self.temporal_conv(event_frames)  # [B, base_channels, H, W, T]
        
        # Flatten temporal dimension for spatial processing
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, base_channels, T, H, W]
        x = x.view(B, -1, H, W)  # [B, base_channels*T, H, W]
        
        # Spatial super-resolution
        output, features = self.spatial_cnn(x)
        
        return output, features
