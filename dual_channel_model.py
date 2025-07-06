"""
Dual-Channel EventSR Model

This module implements the main dual-channel architecture that combines
SNN and CNN pathways for enhanced event-based super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import slayerSNN as snn

# Import existing model components
from model import NetworkBasic, Network1, Network2, Network3
from utils.event_frame_generator import EventFrameGenerator, create_event_frames_from_spike_tensor
from utils.cnn_channel import CNNChannel, TemporalCNNChannel
from utils.simple_cnn_channel import create_simple_cnn_channel
from utils.feature_fusion import FeatureFusionModule, MultiScaleFusion
from utils.simple_fusion import create_simple_fusion
from utils.utils import getNeuronConfig
from utils.slayer_cuda_fix import fix_slayer_model, apply_slayer_cuda_fix_hook


class DualChannelEventSR(nn.Module):
    """
    Dual-Channel Event Super-Resolution Network
    
    Combines SNN temporal processing with CNN spatial processing
    for enhanced super-resolution of event data.
    """
    
    def __init__(self,
                 netParams: Dict[str, Any],
                 snn_model_type: str = 'NetworkBasic',
                 cnn_base_channels: int = 64,
                 fusion_strategy: str = 'adaptive',
                 event_frame_strategy: str = 'time_based',
                 num_event_frames: int = 8,
                 scale_factor: int = 2,
                 use_temporal_cnn: bool = False,
                 output_format: str = 'event_stream',  # 'event_stream' or 'event_frame'
                 snn_theta: list = [30, 50, 100],
                 snn_tauSr: list = [1, 2, 4],
                 snn_tauRef: list = [1, 2, 4],
                 snn_scaleRef: list = [1, 1, 1],
                 snn_tauRho: list = [1, 1, 10],
                 snn_scaleRho: list = [10, 10, 100]):
        super(DualChannelEventSR, self).__init__()
        
        self.scale_factor = scale_factor
        self.fusion_strategy = fusion_strategy
        self.event_frame_strategy = event_frame_strategy
        self.num_event_frames = num_event_frames
        self.use_temporal_cnn = use_temporal_cnn
        self.output_format = output_format  # 'event_stream' or 'event_frame'
        
        # Initialize SNN Channel (existing EventSR model)
        self.snn_channel = self._create_snn_model(
            snn_model_type, netParams, snn_theta, snn_tauSr,
            snn_tauRef, snn_scaleRef, snn_tauRho, snn_scaleRho
        )

        # Apply CUDA fix hook to SNN channel
        apply_slayer_cuda_fix_hook(self.snn_channel)
        
        # Initialize Event Frame Generator
        self.event_frame_generator = None  # Will be initialized dynamically
        
        # Initialize CNN Channel (使用简化版本避免维度问题)
        self.cnn_channel = create_simple_cnn_channel(
            in_channels=None,  # 使用自适应模式
            out_channels=2,
            base_channels=cnn_base_channels,
            scale_factor=scale_factor,
            adaptive=True
        )
        
        # Feature extraction dimensions (will be set after first forward pass)
        self.snn_feature_dim = None
        self.cnn_feature_dim = None
        
        # Feature Fusion Module (will be initialized after determining feature dimensions)
        self.fusion_module = None
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Flag to track initialization
        self._initialized = False
        self._cuda_fixed = False
    
    def _create_snn_model(self, model_type: str, netParams: Dict[str, Any], 
                         theta: list, tauSr: list, tauRef: list, 
                         scaleRef: list, tauRho: list, scaleRho: list):
        """Create SNN model based on specified type."""
        if model_type == 'NetworkBasic':
            return NetworkBasic(netParams, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        elif model_type == 'Network1':
            return Network1(netParams, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        elif model_type == 'Network2':
            return Network2(netParams, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        elif model_type == 'Network3':
            return Network3(netParams, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        else:
            raise ValueError(f"Unknown SNN model type: {model_type}")
    
    def _initialize_components(self, input_shape: Tuple[int, ...]):
        """Initialize components that depend on input dimensions."""
        if self._initialized:
            return

        B, C, H, W, T = input_shape

        # Initialize Event Frame Generator
        self.event_frame_generator = EventFrameGenerator(
            height=H,
            width=W,
            accumulation_strategy=self.event_frame_strategy,
            num_frames=self.num_event_frames,
            polarity_channels=True
        )

        # Use default feature dimensions to avoid forward pass during initialization
        # These will be updated during the first actual forward pass
        self.snn_feature_dim = 2  # Default: same as input channels
        self.cnn_feature_dim = 2  # Default: same as input channels

        # Initialize Fusion Module with default dimensions (使用简化版本)
        self.fusion_module = create_simple_fusion(
            snn_channels=self.snn_feature_dim,
            cnn_channels=self.cnn_feature_dim,
            output_channels=2,  # Final output channels
            strategy='ultra_simple'  # 使用最简化的融合策略
        )

        # Move fusion module to the same device as the model
        device = next(self.parameters()).device
        self.fusion_module = self.fusion_module.to(device)

        self._initialized = True
    
    def _apply_cuda_fix_if_needed(self):
        """Apply CUDA fix to SNN channel if needed."""
        if not self._cuda_fixed and next(self.parameters()).is_cuda:
            print("Applying slayerSNN CUDA fix...")
            fix_slayer_model(self.snn_channel, 'cuda')
            self._cuda_fixed = True
            print("slayerSNN CUDA fix applied successfully")

    def forward(self, spike_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-channel architecture.

        Args:
            spike_input: Input spike tensor [B, C, H, W, T]

        Returns:
            Dictionary containing:
                - 'output': Final super-resolved output
                - 'snn_output': SNN pathway output
                - 'cnn_output': CNN pathway output
                - 'fused_features': Fused features before final layer
        """
        # Apply CUDA fix if needed
        self._apply_cuda_fix_if_needed()

        # Initialize components if needed
        if not self._initialized:
            self._initialize_components(spike_input.shape)
        
        B, C, H, W, T = spike_input.shape
        
        # SNN Channel Processing
        snn_output = self.snn_channel(spike_input)  # [B, C, 2H, 2W, T]
        
        # Generate Event Frames for CNN Channel
        event_frames = self.event_frame_generator.events_to_frames(spike_input)
        if len(event_frames.shape) == 4:  # [C, H, W, T]
            event_frames = event_frames.unsqueeze(0)  # Add batch dimension
        
        # Handle batch dimension for event frames
        if event_frames.shape[0] != B:
            event_frames = event_frames.repeat(B, 1, 1, 1, 1)
        
        # CNN Channel Processing
        cnn_output, cnn_features = self.cnn_channel(event_frames)
        
        # Extract features for fusion
        # For SNN: use the output directly as features
        snn_features = snn_output
        
        # For CNN: use the output as features
        cnn_features_for_fusion = cnn_output
        
        # Ensure spatial dimensions match for fusion
        target_height, target_width = snn_features.shape[2], snn_features.shape[3]
        if cnn_features_for_fusion.shape[2:4] != (target_height, target_width):
            cnn_features_for_fusion = F.interpolate(
                cnn_features_for_fusion,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Convert temporal dimension for fusion (average over time)
        if len(snn_features.shape) == 5:  # [B, C, H, W, T]
            snn_features_2d = snn_features.mean(dim=-1)  # [B, C, H, W]
        else:
            snn_features_2d = snn_features
        
        # Feature Fusion
        fused_features = self.fusion_module(snn_features_2d, cnn_features_for_fusion)
        
        # Final output processing
        final_output = self.final_conv(fused_features)

        # Output format configuration based on self.output_format
        if self.output_format == 'event_stream' and len(snn_output.shape) == 5:
            # Event Stream format: [B, C, H, W, T]
            final_output = final_output.unsqueeze(-1).repeat(1, 1, 1, 1, snn_output.shape[-1])
        # else: Event Frame format: [B, C, H, W] (keep as is)
        
        return {
            'output': final_output,
            'snn_output': snn_output,
            'cnn_output': cnn_output,
            'fused_features': fused_features,
            'event_frames': event_frames
        }
    
    def get_snn_output(self, spike_input: torch.Tensor) -> torch.Tensor:
        """Get only SNN pathway output."""
        return self.snn_channel(spike_input)
    
    def get_cnn_output(self, spike_input: torch.Tensor) -> torch.Tensor:
        """Get only CNN pathway output."""
        if not self._initialized:
            self._initialize_components(spike_input.shape)
        
        event_frames = self.event_frame_generator.events_to_frames(spike_input)
        if len(event_frames.shape) == 4:
            event_frames = event_frames.unsqueeze(0)
        
        cnn_output, _ = self.cnn_channel(event_frames)
        return cnn_output
    
    def set_fusion_strategy(self, strategy: str):
        """Change fusion strategy and reinitialize fusion module."""
        self.fusion_strategy = strategy
        if self._initialized and self.fusion_module is not None:
            self.fusion_module = FeatureFusionModule(
                snn_channels=self.snn_feature_dim,
                cnn_channels=self.cnn_feature_dim,
                output_channels=2,
                fusion_strategy=strategy
            )
    
    def freeze_snn_channel(self):
        """Freeze SNN channel parameters for fine-tuning CNN channel."""
        for param in self.snn_channel.parameters():
            param.requires_grad = False
    
    def unfreeze_snn_channel(self):
        """Unfreeze SNN channel parameters."""
        for param in self.snn_channel.parameters():
            param.requires_grad = True
    
    def freeze_cnn_channel(self):
        """Freeze CNN channel parameters for fine-tuning SNN channel."""
        for param in self.cnn_channel.parameters():
            param.requires_grad = False
    
    def unfreeze_cnn_channel(self):
        """Unfreeze CNN channel parameters."""
        for param in self.cnn_channel.parameters():
            param.requires_grad = True


def create_dual_channel_model(config: Dict[str, Any]) -> DualChannelEventSR:
    """
    Factory function to create dual-channel model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DualChannelEventSR model instance
    """
    # Default SNN parameters
    netParams = config.get('netParams', {'simulation': {'Ts': 1, 'tSample': 350}})
    
    model = DualChannelEventSR(
        netParams=netParams,
        snn_model_type=config.get('snn_model_type', 'NetworkBasic'),
        cnn_base_channels=config.get('cnn_base_channels', 64),
        fusion_strategy=config.get('fusion_strategy', 'adaptive'),
        event_frame_strategy=config.get('event_frame_strategy', 'time_based'),
        num_event_frames=config.get('num_event_frames', 8),
        scale_factor=config.get('scale_factor', 2),
        use_temporal_cnn=config.get('use_temporal_cnn', False),
        output_format=config.get('output_format', 'event_stream'),
        snn_theta=config.get('snn_theta', [30, 50, 100]),
        snn_tauSr=config.get('snn_tauSr', [1, 2, 4]),
        snn_tauRef=config.get('snn_tauRef', [1, 2, 4]),
        snn_scaleRef=config.get('snn_scaleRef', [1, 1, 1]),
        snn_tauRho=config.get('snn_tauRho', [1, 1, 10]),
        snn_scaleRho=config.get('snn_scaleRho', [10, 10, 100])
    )
    
    return model
