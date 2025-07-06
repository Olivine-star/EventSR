"""
Configuration Manager for Dual-Channel EventSR

This module provides utilities for loading and managing configuration
files for the dual-channel architecture.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SNNConfig:
    """SNN channel configuration."""
    model_type: str = "NetworkBasic"
    theta: list = field(default_factory=lambda: [30, 50, 100])
    tauSr: list = field(default_factory=lambda: [1, 2, 4])
    tauRef: list = field(default_factory=lambda: [1, 2, 4])
    scaleRef: list = field(default_factory=lambda: [1, 1, 1])
    tauRho: list = field(default_factory=lambda: [1, 1, 10])
    scaleRho: list = field(default_factory=lambda: [10, 10, 100])


@dataclass
class CNNConfig:
    """CNN channel configuration."""
    base_channels: int = 64
    num_encoder_blocks: int = 3
    use_temporal_cnn: bool = False
    temporal_frames: int = 8


@dataclass
class FusionConfig:
    """Feature fusion configuration."""
    strategy: str = "adaptive"
    hidden_dim: int = 256
    num_attention_heads: int = 8


@dataclass
class EventFrameConfig:
    """Event frame generation configuration."""
    strategy: str = "time_based"
    num_frames: int = 8
    time_window: float = 50.0
    event_count: int = 1000
    normalize: bool = True
    polarity_channels: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration."""
    snn: SNNConfig = field(default_factory=SNNConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    event_frames: EventFrameConfig = field(default_factory=EventFrameConfig)
    scale_factor: int = 2
    input_channels: int = 2
    output_channels: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 0.001
    optimizer_type: str = "Adam"
    lr_scheduler_type: str = "ExponentialLR"
    lr_gamma: float = 0.95
    save_interval: int = 10
    log_interval: int = 10
    checkpoint_dir: str = "./ckpt_dual_channel/"
    
    # Loss weights
    snn_weight: float = 1.0
    cnn_weight: float = 1.0
    fusion_weight: float = 2.0
    temporal_weight: float = 5.0


@dataclass
class DualChannelConfig:
    """Complete dual-channel configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    cuda_devices: str = "0"
    experiment_name: str = "dual_channel_eventsr"


class ConfigManager:
    """Configuration manager for dual-channel EventSR."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.config = self._dict_to_config(config_dict)
        self.config_path = config_path
        
        return config_dict
    
    def save_config(self, config_path: str):
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = self._config_to_dict(self.config)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for creating dual-channel model.
        
        Returns:
            Model configuration dictionary
        """
        if self.config is None:
            self.config = self._get_default_config()
        
        return {
            'netParams': {'simulation': {'Ts': 1, 'tSample': 350}},
            'snn_model_type': self.config.model.snn.model_type,
            'cnn_base_channels': self.config.model.cnn.base_channels,
            'fusion_strategy': self.config.model.fusion.strategy,
            'event_frame_strategy': self.config.model.event_frames.strategy,
            'num_event_frames': self.config.model.event_frames.num_frames,
            'scale_factor': self.config.model.scale_factor,
            'use_temporal_cnn': self.config.model.cnn.use_temporal_cnn,
            'snn_theta': self.config.model.snn.theta,
            'snn_tauSr': self.config.model.snn.tauSr,
            'snn_tauRef': self.config.model.snn.tauRef,
            'snn_scaleRef': self.config.model.snn.scaleRef,
            'snn_tauRho': self.config.model.snn.tauRho,
            'snn_scaleRho': self.config.model.snn.scaleRho
        }
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        if self.config is None:
            self.config = self._get_default_config()
        
        return self.config.training
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            self.config = self._get_default_config()
        
        self._update_nested_dict(self._config_to_dict(self.config), updates)
        self.config = self._dict_to_config(self._config_to_dict(self.config))
    
    def _get_default_config(self) -> DualChannelConfig:
        """Get default configuration."""
        return DualChannelConfig()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> DualChannelConfig:
        """Convert dictionary to configuration object."""
        # Extract model configuration
        model_dict = config_dict.get('model', {})
        
        snn_config = SNNConfig(**model_dict.get('snn', {}))
        cnn_config = CNNConfig(**model_dict.get('cnn', {}))
        fusion_config = FusionConfig(**model_dict.get('fusion', {}))
        event_frame_config = EventFrameConfig(**model_dict.get('event_frames', {}))
        
        model_config = ModelConfig(
            snn=snn_config,
            cnn=cnn_config,
            fusion=fusion_config,
            event_frames=event_frame_config,
            scale_factor=model_dict.get('scale_factor', 2),
            input_channels=model_dict.get('input_channels', 2),
            output_channels=model_dict.get('output_channels', 2)
        )
        
        # Extract training configuration
        training_dict = config_dict.get('training', {})
        data_dict = training_dict.get('data', {})
        optimizer_dict = training_dict.get('optimizer', {})
        lr_scheduler_dict = training_dict.get('lr_scheduler', {})
        loss_dict = training_dict.get('loss', {})
        checkpoint_dict = training_dict.get('checkpoint', {})
        
        training_config = TrainingConfig(
            batch_size=data_dict.get('batch_size', 16),
            num_workers=data_dict.get('num_workers', 4),
            epochs=training_dict.get('epochs', 100),
            lr=optimizer_dict.get('lr', 0.001),
            optimizer_type=optimizer_dict.get('type', 'Adam'),
            lr_scheduler_type=lr_scheduler_dict.get('type', 'ExponentialLR'),
            lr_gamma=lr_scheduler_dict.get('gamma', 0.95),
            save_interval=training_dict.get('save_interval', 10),
            log_interval=training_dict.get('log_interval', 10),
            checkpoint_dir=checkpoint_dict.get('save_dir', './ckpt_dual_channel/'),
            snn_weight=loss_dict.get('snn_weight', 1.0),
            cnn_weight=loss_dict.get('cnn_weight', 1.0),
            fusion_weight=loss_dict.get('fusion_weight', 2.0),
            temporal_weight=loss_dict.get('temporal_weight', 5.0)
        )
        
        # Extract hardware configuration
        hardware_dict = config_dict.get('hardware', {})
        experiment_dict = config_dict.get('experiment', {})
        
        return DualChannelConfig(
            model=model_config,
            training=training_config,
            device=hardware_dict.get('device', 'cuda'),
            cuda_devices=hardware_dict.get('cuda_devices', '0'),
            experiment_name=experiment_dict.get('name', 'dual_channel_eventsr')
        )
    
    def _config_to_dict(self, config: DualChannelConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'model': {
                'snn': {
                    'model_type': config.model.snn.model_type,
                    'theta': config.model.snn.theta,
                    'tauSr': config.model.snn.tauSr,
                    'tauRef': config.model.snn.tauRef,
                    'scaleRef': config.model.snn.scaleRef,
                    'tauRho': config.model.snn.tauRho,
                    'scaleRho': config.model.snn.scaleRho
                },
                'cnn': {
                    'base_channels': config.model.cnn.base_channels,
                    'num_encoder_blocks': config.model.cnn.num_encoder_blocks,
                    'use_temporal_cnn': config.model.cnn.use_temporal_cnn,
                    'temporal_frames': config.model.cnn.temporal_frames
                },
                'fusion': {
                    'strategy': config.model.fusion.strategy,
                    'hidden_dim': config.model.fusion.hidden_dim,
                    'num_attention_heads': config.model.fusion.num_attention_heads
                },
                'event_frames': {
                    'strategy': config.model.event_frames.strategy,
                    'num_frames': config.model.event_frames.num_frames,
                    'time_window': config.model.event_frames.time_window,
                    'event_count': config.model.event_frames.event_count,
                    'normalize': config.model.event_frames.normalize,
                    'polarity_channels': config.model.event_frames.polarity_channels
                },
                'scale_factor': config.model.scale_factor,
                'input_channels': config.model.input_channels,
                'output_channels': config.model.output_channels
            },
            'training': {
                'batch_size': config.training.batch_size,
                'num_workers': config.training.num_workers,
                'epochs': config.training.epochs,
                'lr': config.training.lr,
                'optimizer_type': config.training.optimizer_type,
                'lr_scheduler_type': config.training.lr_scheduler_type,
                'lr_gamma': config.training.lr_gamma,
                'save_interval': config.training.save_interval,
                'log_interval': config.training.log_interval,
                'checkpoint_dir': config.training.checkpoint_dir,
                'snn_weight': config.training.snn_weight,
                'cnn_weight': config.training.cnn_weight,
                'fusion_weight': config.training.fusion_weight,
                'temporal_weight': config.training.temporal_weight
            },
            'hardware': {
                'device': config.device,
                'cuda_devices': config.cuda_devices
            },
            'experiment': {
                'name': config.experiment_name
            }
        }
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], updates: Dict[str, Any]):
        """Update nested dictionary with new values."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value


def load_config(config_path: str) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def create_default_config(save_path: str):
    """
    Create and save default configuration file.
    
    Args:
        save_path: Path to save default configuration
    """
    config_manager = ConfigManager()
    config_manager.save_config(save_path)
