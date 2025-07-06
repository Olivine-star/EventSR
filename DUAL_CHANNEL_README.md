# Dual-Channel EventSR Implementation

This document describes the implementation of a dual-channel architecture for Event-based Super-Resolution (EventSR) that combines Spiking Neural Networks (SNN) and Convolutional Neural Networks (CNN) for enhanced temporal continuity and spatial detail reconstruction.

## Overview

The dual-channel architecture processes event data through two parallel pathways:

1. **SNN Channel**: Maintains the existing SNN-based processing pipeline for temporal dynamics
2. **CNN Channel**: Processes event frames using traditional CNN layers for spatial feature extraction
3. **Feature Fusion**: Intelligently combines features from both channels for enhanced super-resolution

## Architecture Components

### 1. Event Frame Generation (`utils/event_frame_generator.py`)

Converts raw event data into accumulated event frames for CNN processing.

**Features:**
- Multiple accumulation strategies: time-based, count-based, adaptive
- Support for polarity channels (positive/negative events)
- Configurable temporal windows and frame counts
- Normalization options

**Usage:**
```python
from utils.event_frame_generator import EventFrameGenerator

generator = EventFrameGenerator(
    height=17, width=17,
    accumulation_strategy='time_based',
    num_frames=8,
    polarity_channels=True
)

event_frames = generator.events_to_frames(spike_tensor)
```

### 2. CNN Channel (`utils/cnn_channel.py`)

Implements CNN pathway for spatial feature extraction and super-resolution.

**Components:**
- **CNNEncoder**: Spatial feature extraction with residual blocks and attention
- **CNNDecoder**: Super-resolution reconstruction with skip connections
- **TemporalCNNChannel**: Explicit temporal processing with 3D convolutions

**Features:**
- Residual blocks for deep feature extraction
- Spatial attention mechanisms
- Multi-scale feature processing
- Configurable encoder/decoder depths

### 3. Feature Fusion (`utils/feature_fusion.py`)

Intelligent fusion mechanisms to combine SNN and CNN features.

**Fusion Strategies:**
- **Adaptive Fusion**: Learnable weights for dynamic feature combination
- **Cross-Attention**: Multi-head attention between SNN and CNN features
- **Concatenation**: Simple feature concatenation with enhancement
- **Element-wise**: Element-wise addition with attention refinement

**Components:**
- Channel and spatial attention modules (CBAM)
- Multi-scale fusion for hierarchical features
- Configurable fusion parameters

### 4. Dual-Channel Model (`dual_channel_model.py`)

Main model class that integrates all components.

**Features:**
- Unified architecture combining SNN and CNN channels
- Dynamic component initialization
- Multiple SNN model support (NetworkBasic, Network1, Network2, Network3)
- Flexible fusion strategy selection
- Training mode controls (freeze/unfreeze channels)

## Configuration System

### Configuration File (`configs/dual_channel_config.yaml`)

Comprehensive configuration system supporting:
- Model architecture parameters
- Training hyperparameters
- Fusion strategy settings
- Event frame generation options
- Hardware and logging configuration

### Configuration Manager (`utils/config_manager.py`)

Python interface for configuration management:
- YAML file loading/saving
- Type-safe configuration objects
- Dynamic configuration updates
- Default configuration generation

## Data Loading

### Extended Dataset (`nMnist/mnistDatasetSR.py`)

**mnistDatasetDualChannel**: Extended dataset class that provides:
- Original spike tensors for SNN processing
- Generated event frames for CNN processing
- Configurable event frame generation strategies
- Backward compatibility with existing datasets

**Usage:**
```python
from nMnist.mnistDatasetSR import mnistDatasetDualChannel

dataset = mnistDatasetDualChannel(
    train=True,
    event_frame_strategy='time_based',
    num_event_frames=8
)

# Returns dictionary with both spike tensors and event frames
batch = dataset[0]
# batch['lr_spikes'], batch['hr_spikes'], batch['lr_frames'], batch['hr_frames']
```

## Training

### Training Script (`nMnist/trainDualChannel.py`)

Comprehensive training pipeline with:
- **DualChannelLoss**: Combined loss function for SNN, CNN, and fusion outputs
- Multi-component optimization
- Temporal consistency loss (ECM loss)
- Comprehensive logging and checkpointing

**Key Features:**
- Separate loss terms for each pathway
- Temporal consistency enforcement
- Configurable loss weights
- TensorBoard integration
- Best model saving

**Usage:**
```bash
cd nMnist
python trainDualChannel.py --bs 16 --lr 0.001 --epoch 100 --cuda 0
```

## Testing and Evaluation

### Evaluation Script (`nMnist/testDualChannel.py`)

Comprehensive evaluation framework:
- **Model Comparison**: Dual-channel vs. baseline SNN
- **Fusion Strategy Analysis**: Compare different fusion approaches
- **Visualization**: Generate comparison plots and error maps
- **Metrics**: MSE, PSNR, SSIM, temporal consistency

**Evaluation Metrics:**
- **MSE**: Mean Squared Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Temporal Consistency**: Frame-to-frame coherence

## Installation and Setup

### Prerequisites

1. **Python Dependencies** (same as original EventSR):
   ```
   torch==1.3.1
   torchvision==0.4.2
   numpy==1.19.2
   pyyaml==5.4.1
   tensorboardX==2.2
   scikit-image
   scikit-learn
   matplotlib
   ```

2. **slayerPytorch**: Follow original EventSR installation instructions

### Setup Steps

1. **Install Dependencies**:
   ```bash
   pip install pyyaml scikit-image scikit-learn matplotlib
   ```

2. **Create Configuration**:
   ```python
   from utils.config_manager import create_default_config
   create_default_config('configs/my_config.yaml')
   ```

3. **Prepare Data**: Use existing EventSR dataset structure

## Usage Examples

### Basic Training

```python
from dual_channel_model import create_dual_channel_model
from utils.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager('configs/dual_channel_config.yaml')
model_config = config_manager.get_model_config()

# Create model
model = create_dual_channel_model(model_config)

# Train using provided script
# python nMnist/trainDualChannel.py
```

### Custom Model Configuration

```python
from dual_channel_model import DualChannelEventSR
import slayerSNN as snn

# Custom configuration
netParams = snn.params('network.yaml')
model = DualChannelEventSR(
    netParams=netParams,
    snn_model_type='NetworkBasic',
    cnn_base_channels=64,
    fusion_strategy='adaptive',
    event_frame_strategy='time_based',
    num_event_frames=8,
    scale_factor=2
)
```

### Inference

```python
# Load trained model
model = create_dual_channel_model(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(input_spikes)
    super_resolved = outputs['output']
    snn_output = outputs['snn_output']
    cnn_output = outputs['cnn_output']
```

## Integration with Existing EventSR

The dual-channel implementation is designed for seamless integration:

1. **Backward Compatibility**: Original EventSR models and datasets remain functional
2. **Modular Design**: Components can be used independently
3. **Configuration-Driven**: Easy switching between single and dual-channel modes
4. **Existing Infrastructure**: Leverages existing training, evaluation, and utility functions

## Performance Expectations

Based on the SpikeSR architecture inspiration, expected improvements:

1. **Temporal Continuity**: Enhanced through SNN pathway's temporal processing
2. **Spatial Detail**: Improved through CNN pathway's spatial feature extraction
3. **Overall Quality**: Better super-resolution through intelligent feature fusion
4. **Flexibility**: Multiple fusion strategies for different scenarios

## Troubleshooting

### Common Issues

1. **Memory Usage**: Dual-channel processing requires more GPU memory
   - Solution: Reduce batch size or use gradient checkpointing

2. **Training Instability**: Multiple loss terms can cause instability
   - Solution: Adjust loss weights in configuration

3. **Convergence**: Different pathways may converge at different rates
   - Solution: Use progressive training (freeze/unfreeze channels)

### Debug Mode

Enable detailed logging by setting log level to DEBUG in configuration:
```yaml
logging:
  console:
    level: "DEBUG"
```

## Future Enhancements

Potential improvements and extensions:

1. **Multi-Scale Fusion**: Hierarchical feature fusion at multiple scales
2. **Attention Mechanisms**: More sophisticated attention between pathways
3. **Progressive Training**: Curriculum learning strategies
4. **Model Compression**: Pruning and quantization for deployment
5. **Real-time Processing**: Optimizations for real-time applications

## Citation

If you use this dual-channel implementation, please cite both the original EventSR paper and acknowledge this extension:

```bibtex
@inproceedings{eventsr2021,
  title={Event Stream Super-Resolution via Spatiotemporal Constraint Learning},
  author={...},
  booktitle={ICCV},
  year={2021}
}
```

## Quick Start Example

Here's a complete example to get started with the dual-channel architecture:

```python
# example_usage.py
import torch
from dual_channel_model import create_dual_channel_model
from utils.config_manager import ConfigManager
from nMnist.mnistDatasetSR import mnistDatasetDualChannel
from torch.utils.data import DataLoader

# 1. Load configuration
config_manager = ConfigManager('configs/dual_channel_config.yaml')
model_config = config_manager.get_model_config()

# 2. Create model
model = create_dual_channel_model(model_config)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# 3. Create dataset
dataset = mnistDatasetDualChannel(train=False, num_event_frames=8)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 4. Run inference
model.eval()
with torch.no_grad():
    for batch in dataloader:
        lr_spikes = batch['lr_spikes']
        outputs = model(lr_spikes)

        print(f"Input shape: {lr_spikes.shape}")
        print(f"Output shape: {outputs['output'].shape}")
        print(f"SNN output shape: {outputs['snn_output'].shape}")
        print(f"CNN output shape: {outputs['cnn_output'].shape}")
        break
```

## License

This implementation follows the same license as the original EventSR project.
