"""
Example usage of the Dual-Channel EventSR implementation

This script demonstrates how to use the dual-channel architecture
for event-based super-resolution.
"""

import torch
import os
import sys
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append('.')

from dual_channel_model import create_dual_channel_model
from utils.config_manager import ConfigManager, create_default_config
from nMnist.mnistDatasetSR import mnistDatasetDualChannel
from utils.event_frame_generator import EventFrameGenerator


def create_sample_config():
    """Create a sample configuration file."""
    config_path = 'configs/example_config.yaml'
    os.makedirs('configs', exist_ok=True)
    
    if not os.path.exists(config_path):
        create_default_config(config_path)
        print(f"Created sample configuration at {config_path}")
    
    return config_path


def demonstrate_event_frame_generation():
    """Demonstrate event frame generation."""
    print("\n" + "="*50)
    print("DEMONSTRATING EVENT FRAME GENERATION")
    print("="*50)
    
    # Create a sample spike tensor
    spike_tensor = torch.randn(2, 17, 17, 350)  # [C, H, W, T]
    spike_tensor = (spike_tensor > 0).float()  # Convert to binary spikes
    
    print(f"Input spike tensor shape: {spike_tensor.shape}")
    print(f"Number of spikes: {spike_tensor.sum().item()}")
    
    # Create event frame generator
    generator = EventFrameGenerator(
        height=17, width=17,
        accumulation_strategy='time_based',
        num_frames=8,
        polarity_channels=True
    )
    
    # Generate event frames
    event_frames = generator.events_to_frames(spike_tensor)
    print(f"Generated event frames shape: {event_frames.shape}")
    print(f"Event frames range: [{event_frames.min().item():.3f}, {event_frames.max().item():.3f}]")
    
    # Test different strategies
    strategies = ['time_based', 'count_based', 'adaptive']
    for strategy in strategies:
        generator.accumulation_strategy = strategy
        frames = generator.events_to_frames(spike_tensor)
        print(f"Strategy '{strategy}': {frames.shape}, sum={frames.sum().item():.1f}")


def demonstrate_model_creation():
    """Demonstrate model creation and basic usage."""
    print("\n" + "="*50)
    print("DEMONSTRATING MODEL CREATION")
    print("="*50)
    
    # Create configuration
    config_path = create_sample_config()
    config_manager = ConfigManager(config_path)
    model_config = config_manager.get_model_config()
    
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print("\nCreating dual-channel model...")
    model = create_dual_channel_model(model_config)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def demonstrate_inference():
    """Demonstrate model inference."""
    print("\n" + "="*50)
    print("DEMONSTRATING MODEL INFERENCE")
    print("="*50)
    
    # Create model
    model = demonstrate_model_creation()
    model.eval()
    
    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 2, 17, 17, 350)
    input_tensor = (input_tensor > 0).float()  # Convert to binary spikes
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input spikes per sample: {input_tensor.sum(dim=(1,2,3,4))}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Display results
    print("\nModel outputs:")
    for key, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    
    # Test individual pathways
    print("\nTesting individual pathways:")
    
    with torch.no_grad():
        snn_only = model.get_snn_output(input_tensor)
        cnn_only = model.get_cnn_output(input_tensor)
    
    print(f"  SNN pathway: {snn_only.shape}")
    print(f"  CNN pathway: {cnn_only.shape}")


def demonstrate_fusion_strategies():
    """Demonstrate different fusion strategies."""
    print("\n" + "="*50)
    print("DEMONSTRATING FUSION STRATEGIES")
    print("="*50)
    
    # Create model
    model = demonstrate_model_creation()
    model.eval()
    
    # Sample input
    input_tensor = torch.randn(1, 2, 17, 17, 350)
    input_tensor = (input_tensor > 0).float()
    
    fusion_strategies = ['adaptive', 'cross_attention', 'concatenation', 'element_wise']
    
    for strategy in fusion_strategies:
        print(f"\nTesting fusion strategy: {strategy}")
        
        try:
            model.set_fusion_strategy(strategy)
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            output_tensor = outputs['output']
            print(f"  Output shape: {output_tensor.shape}")
            print(f"  Output range: [{output_tensor.min().item():.3f}, {output_tensor.max().item():.3f}]")
            print(f"  Output sum: {output_tensor.sum().item():.1f}")
            
        except Exception as e:
            print(f"  Error with {strategy}: {e}")


def demonstrate_dataset_usage():
    """Demonstrate dataset usage."""
    print("\n" + "="*50)
    print("DEMONSTRATING DATASET USAGE")
    print("="*50)
    
    try:
        # Create dual-channel dataset
        print("Creating dual-channel dataset...")
        dataset = mnistDatasetDualChannel(
            train=False,
            event_frame_strategy='time_based',
            num_event_frames=8
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        print("\nGetting sample from dataset...")
        sample = dataset[0]
        
        print("Sample contents:")
        for key, tensor in sample.items():
            print(f"  {key}: {tensor.shape}, sum={tensor.sum().item():.1f}")
        
        # Create dataloader
        print("\nCreating dataloader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        # Get a batch
        batch = next(iter(dataloader))
        print("\nBatch contents:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        
    except Exception as e:
        print(f"Dataset demonstration failed: {e}")
        print("This is expected if the dataset files are not available.")


def main():
    """Main demonstration function."""
    print("DUAL-CHANNEL EVENTSR DEMONSTRATION")
    print("="*60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run demonstrations
    try:
        demonstrate_event_frame_generation()
        demonstrate_model_creation()
        demonstrate_inference()
        demonstrate_fusion_strategies()
        demonstrate_dataset_usage()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Prepare your dataset following the EventSR format")
        print("2. Modify configs/example_config.yaml for your needs")
        print("3. Run training: python nMnist/trainDualChannel.py")
        print("4. Evaluate results: python nMnist/testDualChannel.py")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        print("Please check your installation and dependencies.")


if __name__ == '__main__':
    main()
