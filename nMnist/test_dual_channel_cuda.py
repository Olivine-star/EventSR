"""
测试双通道EventSR模型的CUDA兼容性
"""

import sys
import os
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dual_channel_model import create_dual_channel_model
import slayerSNN as snn

def test_dual_channel_cuda():
    print('Testing dual-channel EventSR model CUDA compatibility...')
    print(f'Current directory: {os.getcwd()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    try:
        # Model configuration
        network_yaml_path = os.path.join(current_dir, 'network.yaml')
        netParams = snn.params(network_yaml_path)
        
        model_config = {
            'netParams': netParams,
            'snn_model_type': 'NetworkBasic',
            'cnn_base_channels': 32,  # 减小以便快速测试
            'fusion_strategy': 'adaptive',
            'event_frame_strategy': 'time_based',
            'num_event_frames': 8,
            'scale_factor': 2,
            'use_temporal_cnn': False,
            'output_format': 'event_stream'
        }
        
        print('Creating dual-channel model...')
        model = create_dual_channel_model(model_config)
        print('Model created successfully')
        
        print('Moving model to CUDA...')
        model = model.cuda()
        print('Model moved to CUDA')
        
        # Test individual components
        print('\nTesting individual components:')
        
        # Test SNN channel
        print('Testing SNN channel...')
        dummy_input = torch.randn(1, 2, 17, 17, 350).cuda()
        dummy_input = (dummy_input > 0).float()
        
        with torch.no_grad():
            snn_output = model.get_snn_output(dummy_input)
        print(f'SNN channel test successful: {snn_output.shape}')
        
        # Test CNN channel
        print('Testing CNN channel...')
        with torch.no_grad():
            cnn_output = model.get_cnn_output(dummy_input)
        print(f'CNN channel test successful: {cnn_output.shape}')
        
        # Test full forward pass
        print('Testing full forward pass...')
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f'Dual-channel model test successful!')
        print(f'Input shape: {dummy_input.shape}')
        print(f'Output shapes:')
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f'  {key}: {tensor.shape}')
        
        return True
        
    except Exception as e:
        print(f'Dual-channel model test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_step_by_step():
    """逐步测试每个组件"""
    print('\n' + '='*60)
    print('STEP-BY-STEP COMPONENT TESTING')
    print('='*60)
    
    try:
        # 1. Test SNN model creation
        print('Step 1: Testing SNN model creation...')
        network_yaml_path = os.path.join(current_dir, 'network.yaml')
        netParams = snn.params(network_yaml_path)
        
        from model import NetworkBasic
        snn_model = NetworkBasic(netParams).cuda()
        print('✓ SNN model created and moved to CUDA')
        
        # 2. Test CNN channel creation
        print('Step 2: Testing CNN channel creation...')
        from utils.simple_cnn_channel import create_simple_cnn_channel
        cnn_model = create_simple_cnn_channel(
            in_channels=None,  # 使用自适应模式
            out_channels=2,
            base_channels=32,
            scale_factor=2,
            adaptive=True
        ).cuda()
        print('✓ CNN channel created and moved to CUDA')
        
        # 3. Test feature fusion
        print('Step 3: Testing feature fusion...')
        from utils.feature_fusion import FeatureFusionModule
        fusion_model = FeatureFusionModule(
            snn_channels=2,
            cnn_channels=2,
            output_channels=2,
            fusion_strategy='adaptive'
        ).cuda()
        print('✓ Feature fusion module created and moved to CUDA')
        
        # 4. Test event frame generator
        print('Step 4: Testing event frame generator...')
        from utils.event_frame_generator import EventFrameGenerator
        frame_generator = EventFrameGenerator(
            height=17, width=17,
            accumulation_strategy='time_based',
            num_frames=8,
            polarity_channels=True
        )
        print('✓ Event frame generator created')
        
        # 5. Test with dummy data
        print('Step 5: Testing with dummy data...')
        dummy_input = torch.randn(1, 2, 17, 17, 350).cuda()
        dummy_input = (dummy_input > 0).float()
        
        # Test SNN
        with torch.no_grad():
            snn_out = snn_model(dummy_input)
        print(f'✓ SNN forward pass: {snn_out.shape}')
        
        # Test event frame generation
        event_frames = frame_generator.events_to_frames(dummy_input.cpu())
        event_frames = event_frames.unsqueeze(0).cuda()  # Add batch dim and move to CUDA
        print(f'✓ Event frame generation: {event_frames.shape}')
        
        # Test CNN
        with torch.no_grad():
            cnn_out, _ = cnn_model(event_frames)
        print(f'✓ CNN forward pass: {cnn_out.shape}')
        
        # Test fusion
        snn_features = snn_out.mean(dim=-1)  # Remove time dimension
        with torch.no_grad():
            fused_out = fusion_model(snn_features, cnn_out)
        print(f'✓ Feature fusion: {fused_out.shape}')
        
        print('\n✓ All components work correctly!')
        return True
        
    except Exception as e:
        print(f'\n✗ Component test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print('DUAL-CHANNEL EVENTSR CUDA COMPATIBILITY TEST')
    print('='*60)
    
    # Test step by step first
    step_success = test_step_by_step()
    
    if step_success:
        # Test full model
        full_success = test_dual_channel_cuda()
        
        if full_success:
            print('\n' + '='*60)
            print('✓ ALL TESTS PASSED - DUAL-CHANNEL MODEL IS CUDA COMPATIBLE!')
            print('='*60)
        else:
            print('\n' + '='*60)
            print('✗ FULL MODEL TEST FAILED')
            print('='*60)
    else:
        print('\n' + '='*60)
        print('✗ COMPONENT TESTS FAILED')
        print('='*60)
