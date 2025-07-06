"""
测试修复后的双通道EventSR模型
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

def test_fixed_dual_channel():
    print('Testing fixed dual-channel EventSR model...')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    try:
        # Model configuration
        network_yaml_path = os.path.join(current_dir, 'network.yaml')
        netParams = snn.params(network_yaml_path)
        
        model_config = {
            'netParams': netParams,
            'snn_model_type': 'NetworkBasic',
            'cnn_base_channels': 32,
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
        
        # Test forward pass
        print('Testing forward pass...')
        dummy_input = torch.randn(2, 2, 17, 17, 350).cuda()
        dummy_input = (dummy_input > 0).float()
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f'Fixed dual-channel model test successful!')
        print(f'Input shape: {dummy_input.shape}')
        print(f'Output shapes:')
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f'  {key}: {tensor.shape}')
        
        return True
        
    except Exception as e:
        print(f'Fixed dual-channel model test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_fixed_dual_channel()
    if success:
        print('\n✅ Fixed dual-channel model works correctly!')
    else:
        print('\n❌ Fixed dual-channel model still has issues.')
