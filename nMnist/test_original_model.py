"""
测试原始EventSR模型是否正常工作
"""

import sys
import os
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model import NetworkBasic
import slayerSNN as snn

def test_original_model():
    print('Testing original EventSR model...')
    print(f'Current directory: {os.getcwd()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    try:
        # Load network parameters
        network_yaml_path = os.path.join(current_dir, 'network.yaml')
        netParams = snn.params(network_yaml_path)
        print('Network parameters loaded successfully')
        
        # Create model
        model = NetworkBasic(netParams)
        print('Model created successfully')
        
        # Move to CUDA
        model = model.cuda()
        print('Model moved to CUDA')
        
        # Test with dummy input
        dummy_input = torch.randn(1, 2, 17, 17, 350).cuda()
        dummy_input = (dummy_input > 0).float()
        print(f'Dummy input created: {dummy_input.shape}')
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f'Original model test successful!')
        print(f'Input shape: {dummy_input.shape}')
        print(f'Output shape: {output.shape}')
        return True
        
    except Exception as e:
        print(f'Original model test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_original_model()
    if success:
        print('Original EventSR model works correctly!')
    else:
        print('Original EventSR model has issues.')
