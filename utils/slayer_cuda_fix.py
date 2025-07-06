"""
slayerSNN CUDA兼容性修复工具

这个模块提供了修复slayerSNN库CUDA兼容性问题的工具函数。
主要解决filter权重没有正确移动到CUDA设备的问题。
"""

import torch
import torch.nn as nn


def fix_slayer_cuda_recursive(module, device='cuda'):
    """
    递归修复slayerSNN模块的CUDA兼容性问题
    
    Args:
        module: 要修复的模块
        device: 目标设备
    """
    # 修复当前模块
    fix_slayer_cuda_module(module, device)
    
    # 递归修复子模块
    for child in module.children():
        fix_slayer_cuda_recursive(child, device)


def fix_slayer_cuda_module(module, device='cuda'):
    """
    修复单个slayerSNN模块的CUDA兼容性问题
    
    Args:
        module: 要修复的模块
        device: 目标设备
    """
    # 检查并修复srmKernel
    if hasattr(module, 'srmKernel'):
        if module.srmKernel is not None:
            if not module.srmKernel.is_cuda and device == 'cuda':
                print(f"Moving srmKernel to CUDA for {type(module).__name__}")
                module.srmKernel = module.srmKernel.cuda()
            elif module.srmKernel.is_cuda and device == 'cpu':
                print(f"Moving srmKernel to CPU for {type(module).__name__}")
                module.srmKernel = module.srmKernel.cpu()
    
    # 检查并修复refKernel
    if hasattr(module, 'refKernel'):
        if module.refKernel is not None:
            if not module.refKernel.is_cuda and device == 'cuda':
                print(f"Moving refKernel to CUDA for {type(module).__name__}")
                module.refKernel = module.refKernel.cuda()
            elif module.refKernel.is_cuda and device == 'cpu':
                print(f"Moving refKernel to CPU for {type(module).__name__}")
                module.refKernel = module.refKernel.cpu()
    
    # 检查并修复其他可能的kernel属性
    kernel_attrs = ['kernel', 'filter', 'weight_kernel', 'bias_kernel']
    for attr in kernel_attrs:
        if hasattr(module, attr):
            kernel = getattr(module, attr)
            if kernel is not None and isinstance(kernel, torch.Tensor):
                if not kernel.is_cuda and device == 'cuda':
                    print(f"Moving {attr} to CUDA for {type(module).__name__}")
                    setattr(module, attr, kernel.cuda())
                elif kernel.is_cuda and device == 'cpu':
                    print(f"Moving {attr} to CPU for {type(module).__name__}")
                    setattr(module, attr, kernel.cpu())


def fix_slayer_model(model, device='cuda'):
    """
    修复整个模型的slayerSNN CUDA兼容性问题
    
    Args:
        model: 要修复的模型
        device: 目标设备
        
    Returns:
        修复后的模型
    """
    print(f"Fixing slayerSNN CUDA compatibility for device: {device}")
    
    # 首先移动模型到目标设备
    model = model.to(device)
    
    # 然后修复slayerSNN特定的问题
    fix_slayer_cuda_recursive(model, device)
    
    print("slayerSNN CUDA fix completed")
    return model


def check_slayer_cuda_status(model):
    """
    检查模型中slayerSNN组件的CUDA状态
    
    Args:
        model: 要检查的模型
        
    Returns:
        检查报告字典
    """
    report = {
        'modules_checked': 0,
        'cuda_kernels': 0,
        'cpu_kernels': 0,
        'issues': []
    }
    
    def check_module(module, name=""):
        report['modules_checked'] += 1
        
        # 检查srmKernel
        if hasattr(module, 'srmKernel') and module.srmKernel is not None:
            if module.srmKernel.is_cuda:
                report['cuda_kernels'] += 1
            else:
                report['cpu_kernels'] += 1
                report['issues'].append(f"{name}.srmKernel is on CPU")
        
        # 检查refKernel
        if hasattr(module, 'refKernel') and module.refKernel is not None:
            if module.refKernel.is_cuda:
                report['cuda_kernels'] += 1
            else:
                report['cpu_kernels'] += 1
                report['issues'].append(f"{name}.refKernel is on CPU")
        
        # 递归检查子模块
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            check_module(child, full_name)
    
    check_module(model, "model")
    return report


def apply_slayer_cuda_fix_hook(model):
    """
    为模型添加CUDA修复钩子，在每次forward前自动修复

    Args:
        model: 要添加钩子的模型
    """
    def cuda_fix_hook(module, input):
        # 检查输入是否在CUDA上
        if isinstance(input, (list, tuple)) and len(input) > 0:
            first_input = input[0]
            if isinstance(first_input, torch.Tensor) and first_input.is_cuda:
                # 强制修复所有kernel到CUDA
                fix_slayer_cuda_module(module, 'cuda')
                # 额外检查：确保所有参数都在CUDA上
                for param in module.parameters():
                    if not param.is_cuda:
                        param.data = param.data.cuda()
        return None

    # 为所有模块添加前向钩子
    def add_hooks_recursive(module):
        # 为所有模块添加钩子，不只是slayerSNN模块
        module.register_forward_pre_hook(cuda_fix_hook)

        for child in module.children():
            add_hooks_recursive(child)

    add_hooks_recursive(model)
    print("Enhanced CUDA fix hooks added to all modules")


def test_slayer_cuda_fix():
    """
    测试slayerSNN CUDA修复功能
    """
    print("Testing slayerSNN CUDA fix...")
    
    try:
        import sys
        import os
        
        # 添加项目路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.insert(0, project_root)
        
        from model import NetworkBasic
        import slayerSNN as snn
        
        # 创建测试模型
        netParams = snn.params(os.path.join(project_root, 'nMnist', 'network.yaml'))
        model = NetworkBasic(netParams)
        
        print("Original model created")
        
        # 检查初始状态
        print("\nChecking initial CUDA status...")
        initial_report = check_slayer_cuda_status(model)
        print(f"Initial report: {initial_report}")
        
        # 应用修复
        print("\nApplying CUDA fix...")
        model = fix_slayer_model(model, 'cuda')
        
        # 检查修复后状态
        print("\nChecking CUDA status after fix...")
        fixed_report = check_slayer_cuda_status(model)
        print(f"Fixed report: {fixed_report}")
        
        # 测试前向传播
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 2, 17, 17, 350).cuda()
        dummy_input = (dummy_input > 0).float()
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Forward pass successful! Output shape: {output.shape}")
        print("slayerSNN CUDA fix test PASSED!")
        return True
        
    except Exception as e:
        print(f"slayerSNN CUDA fix test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_slayer_cuda_fix()
