"""
简化的双通道EventSR训练启动脚本

这个脚本提供了一个简单的方式来启动双通道模型的训练，
包含了必要的错误检查和配置。
"""

import sys
import os
import torch
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
print(f"项目根目录: {project_root}")
print(f"当前目录: {current_dir}")

def check_environment():
    """检查训练环境"""
    print("="*60)
    print("环境检查")
    print("="*60)
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
    else:
        print("警告: CUDA不可用，slayerSNN需要CUDA支持")
        return False
    
    # 检查数据集
    try:
        from nMnist.mnistDatasetSR import mnistDatasetDualChannel
        dataset = mnistDatasetDualChannel(train=True, event_frame_strategy='time_based', num_event_frames=8)
        print(f"训练数据集大小: {len(dataset)}")
        
        test_dataset = mnistDatasetDualChannel(train=False, event_frame_strategy='time_based', num_event_frames=8)
        print(f"测试数据集大小: {len(test_dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print("数据样本检查通过")
        for key, tensor in sample.items():
            print(f"  {key}: {tensor.shape}")
            
    except Exception as e:
        print(f"数据集检查失败: {e}")
        return False
    
    # 检查模型创建
    try:
        from dual_channel_model import create_dual_channel_model
        import slayerSNN as snn
        
        netParams = snn.params(os.path.join(current_dir, 'network.yaml'))
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
        
        model = create_dual_channel_model(model_config)
        model = model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型创建成功，参数数量: {total_params:,}")
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        return False
    
    print("环境检查通过！")
    return True


def start_training(batch_size=8, learning_rate=0.001, epochs=50, cuda_device="0"):
    """启动训练"""
    
    print("\n" + "="*60)
    print("开始训练双通道EventSR模型")
    print("="*60)
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    # 导入训练函数
    from nMnist.trainDualChannel import train_dual_channel
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.bs = batch_size
            self.lr = learning_rate
            self.epoch = epochs
            self.cuda = cuda_device
            self.showFreq = 10
            self.savepath = './ckpt_dual_channel/'
            self.add = None
            self.j = 4  # 减少worker数量
    
    # 替换sys.argv以模拟命令行参数
    original_argv = sys.argv
    sys.argv = ['trainDualChannel.py']
    
    # 创建参数对象
    import opts
    opts.parser.parse_args = lambda: Args()
    
    try:
        # 开始训练
        train_dual_channel()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始argv
        sys.argv = original_argv


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双通道EventSR训练启动器')
    parser.add_argument('--bs', type=int, default=8, help='批次大小 (默认: 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    parser.add_argument('--epoch', type=int, default=50, help='训练轮数 (默认: 50)')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA设备 (默认: 0)')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境，不开始训练')
    
    args = parser.parse_args()
    
    print("双通道EventSR训练启动器")
    print(f"批次大小: {args.bs}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epoch}")
    print(f"CUDA设备: {args.cuda}")
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请解决上述问题后重试")
        return
    
    if args.check_only:
        print("\n仅进行环境检查，训练未启动")
        return
    
    # 用户确认
    print(f"\n准备开始训练...")
    print(f"配置:")
    print(f"  批次大小: {args.bs}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epoch}")
    print(f"  CUDA设备: {args.cuda}")
    
    response = input("\n是否开始训练? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("训练取消")
        return
    
    # 开始训练
    start_training(
        batch_size=args.bs,
        learning_rate=args.lr,
        epochs=args.epoch,
        cuda_device=args.cuda
    )


if __name__ == '__main__':
    main()
