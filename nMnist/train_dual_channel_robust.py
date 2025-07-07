"""
稳健的双通道EventSR训练脚本

这个脚本专门处理slayerSNN的CUDA兼容性问题，
使用多种策略确保训练能够正常进行。
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dual_channel_model import create_dual_channel_model
from mnistDatasetSR import mnistDatasetDualChannel
from utils.slayer_cuda_fix import fix_slayer_model
import slayerSNN as snn


class RobustDualChannelLoss(nn.Module):
    """稳健的双通道损失函数"""
    
    def __init__(self):
        super(RobustDualChannelLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        # 主要使用融合输出的损失
        main_loss = self.mse_loss(outputs['output'], targets)
        
        # 添加SNN输出损失（如果可用）
        snn_loss = 0
        if 'snn_output' in outputs:
            try:
                snn_loss = self.mse_loss(outputs['snn_output'], targets) * 0.5
            except:
                snn_loss = 0
        
        # 添加CNN输出损失（如果可用）
        cnn_loss = 0
        if 'cnn_output' in outputs:
            try:
                cnn_output = outputs['cnn_output']
                if len(cnn_output.shape) == 4 and len(targets.shape) == 5:
                    target_2d = targets.mean(dim=-1)
                    cnn_loss = self.mse_loss(cnn_output, target_2d) * 0.3
                else:
                    cnn_loss = self.mse_loss(cnn_output, targets) * 0.3
            except:
                cnn_loss = 0
        
        total_loss = main_loss + snn_loss + cnn_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'snn_loss': snn_loss if isinstance(snn_loss, torch.Tensor) else torch.tensor(snn_loss),
            'cnn_loss': cnn_loss if isinstance(cnn_loss, torch.Tensor) else torch.tensor(cnn_loss)
        }


def save_model_checkpoint(model, optimizer, epoch, avg_loss, save_dir):
    """
    保存模型检查点

    Args:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前epoch
        avg_loss: 平均损失
        save_dir: 保存目录

    Returns:
        保存的文件路径
    """
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 创建文件名
    filename = f"dual_channel_epoch_{epoch}_loss_{avg_loss:.6f}.pth"
    filepath = os.path.join(save_dir, filename)

    # 准备保存的数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'model_config': {
            'model_type': 'dual_channel_eventsr',
            'parameters': sum(p.numel() for p in model.parameters())
        }
    }

    # 保存检查点
    torch.save(checkpoint, filepath)

    # 打印保存路径
    full_path = os.path.abspath(filepath)
    print(f"Model checkpoint saved to: {full_path}")

    return full_path


def robust_model_forward(model, input_tensor, max_retries=3):
    """稳健的模型前向传播，包含重试机制"""
    
    for attempt in range(max_retries):
        try:
            # 在每次尝试前都检查并修复CUDA状态
            if input_tensor.is_cuda:
                fix_slayer_model(model.snn_channel, 'cuda')
            
            # 尝试前向传播
            outputs = model(input_tensor)
            return outputs, None
            
        except RuntimeError as e:
            if "filter must be a CUDA tensor" in str(e) and attempt < max_retries - 1:
                print(f"CUDA error on attempt {attempt + 1}, retrying...")
                
                # 强制修复所有CUDA相关问题
                model = model.cuda()
                fix_slayer_model(model.snn_channel, 'cuda')
                
                # 确保输入也在正确的设备上
                input_tensor = input_tensor.cuda()
                
                continue
            else:
                return None, e
    
    return None, Exception("Max retries exceeded")


def train_robust_dual_channel():
    """稳健的双通道训练函数"""
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device')
    parser.add_argument('--j', type=int, default=2, help='Number of workers')
    parser.add_argument('--savepath', type=str, default='./ckpt_dual_channel_robust/', 
                        help='Path to save model checkpoints')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.savepath, exist_ok=True)
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # 创建数据集
    try:
        train_dataset = mnistDatasetDualChannel(
            train=True,
            event_frame_strategy='time_based',
            num_event_frames=8
        )
        test_dataset = mnistDatasetDualChannel(
            train=False,
            event_frame_strategy='time_based',
            num_event_frames=8
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.bs, 
        shuffle=True, 
        num_workers=args.j
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.j
    )
    
    # 创建模型
    try:
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
        
        model = create_dual_channel_model(model_config)
        model = model.to(device)
        
        # 立即应用CUDA修复
        if device == 'cuda':
            fix_slayer_model(model.snn_channel, device)
            print("Applied initial CUDA fix")
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    except Exception as e:
        print(f"Model creation failed: {e}")
        return
    
    # 创建损失函数和优化器
    criterion = RobustDualChannelLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 初始化训练变量
    start_epoch = 0
    best_val_loss = float('inf')

    # 打印保存目录信息
    print(f"Model checkpoints will be saved to: {os.path.abspath(args.savepath)}")

    # 训练循环
    print(f"Starting training for {args.epoch} epochs...")

    for epoch in range(start_epoch, args.epoch):
        model.train()
        epoch_loss = 0
        successful_batches = 0
        failed_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 获取数据
                lr_spikes = batch['lr_spikes'].to(device)
                hr_spikes = batch['hr_spikes'].to(device)
                
                # 稳健的前向传播
                outputs, error = robust_model_forward(model, lr_spikes)
                
                if outputs is None:
                    print(f"Batch {batch_idx} failed: {error}")
                    failed_batches += 1
                    continue
                
                # 计算损失
                losses = criterion(outputs, hr_spikes)
                
                # 反向传播
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
                
                epoch_loss += losses['total_loss'].item()
                successful_batches += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {losses['total_loss'].item():.6f}, "
                          f"Success: {successful_batches}, Failed: {failed_batches}")
                
            except Exception as e:
                print(f"Batch {batch_idx} error: {e}")
                failed_batches += 1
                continue
        
        # 计算平均损失
        if successful_batches > 0:
            avg_loss = epoch_loss / successful_batches
            print(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.6f}, "
                  f"Success Rate = {successful_batches}/{successful_batches + failed_batches}")
        else:
            print(f"Epoch {epoch} failed: No successful batches")
            break
        
        # 验证阶段
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        lr_spikes = batch['lr_spikes'].to(device)
                        hr_spikes = batch['hr_spikes'].to(device)
                        
                        outputs, error = robust_model_forward(model, lr_spikes)
                        if outputs is not None:
                            losses = criterion(outputs, hr_spikes)
                            val_loss += losses['total_loss'].item()
                            val_batches += 1
                        
                        if val_batches >= 10:  # 只验证前10个批次
                            break
                            
                    except Exception as e:
                        continue
            
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                print(f"Validation Loss: {avg_val_loss:.6f}")

                # 保存模型检查点
                save_model_checkpoint(model, optimizer, epoch, avg_loss, args.savepath)

                # 如果是最佳模型，则保存为最佳检查点
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_filename = f"dual_channel_BEST_epoch_{epoch}_loss_{avg_val_loss:.6f}.pth"
                    best_filepath = os.path.join(args.savepath, best_filename)

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_loss,
                        'val_loss': avg_val_loss,
                        'model_config': {
                            'model_type': 'dual_channel_eventsr',
                            'parameters': sum(p.numel() for p in model.parameters())
                        }
                    }
                    torch.save(checkpoint, best_filepath)
                    print(f"Best model saved to: {os.path.abspath(best_filepath)}")
                    print(f"New best validation loss: {best_val_loss:.6f}")

        # 如果不是验证epoch，仍然保存常规检查点
        else:
            save_model_checkpoint(model, optimizer, epoch, avg_loss, args.savepath)
    
    # 保存最终模型
    final_filename = f"dual_channel_FINAL_epoch_{args.epoch-1}.pth"
    final_filepath = os.path.join(args.savepath, final_filename)

    final_checkpoint = {
        'epoch': args.epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'model_type': 'dual_channel_eventsr',
            'parameters': sum(p.numel() for p in model.parameters())
        }
    }
    torch.save(final_checkpoint, final_filepath)
    print(f"Final model saved to: {os.path.abspath(final_filepath)}")
    print(f"Training completed! All models saved to: {os.path.abspath(args.savepath)}")


if __name__ == '__main__':
    train_robust_dual_channel()


