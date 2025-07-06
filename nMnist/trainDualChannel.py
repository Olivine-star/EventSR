"""
Training script for Dual-Channel EventSR model

This script trains the dual-channel architecture that combines
SNN and CNN pathways for enhanced event-based super-resolution.
"""

import sys
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../')
from dual_channel_model import DualChannelEventSR, create_dual_channel_model
from nMnist.mnistDatasetSR import mnistDatasetDualChannel
from utils.ckpt import checkpoint_restore, checkpoint_save
from opts import parser
from statistic import Metric
import slayerSNN as snn


class DualChannelLoss(nn.Module):
    """
    Combined loss function for dual-channel training.
    Includes losses for SNN output, CNN output, and fused output.
    """
    
    def __init__(self, 
                 snn_weight=1.0, 
                 cnn_weight=1.0, 
                 fusion_weight=2.0,
                 temporal_weight=5.0,
                 shape=[17, 17, 350]):
        super(DualChannelLoss, self).__init__()
        
        self.snn_weight = snn_weight
        self.cnn_weight = cnn_weight
        self.fusion_weight = fusion_weight
        self.temporal_weight = temporal_weight
        self.shape = shape
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        """
        Compute combined loss.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Target spike tensor
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Main fusion output loss
        fusion_loss = self.mse_loss(outputs['output'], targets)
        losses['fusion_loss'] = fusion_loss
        
        # SNN pathway loss
        if 'snn_output' in outputs:
            snn_loss = self.mse_loss(outputs['snn_output'], targets)
            losses['snn_loss'] = snn_loss
        else:
            losses['snn_loss'] = torch.tensor(0.0, device=targets.device)
        
        # CNN pathway loss (if available)
        if 'cnn_output' in outputs:
            # For CNN output, we might need to handle temporal dimension differently
            cnn_output = outputs['cnn_output']
            if len(cnn_output.shape) == 4 and len(targets.shape) == 5:
                # CNN output is 2D, target is 3D with time
                target_2d = targets.mean(dim=-1)  # Average over time
                cnn_loss = self.mse_loss(cnn_output, target_2d)
            else:
                cnn_loss = self.mse_loss(cnn_output, targets)
            losses['cnn_loss'] = cnn_loss
        else:
            losses['cnn_loss'] = torch.tensor(0.0, device=targets.device)
        
        # Temporal consistency loss (ECM loss from original implementation)
        temporal_loss = self._compute_temporal_loss(outputs['output'], targets)
        losses['temporal_loss'] = temporal_loss
        
        # Total loss
        total_loss = (self.fusion_weight * fusion_loss + 
                     self.snn_weight * losses['snn_loss'] + 
                     self.cnn_weight * losses['cnn_loss'] + 
                     self.temporal_weight * temporal_loss)
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_temporal_loss(self, output, target):
        """Compute temporal consistency loss (ECM loss)."""
        if len(output.shape) != 5:
            return torch.tensor(0.0, device=output.device)
        
        time_bins = self.shape[2] // 50  # 50ms time windows
        temporal_loss = 0
        
        for i in range(time_bins):
            start_t = i * 50
            end_t = (i + 1) * 50
            
            output_window = torch.sum(output[:, :, :, :, start_t:end_t], dim=4)
            target_window = torch.sum(target[:, :, :, :, start_t:end_t], dim=4)
            
            temporal_loss += self.mse_loss(output_window, target_window)
        
        return temporal_loss / time_bins if time_bins > 0 else torch.tensor(0.0, device=output.device)


def train_dual_channel():
    """Main training function for dual-channel model."""
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("Warning: CUDA not available. slayerSNN requires CUDA for training.")
        print("Please ensure CUDA is available and try again.")
        return

    print(f"Using device: {device}")
    print(f"CUDA devices: {args.cuda}")
    
    # Dataset configuration
    shape = [17, 17, 350]
    
    # Create datasets
    trainDataset = mnistDatasetDualChannel(
        train=True, 
        event_frame_strategy='time_based',
        num_event_frames=8
    )
    testDataset = mnistDatasetDualChannel(
        train=False,
        event_frame_strategy='time_based', 
        num_event_frames=8
    )
    
    print(f"Training samples: {len(trainDataset)}, Testing samples: {len(testDataset)}")
    
    # Create data loaders
    trainLoader = DataLoader(trainDataset, batch_size=args.bs, shuffle=True, num_workers=4)
    testLoader = DataLoader(testDataset, batch_size=args.bs, shuffle=False, num_workers=4)
    
    # Model configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    network_yaml_path = os.path.join(current_dir, 'network.yaml')
    netParams = snn.params(network_yaml_path)
    model_config = {
        'netParams': netParams,
        'snn_model_type': 'NetworkBasic',
        'cnn_base_channels': 64,
        'fusion_strategy': 'adaptive',
        'event_frame_strategy': 'time_based',
        'num_event_frames': 8,
        'scale_factor': 2,
        'use_temporal_cnn': False
    }
    
    # Create model
    model = create_dual_channel_model(model_config).to(device)

    # Apply CUDA fix immediately after moving to device
    from utils.slayer_cuda_fix import fix_slayer_model
    model.snn_channel = fix_slayer_model(model.snn_channel, device)
    print("Applied slayerSNN CUDA fix to training model")
    
    # Loss function
    criterion = DualChannelLoss(
        snn_weight=1.0,
        cnn_weight=1.0, 
        fusion_weight=2.0,
        temporal_weight=5.0,
        shape=shape
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    # Checkpoint and logging
    ckptPath = './ckpt_dual_channel/'
    os.makedirs(ckptPath, exist_ok=True)

    # Try to restore checkpoint, start from scratch if not found
    try:
        model, epoch0 = checkpoint_restore(model, ckptPath, name="ckptBest")
        print(f"Resumed training from epoch {epoch0 + 1}")
    except FileNotFoundError:
        print("No checkpoint found, starting training from scratch")
        epoch0 = -1
    tf_writer = SummaryWriter(log_dir=ckptPath)
    
    # Training parameters
    maxEpoch = args.epoch
    iter_per_epoch = len(trainLoader)
    
    print(f"Starting training from epoch {epoch0 + 1}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epoch0 + 1, maxEpoch):
        # Training phase
        model.train()
        train_metrics = Metric()
        
        for i, batch in enumerate(trainLoader):
            # Extract data from batch
            lr_spikes = batch['lr_spikes'].to(device)
            hr_spikes = batch['hr_spikes'].to(device)
            
            # Forward pass
            outputs = model(lr_spikes)
            
            # Compute losses
            losses = criterion(outputs, hr_spikes)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # Update metrics
            train_metrics.update(losses['total_loss'].item(), lr_spikes.size(0))
            
            # Logging
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}/{iter_per_epoch}, "
                      f"Total Loss: {losses['total_loss'].item():.6f}, "
                      f"Fusion Loss: {losses['fusion_loss'].item():.6f}, "
                      f"SNN Loss: {losses['snn_loss'].item():.6f}, "
                      f"CNN Loss: {losses['cnn_loss'].item():.6f}")
        
        # Validation phase
        model.eval()
        val_metrics = Metric()
        
        with torch.no_grad():
            for batch in testLoader:
                lr_spikes = batch['lr_spikes'].to(device)
                hr_spikes = batch['hr_spikes'].to(device)
                
                outputs = model(lr_spikes)
                losses = criterion(outputs, hr_spikes)
                
                val_metrics.update(losses['total_loss'].item(), lr_spikes.size(0))
        
        # Learning rate scheduling
        lr_scheduler.step()
        
        # Logging
        train_loss = train_metrics.avg
        val_loss = val_metrics.avg
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # TensorBoard logging
        tf_writer.add_scalar('Loss/Train', train_loss, epoch)
        tf_writer.add_scalar('Loss/Validation', val_loss, epoch)
        tf_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_save(model, ckptPath, epoch, name="ckptBest")
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Regular checkpoint
        if epoch % 10 == 0:
            checkpoint_save(model, ckptPath, epoch, name=f"ckpt_epoch_{epoch}")
    
    print("Training completed!")
    tf_writer.close()


if __name__ == '__main__':
    train_dual_channel()
