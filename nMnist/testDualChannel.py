"""
Testing script for Dual-Channel EventSR model

This script evaluates the dual-channel architecture and compares
it against single-channel baselines.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append('../')
from dual_channel_model import DualChannelEventSR, create_dual_channel_model
from model import NetworkBasic
from nMnist.mnistDatasetSR import mnistDatasetDualChannel, mnistDataset
from utils.ckpt import checkpoint_restore
from utils.config_manager import ConfigManager
from utils.utils import getEventFromTensor
import slayerSNN as snn


class DualChannelEvaluator:
    """Evaluator for dual-channel EventSR model."""
    
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda'):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to run evaluation on
        """
        self.device = device
        self.config_manager = ConfigManager(config_path)
        
        # Load dual-channel model
        model_config = self.config_manager.get_model_config()
        self.dual_model = create_dual_channel_model(model_config).to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.dual_model, _ = checkpoint_restore(
                self.dual_model, 
                os.path.dirname(checkpoint_path), 
                name=os.path.basename(checkpoint_path).replace('.pth', '')
            )
        
        # Load baseline SNN model for comparison
        netParams = snn.params('../nMnist/network.yaml')
        self.baseline_model = NetworkBasic(netParams).to(device)
        
        # Try to load baseline checkpoint
        baseline_ckpt_path = '../nMnist/ckpt/'
        if os.path.exists(baseline_ckpt_path):
            try:
                self.baseline_model, _ = checkpoint_restore(
                    self.baseline_model, baseline_ckpt_path, name="ckptBest"
                )
            except:
                print("Warning: Could not load baseline model checkpoint")
        
        self.dual_model.eval()
        self.baseline_model.eval()
    
    def evaluate_model(self, test_loader, model_name="dual_channel"):
        """
        Evaluate a model on test data.
        
        Args:
            test_loader: Test data loader
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mse': [],
            'psnr': [],
            'ssim': [],
            'temporal_consistency': []
        }
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if isinstance(batch, dict):
                    # Dual-channel dataset
                    lr_spikes = batch['lr_spikes'].to(self.device)
                    hr_spikes = batch['hr_spikes'].to(self.device)
                else:
                    # Regular dataset
                    lr_spikes, hr_spikes = batch
                    lr_spikes = lr_spikes.to(self.device)
                    hr_spikes = hr_spikes.to(self.device)
                
                # Forward pass
                if model_name == "dual_channel":
                    outputs = self.dual_model(lr_spikes)
                    predictions = outputs['output']
                else:
                    predictions = self.baseline_model(lr_spikes)
                
                # Compute metrics for each sample in batch
                batch_metrics = self._compute_batch_metrics(predictions, hr_spikes)
                
                for key in metrics:
                    metrics[key].extend(batch_metrics[key])
                
                if i % 10 == 0:
                    print(f"Processed {i+1} batches for {model_name}")
        
        # Compute average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        std_metrics = {key: np.std(values) for key, values in metrics.items()}
        
        return avg_metrics, std_metrics, metrics
    
    def _compute_batch_metrics(self, predictions, targets):
        """Compute metrics for a batch of predictions."""
        batch_size = predictions.shape[0]
        batch_metrics = {
            'mse': [],
            'psnr': [],
            'ssim': [],
            'temporal_consistency': []
        }
        
        for b in range(batch_size):
            pred = predictions[b].cpu().numpy()
            target = targets[b].cpu().numpy()
            
            # MSE
            mse = mean_squared_error(target.flatten(), pred.flatten())
            batch_metrics['mse'].append(mse)
            
            # PSNR and SSIM (computed on spatial frames)
            if len(pred.shape) == 4:  # [C, H, W, T]
                # Average over time and channels for PSNR/SSIM
                pred_2d = np.mean(pred, axis=(0, -1))  # [H, W]
                target_2d = np.mean(target, axis=(0, -1))  # [H, W]
            else:  # [C, H, W]
                pred_2d = np.mean(pred, axis=0)  # [H, W]
                target_2d = np.mean(target, axis=0)  # [H, W]
            
            # Normalize to [0, 1] for PSNR/SSIM
            pred_norm = (pred_2d - pred_2d.min()) / (pred_2d.max() - pred_2d.min() + 1e-8)
            target_norm = (target_2d - target_2d.min()) / (target_2d.max() - target_2d.min() + 1e-8)
            
            psnr = peak_signal_noise_ratio(target_norm, pred_norm, data_range=1.0)
            ssim = structural_similarity(target_norm, pred_norm, data_range=1.0)
            
            batch_metrics['psnr'].append(psnr)
            batch_metrics['ssim'].append(ssim)
            
            # Temporal consistency (if temporal dimension exists)
            if len(pred.shape) == 4:  # [C, H, W, T]
                temporal_consistency = self._compute_temporal_consistency(pred, target)
                batch_metrics['temporal_consistency'].append(temporal_consistency)
            else:
                batch_metrics['temporal_consistency'].append(0.0)
        
        return batch_metrics
    
    def _compute_temporal_consistency(self, pred, target):
        """Compute temporal consistency metric."""
        # Compute frame-to-frame differences
        pred_diff = np.diff(pred, axis=-1)  # [C, H, W, T-1]
        target_diff = np.diff(target, axis=-1)  # [C, H, W, T-1]
        
        # MSE of temporal differences
        temporal_mse = mean_squared_error(target_diff.flatten(), pred_diff.flatten())
        
        return 1.0 / (1.0 + temporal_mse)  # Higher is better
    
    def compare_models(self, test_dataset):
        """
        Compare dual-channel model against baseline.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Comparison results
        """
        print("Evaluating Dual-Channel Model...")
        
        # Create dual-channel data loader
        dual_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        dual_metrics, dual_std, _ = self.evaluate_model(dual_loader, "dual_channel")
        
        print("Evaluating Baseline SNN Model...")
        
        # Create baseline dataset and loader
        baseline_dataset = mnistDataset(train=False)
        baseline_loader = DataLoader(baseline_dataset, batch_size=8, shuffle=False, num_workers=2)
        baseline_metrics, baseline_std, _ = self.evaluate_model(baseline_loader, "baseline")
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for metric in ['mse', 'psnr', 'ssim', 'temporal_consistency']:
            dual_val = dual_metrics[metric]
            dual_std_val = dual_std[metric]
            baseline_val = baseline_metrics[metric]
            baseline_std_val = baseline_std[metric]
            
            improvement = ((dual_val - baseline_val) / baseline_val) * 100
            if metric == 'mse':  # Lower is better for MSE
                improvement = -improvement
            
            print(f"{metric.upper()}:")
            print(f"  Dual-Channel: {dual_val:.6f} ± {dual_std_val:.6f}")
            print(f"  Baseline SNN: {baseline_val:.6f} ± {baseline_std_val:.6f}")
            print(f"  Improvement: {improvement:+.2f}%")
            print()
        
        return {
            'dual_channel': dual_metrics,
            'baseline': baseline_metrics,
            'dual_channel_std': dual_std,
            'baseline_std': baseline_std
        }
    
    def analyze_fusion_strategies(self, test_dataset):
        """
        Analyze different fusion strategies.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Results for different fusion strategies
        """
        fusion_strategies = ['adaptive', 'cross_attention', 'concatenation', 'element_wise']
        results = {}
        
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        for strategy in fusion_strategies:
            print(f"Testing fusion strategy: {strategy}")
            
            # Update model fusion strategy
            self.dual_model.set_fusion_strategy(strategy)
            
            # Evaluate
            metrics, std_metrics, _ = self.evaluate_model(test_loader, f"dual_channel_{strategy}")
            
            results[strategy] = {
                'metrics': metrics,
                'std': std_metrics
            }
        
        # Print results
        print("\n" + "="*60)
        print("FUSION STRATEGY COMPARISON")
        print("="*60)
        
        for metric in ['mse', 'psnr', 'ssim', 'temporal_consistency']:
            print(f"\n{metric.upper()}:")
            for strategy in fusion_strategies:
                val = results[strategy]['metrics'][metric]
                std_val = results[strategy]['std'][metric]
                print(f"  {strategy:15}: {val:.6f} ± {std_val:.6f}")
        
        return results
    
    def visualize_results(self, test_dataset, save_dir="./results/"):
        """
        Visualize model outputs.
        
        Args:
            test_dataset: Test dataset
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 5:  # Only visualize first 5 samples
                    break
                
                lr_spikes = batch['lr_spikes'].to(self.device)
                hr_spikes = batch['hr_spikes'].to(self.device)
                
                # Get predictions
                dual_outputs = self.dual_model(lr_spikes)
                dual_pred = dual_outputs['output']
                snn_pred = dual_outputs['snn_output']
                cnn_pred = dual_outputs['cnn_output']
                
                # Convert to numpy
                lr_np = lr_spikes[0].cpu().numpy()
                hr_np = hr_spikes[0].cpu().numpy()
                dual_np = dual_pred[0].cpu().numpy()
                snn_np = snn_pred[0].cpu().numpy()
                
                # Create visualization
                self._create_visualization(
                    lr_np, hr_np, dual_np, snn_np, 
                    save_path=os.path.join(save_dir, f"sample_{i}.png")
                )
        
        print(f"Visualizations saved to {save_dir}")
    
    def _create_visualization(self, lr, hr, dual_pred, snn_pred, save_path):
        """Create and save visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Average over time and channels for visualization
        if len(lr.shape) == 4:  # [C, H, W, T]
            lr_vis = np.mean(lr, axis=(0, -1))
            hr_vis = np.mean(hr, axis=(0, -1))
            dual_vis = np.mean(dual_pred, axis=(0, -1))
            snn_vis = np.mean(snn_pred, axis=(0, -1))
        else:  # [C, H, W]
            lr_vis = np.mean(lr, axis=0)
            hr_vis = np.mean(hr, axis=0)
            dual_vis = np.mean(dual_pred, axis=0)
            snn_vis = np.mean(snn_pred, axis=0)
        
        # Plot images
        axes[0, 0].imshow(lr_vis, cmap='gray')
        axes[0, 0].set_title('Low Resolution Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(hr_vis, cmap='gray')
        axes[0, 1].set_title('High Resolution Target')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(dual_vis, cmap='gray')
        axes[0, 2].set_title('Dual-Channel Prediction')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(snn_vis, cmap='gray')
        axes[1, 0].set_title('SNN-Only Prediction')
        axes[1, 0].axis('off')
        
        # Error maps
        dual_error = np.abs(hr_vis - dual_vis)
        snn_error = np.abs(hr_vis - snn_vis)
        
        im1 = axes[1, 1].imshow(dual_error, cmap='hot')
        axes[1, 1].set_title('Dual-Channel Error')
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1])
        
        im2 = axes[1, 2].imshow(snn_error, cmap='hot')
        axes[1, 2].set_title('SNN-Only Error')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main testing function."""
    # Configuration
    config_path = "../configs/dual_channel_config.yaml"
    checkpoint_path = "../nMnist/ckpt_dual_channel/ckptBest.pth"
    
    # Create evaluator
    evaluator = DualChannelEvaluator(config_path, checkpoint_path)
    
    # Create test dataset
    test_dataset = mnistDatasetDualChannel(
        train=False,
        event_frame_strategy='time_based',
        num_event_frames=8
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Compare models
    comparison_results = evaluator.compare_models(test_dataset)
    
    # Analyze fusion strategies
    fusion_results = evaluator.analyze_fusion_strategies(test_dataset)
    
    # Create visualizations
    evaluator.visualize_results(test_dataset)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
