"""
Ensemble Predictor for Flu Forecasting
Combines DLinear, PatchTST, and Llama2 models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from data_provider.data_factory import data_provider
from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.Llama2 import Llama2
from utils.metrics import metric
from utils.epidemic_metrics import epidemic_metrics, print_epidemic_metrics
from utils.post_processing import clip_predictions, analyze_clipping_impact, print_clipping_analysis


class EnsemblePredictor:
    """
    Ensemble của 3 models: DLinear, PatchTST, Llama2
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.models = {}
        self.checkpoints = {}
        
        # Parse weights
        weights = [float(w) for w in args.ensemble_weights.split(',')]
        assert len(weights) == 3, "Must provide 3 weights for DLinear, PatchTST, Llama2"
        assert abs(sum(weights) - 1.0) < 0.01, f"Weights must sum to 1.0, got {sum(weights)}"
        
        self.weights = {
            'DLinear': weights[0],
            'PatchTST': weights[1],
            'Llama2': weights[2]
        }
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE PREDICTOR")
        print(f"{'='*70}")
        print(f"Weights: DLinear={self.weights['DLinear']:.2f}, "
              f"PatchTST={self.weights['PatchTST']:.2f}, "
              f"Llama2={self.weights['Llama2']:.2f}")
        print(f"{'='*70}\n")
    
    def load_model(self, model_type, checkpoint_path):
        """Load a single model from checkpoint"""
        
        print(f"\n[*] Loading {model_type} from {checkpoint_path}")
        
        # Create model-specific config based on model type
        if model_type in ['DLinear', 'PatchTST']:
            # DLinear and PatchTST use d_model=768, d_ff=512
            model_args = argparse.Namespace(**vars(self.args))
            model_args.d_model = 768
            model_args.d_ff = 512
        elif model_type == 'Llama2':
            # Llama2 uses d_model=4096, d_ff=2048 from self.args (already set)
            model_args = self.args
        else:
            model_args = self.args
        
        # Create model instance (all models need device parameter)
        if model_type == 'DLinear':
            model = DLinear(model_args, self.device)
        elif model_type == 'PatchTST':
            model = PatchTST(model_args, self.device)
        elif model_type == 'Llama2':
            model = Llama2(model_args, self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        
        print(f"[OK] {model_type} loaded successfully (d_model={model_args.d_model})")
        
        self.models[model_type] = model
        self.checkpoints[model_type] = checkpoint_path
    
    def load_all_models(self, checkpoint_paths):
        """
        Load all 3 models
        
        Args:
            checkpoint_paths: dict with keys 'DLinear', 'PatchTST', 'Llama2'
        """
        for model_type, path in checkpoint_paths.items():
            if self.weights[model_type] > 0:  # Only load if weight > 0
                self.load_model(model_type, path)
    
    def predict_single(self, model_type, batch_x, batch_x_mark, batch_y_mark):
        """Get prediction from a single model"""
        
        model = self.models[model_type]
        
        with torch.no_grad():
            batch_x = batch_x.float().to(self.device)
            
            if model_type in ['Llama2', 'DLinear', 'PatchTST']:
                # These models use itr parameter and don't need marks
                outputs = model(batch_x, itr=0)
            else:
                # Other models may need marks
                if batch_x_mark is not None:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                if batch_y_mark is not None:
                    batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Create dummy dec_inp for encoder-decoder models
                dec_inp = torch.zeros([batch_x.shape[0], self.args.pred_len, batch_x.shape[-1]]).float().to(self.device)
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Extract predictions (handle different output formats)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Ensure correct shape [B, pred_len, features]
            if outputs.dim() == 2:
                outputs = outputs.unsqueeze(-1)
        
        return outputs
    
    def predict_ensemble(self, batch_x, batch_x_mark, batch_y_mark):
        """
        Get ensemble prediction by weighted average
        
        Returns:
            ensemble_pred: [B, pred_len, features]
        """
        
        predictions = {}
        
        # Get predictions from each model
        for model_type in self.models.keys():
            if self.weights[model_type] > 0:
                pred = self.predict_single(model_type, batch_x, batch_x_mark, batch_y_mark)
                predictions[model_type] = pred
        
        # Weighted average
        ensemble_pred = None
        for model_type, pred in predictions.items():
            weight = self.weights[model_type]
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def evaluate(self, test_loader):
        """
        Evaluate ensemble on test set
        
        Returns:
            metrics: dict with all evaluation metrics
        """
        
        print(f"\n{'='*70}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*70}\n")
        
        preds = []
        trues = []
        
        for i, batch in enumerate(test_loader):
            # Handle different batch formats
            if len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            elif len(batch) == 2:
                batch_x, batch_y = batch
                batch_x_mark, batch_y_mark = None, None
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            batch_y = batch_y.float().to(self.device)
            
            # Get ensemble prediction
            outputs = self.predict_ensemble(batch_x, batch_x_mark, batch_y_mark)
            
            # Extract ground truth
            if self.args.features == 'MS':
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:]
            else:
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
        
        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print(f"Predictions shape: {preds.shape}")
        print(f"Ground truth shape: {trues.shape}")
        
        # Reshape for metrics
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # Clipping analysis
        print(f"\n{'='*70}")
        print("POST-PROCESSING: CLIPPING ANALYSIS")
        print(f"{'='*70}\n")
        
        preds_original = preds.copy()
        preds_clipped = clip_predictions(preds, min_value=0.0)
        
        clipping_analysis = analyze_clipping_impact(preds_original, preds_clipped, trues)
        print_clipping_analysis(clipping_analysis)
        
        # Standard metrics - BEFORE clipping
        print(f"\n{'='*70}")
        print("STANDARD METRICS - BEFORE CLIPPING")
        print(f"{'='*70}\n")
        
        mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig, smape_orig, nd_orig = metric(preds_original, trues)
        print(f'mse:{mse_orig:.6f}, mae:{mae_orig:.6f}, rmse:{rmse_orig:.6f}')
        print(f'mape:{mape_orig:.6f}, smape:{smape_orig:.6f}')
        
        # Standard metrics - AFTER clipping
        print(f"\n{'='*70}")
        print("STANDARD METRICS - AFTER CLIPPING (RECOMMENDED)")
        print(f"{'='*70}\n")
        
        mae, mse, rmse, mape, mspe, smape, nd = metric(preds_clipped, trues)
        print(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}')
        print(f'mape:{mape:.6f}, smape:{smape:.6f}')
        
        # Epidemic metrics
        print(f"\n{'='*70}")
        print("EPIDEMIC-SPECIFIC METRICS")
        print(f"{'='*70}\n")
        
        epidemic_results = epidemic_metrics(preds_clipped, trues)
        print_epidemic_metrics(epidemic_results)
        
        # Compute Spearman and Pearson correlation
        from scipy.stats import spearmanr, pearsonr
        
        preds_flat = preds_clipped.reshape(-1)
        trues_flat = trues.reshape(-1)
        
        spearman_r, _ = spearmanr(preds_flat, trues_flat)
        pearson_r, _ = pearsonr(preds_flat, trues_flat)
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"spearmanR: {spearman_r:.4f}, pearsonR: {pearson_r:.4f}")
        print(f"mae: {mae:.4f}, mse: {mse:.4f}, mape: {mape:.4f}, smape: {smape:.4f}")
        print(f"Epidemic Metrics:")
        print(f"  Peak Timing Error: {epidemic_results['peak_timing_error_weeks']:.2f} weeks")
        print(f"  Peak Intensity Error: {epidemic_results['peak_intensity_abs_error']:.2f}%")
        print(f"  Direction Accuracy: {epidemic_results['direction_accuracy_pct']:.1f}%")
        print(f"{'='*70}\n")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'spearman_r': spearman_r,
            'pearson_r': pearson_r,
            **epidemic_results
        }


def find_best_checkpoint(base_path, model_name):
    """
    Find best checkpoint for a model (choose itr0 by default)
    
    Args:
        base_path: checkpoints directory
        model_name: 'DLinear_optimized', 'PatchTST_fast', 'Llama2_raw_augmented'
    
    Returns:
        checkpoint_path: path to checkpoint.pth
    """
    # Search pattern
    pattern = f"{model_name}_sl*_itr0"  # Always use iteration 0
    
    matching_dirs = [d for d in os.listdir(base_path) if d.startswith(model_name) and 'itr0' in d]
    
    if not matching_dirs:
        raise FileNotFoundError(f"No checkpoint found for {model_name}")
    
    # Use first match
    checkpoint_dir = os.path.join(base_path, matching_dirs[0])
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    
    # Basic configs
    parser.add_argument('--model', type=str, default='ensemble')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/')
    
    # Model names
    parser.add_argument('--dlinear_name', type=str, default='DLinear_optimized')
    parser.add_argument('--patchtst_name', type=str, default='PatchTST_fast')
    parser.add_argument('--llama2_name', type=str, default='Llama2_raw_augmented')
    
    # Ensemble weights
    parser.add_argument('--ensemble_weights', type=str, default='0.3,0.3,0.4',
                        help='Weights for DLinear,PatchTST,Llama2 (must sum to 1.0)')
    
    # Data configs
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='Flu_in_NorthChina.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='positive_rate')
    parser.add_argument('--freq', type=int, default=0)
    parser.add_argument('--percent', type=int, default=100)
    
    # Model configs
    parser.add_argument('--seq_len', type=int, default=52)
    parser.add_argument('--label_len', type=int, default=18)
    parser.add_argument('--pred_len', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=int, default=0.3)
    
    # PatchTST specific
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    
    # Llama2 specific
    parser.add_argument('--llama_layers', type=int, default=8)
    parser.add_argument('--fc_layer', type=int, default=256)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--is_gpt', type=int, default=1)
    
    # Other
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    print(f"\n{'='*70}")
    print("FINDING CHECKPOINTS")
    print(f"{'='*70}\n")
    
    checkpoint_paths = {}
    
    try:
        checkpoint_paths['DLinear'] = find_best_checkpoint(args.checkpoints_dir, args.dlinear_name)
        print(f"[OK] DLinear: {checkpoint_paths['DLinear']}")
    except FileNotFoundError as e:
        print(f"[!] DLinear not found: {e}")
        checkpoint_paths['DLinear'] = None
    
    try:
        checkpoint_paths['PatchTST'] = find_best_checkpoint(args.checkpoints_dir, args.patchtst_name)
        print(f"[OK] PatchTST: {checkpoint_paths['PatchTST']}")
    except FileNotFoundError as e:
        print(f"[!] PatchTST not found: {e}")
        checkpoint_paths['PatchTST'] = None
    
    try:
        checkpoint_paths['Llama2'] = find_best_checkpoint(args.checkpoints_dir, args.llama2_name)
        print(f"[OK] Llama2: {checkpoint_paths['Llama2']}")
    except FileNotFoundError as e:
        print(f"[!] Llama2 not found: {e}")
        checkpoint_paths['Llama2'] = None
    
    # Filter out None values
    checkpoint_paths = {k: v for k, v in checkpoint_paths.items() if v is not None}
    
    if len(checkpoint_paths) == 0:
        raise RuntimeError("No checkpoints found! Please train models first.")
    
    # Create ensemble
    ensemble = EnsemblePredictor(args, device)
    
    # Load models
    ensemble.load_all_models(checkpoint_paths)
    
    # Load test data
    print(f"\n{'='*70}")
    print("LOADING TEST DATA")
    print(f"{'='*70}\n")
    
    test_data, test_loader_raw = data_provider(args, 'test')
    print(f"Test samples: {len(test_data)}")
    
    # Create new dataloader with custom collate to handle object dtype
    def custom_collate(batch):
        """Custom collate to convert object dtypes to tensors"""
        # Handle different number of return values
        if len(batch[0]) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = zip(*batch)
        elif len(batch[0]) == 5:
            # Some datasets return 5 values
            batch_x, batch_y, batch_x_mark, batch_y_mark, _ = zip(*batch)
        elif len(batch[0]) == 2:
            batch_x, batch_y = zip(*batch)
            batch_x_mark, batch_y_mark = None, None
        else:
            raise ValueError(f"Unexpected batch format: {len(batch[0])} values per sample")
        
        # Helper to convert to tensor, handling object dtype
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            elif isinstance(x, np.ndarray):
                # Convert object dtype to float64 first
                if x.dtype == object:
                    x = x.astype(np.float64)
                return torch.from_numpy(x).float()
            else:
                return torch.FloatTensor(x)
        
        # Convert to tensors
        batch_x = torch.stack([to_tensor(x) for x in batch_x])
        batch_y = torch.stack([to_tensor(y) for y in batch_y])
        
        if batch_x_mark is not None and batch_x_mark[0] is not None:
            batch_x_mark = torch.stack([to_tensor(x) for x in batch_x_mark])
        else:
            batch_x_mark = None
            
        if batch_y_mark is not None and batch_y_mark[0] is not None:
            batch_y_mark = torch.stack([to_tensor(y) for y in batch_y_mark])
        else:
            batch_y_mark = None
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=custom_collate
    )
    
    # Evaluate
    metrics = ensemble.evaluate(test_loader)
    
    print("\n[OK] Ensemble evaluation completed!")


if __name__ == '__main__':
    main()
