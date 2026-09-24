"""
Script để visualize kết quả dự đoán
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import argparse

def load_predictions(model_id, pred_dir='./predictions/'):
    """Load saved predictions"""
    pred_file = os.path.join(pred_dir, f"{model_id}_pred.npy")
    true_file = os.path.join(pred_dir, f"{model_id}_true.npy")
    date_file = os.path.join(pred_dir, f"{model_id}_dates.npy")
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Không tìm thấy file: {pred_file}\nHãy chạy save_predictions.py trước!")
    
    preds = np.load(pred_file)
    trues = np.load(true_file)
    dates = np.load(date_file)
    
    print(f"✅ Loaded predictions:")
    print(f"   Shape: {preds.shape}")
    print(f"   Samples: {preds.shape[0]}")
    print(f"   Pred length: {preds.shape[1]}")
    
    return preds, trues, dates

def plot_comparison(preds, trues, dates, model_id, output_dir='./figures/', num_samples=5):
    """Plot comparison between predictions and ground truth"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples to plot (evenly spaced)
    num_total = preds.shape[0]
    if num_samples > num_total:
        num_samples = num_total
    
    indices = np.linspace(0, num_total-1, num_samples, dtype=int)
    
    # Plot each sample
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        
        pred = preds[sample_idx, :, 0]
        true = trues[sample_idx, :, 0]
        
        # Convert dates
        date_arr = dates[sample_idx, -len(pred):, :]
        date_strs = [f"{int(y)}-{int(m):02d}-{int(d):02d}" 
                     for y, m, d in date_arr.astype(int)]
        
        weeks = np.arange(1, len(pred) + 1)
        
        # Calculate metrics for this sample
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        if len(pred) > 1:
            corr_s, _ = stats.spearmanr(pred, true)
            corr_p, _ = stats.pearsonr(pred, true)
        else:
            corr_s, corr_p = 0, 0
        
        # Plot
        ax.plot(weeks, pred, 'o-', label='Prediction', color='red', linewidth=2, markersize=6)
        ax.plot(weeks, true, 's-', label='Ground Truth', color='blue', linewidth=2, markersize=6)
        
        # Add error bars
        errors = pred - true
        for i, (w, p, t, e) in enumerate(zip(weeks, pred, true, errors)):
            if e > 0:
                ax.plot([w, w], [t, p], 'r--', alpha=0.3, linewidth=1)
            else:
                ax.plot([w, w], [p, t], 'b--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Positive Rate (Difference)', fontsize=12)
        ax.set_title(f'Sample {sample_idx+1}/{num_total} | MAE: {mae:.4f}, MSE: {mse:.4f}, SpearmanR: {corr_s:.3f}, PearsonR: {corr_p:.3f}',
                    fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(weeks)
        ax.set_xticklabels([f'W{w}\n{date_strs[i][-5:]}' for i, w in enumerate(weeks)], 
                          fontsize=8, rotation=0)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{model_id}_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu: {output_file}")
    plt.close()

def plot_scatter(preds, trues, model_id, output_dir='./figures/'):
    """Plot scatter plot of predictions vs ground truth"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten arrays
    pred_flat = preds.reshape(-1)
    true_flat = trues.reshape(-1)
    
    # Calculate metrics
    mae = np.mean(np.abs(pred_flat - true_flat))
    mse = np.mean((pred_flat - true_flat) ** 2)
    rmse = np.sqrt(mse)
    corr_s, p_s = stats.spearmanr(pred_flat, true_flat)
    corr_p, p_p = stats.pearsonr(pred_flat, true_flat)
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.scatter(true_flat, pred_flat, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add statistics text
    text_str = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n'
    text_str += f'Spearman R: {corr_s:.4f}\nPearson R: {corr_p:.4f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Ground Truth', fontsize=14)
    ax.set_ylabel('Prediction', fontsize=14)
    ax.set_title(f'Predictions vs Ground Truth - {model_id}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{model_id}_scatter.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu: {output_file}")
    plt.close()

def plot_error_distribution(preds, trues, model_id, output_dir='./figures/'):
    """Plot error distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate errors
    errors = (preds - trues).reshape(-1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
              label=f'Mean: {errors.mean():.4f}')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute error histogram
    ax = axes[0, 1]
    abs_errors = np.abs(errors)
    ax.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(abs_errors.mean(), color='green', linestyle='--', linewidth=2,
              label=f'MAE: {abs_errors.mean():.4f}')
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Absolute Error Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error over time
    ax = axes[1, 0]
    errors_2d = (preds - trues).squeeze()
    mean_error_per_step = errors_2d.mean(axis=0)
    std_error_per_step = errors_2d.std(axis=0)
    steps = np.arange(1, len(mean_error_per_step) + 1)
    
    ax.plot(steps, mean_error_per_step, 'o-', color='blue', linewidth=2, markersize=8, label='Mean Error')
    ax.fill_between(steps, 
                    mean_error_per_step - std_error_per_step,
                    mean_error_per_step + std_error_per_step,
                    alpha=0.3, color='blue', label='±1 Std')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('Prediction Error', fontsize=12)
    ax.set_title('Error Over Prediction Steps', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    
    # 4. Absolute error over time
    ax = axes[1, 1]
    abs_errors_2d = np.abs(preds - trues).squeeze()
    mean_abs_error_per_step = abs_errors_2d.mean(axis=0)
    std_abs_error_per_step = abs_errors_2d.std(axis=0)
    
    ax.plot(steps, mean_abs_error_per_step, 'o-', color='red', linewidth=2, markersize=8, label='MAE')
    ax.fill_between(steps,
                    mean_abs_error_per_step - std_abs_error_per_step,
                    mean_abs_error_per_step + std_abs_error_per_step,
                    alpha=0.3, color='red', label='±1 Std')
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('MAE Over Prediction Steps', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{model_id}_error_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID (e.g., test_DLinear)')
    parser.add_argument('--pred_dir', type=str, default='./predictions/', help='Directory containing predictions')
    parser.add_argument('--output_dir', type=str, default='./figures/', help='Output directory for figures')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to plot in comparison')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VISUALIZE PREDICTIONS")
    print("=" * 80)
    print(f"\nModel ID: {args.model_id}")
    print(f"Predictions directory: {args.pred_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load predictions
    print("\nĐang load predictions...")
    preds, trues, dates = load_predictions(args.model_id, args.pred_dir)
    
    # Create visualizations
    print("\nĐang tạo visualizations...")
    print("\n1. Comparison plots...")
    plot_comparison(preds, trues, dates, args.model_id, args.output_dir, args.num_samples)
    
    print("\n2. Scatter plot...")
    plot_scatter(preds, trues, args.model_id, args.output_dir)
    
    print("\n3. Error analysis...")
    plot_error_distribution(preds, trues, args.model_id, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH!")
    print("=" * 80)
    print(f"\nCác hình ảnh đã được lưu trong: {args.output_dir}")
    print(f"- {args.model_id}_comparison.png    : So sánh pred vs true")
    print(f"- {args.model_id}_scatter.png       : Scatter plot")
    print(f"- {args.model_id}_error_analysis.png: Phân tích error")

if __name__ == '__main__':
    main()
