"""
Script để lưu kết quả dự đoán ra file
"""
import os
import numpy as np
import pandas as pd
import torch
from data_provider.data_factory import data_provider
from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.Llama2 import Llama2
from models.Llama3 import Llama3
from models.Gemma2 import Gemma2
import argparse
from torch.cuda.amp import autocast

def load_model(args, device):
    """Load trained model from checkpoint"""
    # Initialize model
    if args.model == 'DLinear':
        model = DLinear(args).to(device)
    elif args.model == 'PatchTST':
        model = PatchTST(args).to(device)
    elif args.model == 'GPT4TS':
        model = GPT4TS(args, device).to(device)
    elif args.model == 'Llama2':
        model = Llama2(args, device).to(device)
    elif args.model == 'Llama3':
        model = Llama3(args, device).to(device)
    elif args.model == 'Gemma2':
        model = Gemma2(args, device).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint: {args.checkpoint_path}")
    else:
        print(f"❌ Checkpoint không tồn tại: {args.checkpoint_path}")
        print(f"Hãy train model trước hoặc kiểm tra đường dẫn!")
        exit(1)
    
    return model

def save_predictions(model, test_data, test_loader, args, device, output_dir):
    """Run inference and save predictions"""
    preds = []
    trues = []
    dates = []
    
    model.eval()
    print("\nĐang chạy predictions...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, date) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            date_np = date.detach().cpu().numpy()
            
            with autocast():
                outputs = model(batch_x, 0)  # itr=0 for inference
            
            # Get predictions for pred_len steps
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            dates.append(date_np)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} batches")
    
    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)  # (num_samples, pred_len, features)
    trues = np.concatenate(trues, axis=0)
    dates = np.concatenate(dates, axis=0)
    
    print(f"\n✅ Predictions shape: {preds.shape}")
    print(f"✅ Ground truth shape: {trues.shape}")
    print(f"✅ Dates shape: {dates.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as .npy files
    pred_file = os.path.join(output_dir, f"{args.model_id}_pred.npy")
    true_file = os.path.join(output_dir, f"{args.model_id}_true.npy")
    date_file = os.path.join(output_dir, f"{args.model_id}_dates.npy")
    
    np.save(pred_file, preds)
    np.save(true_file, trues)
    np.save(date_file, dates)
    
    print(f"\n💾 Đã lưu predictions:")
    print(f"  - {pred_file}")
    print(f"  - {true_file}")
    print(f"  - {date_file}")
    
    # Save as CSV with detailed results
    csv_file = os.path.join(output_dir, f"{args.model_id}_results.csv")
    results = []
    
    for i in range(preds.shape[0]):
        for j in range(args.pred_len):
            # Convert date (year, month, day) to string
            year, month, day = dates[i, -args.pred_len + j, :].astype(int)
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            
            pred_val = preds[i, j, 0]
            true_val = trues[i, j, 0]
            error = pred_val - true_val
            
            results.append({
                'sample': i,
                'week': j + 1,
                'date': date_str,
                'prediction': pred_val,
                'ground_truth': true_val,
                'error': error,
                'abs_error': abs(error),
                'squared_error': error ** 2
            })
    
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"  - {csv_file}")
    
    # Calculate and save summary statistics
    summary_file = os.path.join(output_dir, f"{args.model_id}_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"KẾT QUẢ DỰ ĐOÁN - {args.model_id}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Sequence length: {args.seq_len}\n")
        f.write(f"Prediction length: {args.pred_len}\n")
        f.write(f"Target: {args.target}\n\n")
        
        f.write(f"Test samples: {preds.shape[0]}\n")
        f.write(f"Predictions per sample: {args.pred_len}\n")
        f.write(f"Total predictions: {preds.shape[0] * args.pred_len}\n\n")
        
        # Calculate metrics
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        
        f.write(f"METRICS:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n\n")
        
        # Correlation
        from scipy import stats
        if args.pred_len > 1:
            corr_s, p_s = stats.spearmanr(preds.reshape(-1), trues.reshape(-1))
            corr_p, p_p = stats.pearsonr(preds.reshape(-1), trues.reshape(-1))
            f.write(f"Spearman R:  {corr_s:.6f} (p={p_s:.6e})\n")
            f.write(f"Pearson R:   {corr_p:.6f} (p={p_p:.6e})\n\n")
        
        # Distribution stats
        f.write(f"PREDICTION DISTRIBUTION:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Min:    {preds.min():.6f}\n")
        f.write(f"Max:    {preds.max():.6f}\n")
        f.write(f"Mean:   {preds.mean():.6f}\n")
        f.write(f"Median: {np.median(preds):.6f}\n")
        f.write(f"Std:    {preds.std():.6f}\n\n")
        
        f.write(f"GROUND TRUTH DISTRIBUTION:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Min:    {trues.min():.6f}\n")
        f.write(f"Max:    {trues.max():.6f}\n")
        f.write(f"Mean:   {trues.mean():.6f}\n")
        f.write(f"Median: {np.median(trues):.6f}\n")
        f.write(f"Std:    {trues.std():.6f}\n\n")
    
    print(f"  - {summary_file}")
    print(f"\n✅ Hoàn thành! Kiểm tra thư mục: {output_dir}")
    
    return preds, trues, dates

def main():
    parser = argparse.ArgumentParser()
    
    # Model settings (giống main.py)
    parser.add_argument('--model', type=str, required=True, help='Model name: DLinear, PatchTST, Llama2, etc.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID used during training')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint.pth')
    
    # Data settings
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='NorthChina_diff.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='positive_rate')
    parser.add_argument('--freq', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=100)
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=52)
    parser.add_argument('--label_len', type=int, default=18)
    parser.add_argument('--pred_len', type=int, default=8)
    
    # Model params (cần match với training)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--llama_layers', type=int, default=32)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./predictions/')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LƯU KẾT QUẢ DỰ ĐOÁN")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Model ID: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Output: {args.output_dir}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data
    print("\nĐang load test data...")
    test_data, test_loader = data_provider(args, 'test')
    print(f"✅ Test data loaded: {len(test_data)} samples")
    
    # Load model
    print("\nĐang load model...")
    model = load_model(args, device)
    
    # Save predictions
    preds, trues, dates = save_predictions(model, test_data, test_loader, args, device, args.output_dir)
    
    print("\n" + "=" * 80)
    print("💡 CÁCH SỬ DỤNG KẾT QUẢ:")
    print("=" * 80)
    print(f"""
1. Xem summary:
   notepad {os.path.join(args.output_dir, f"{args.model_id}_summary.txt")}

2. Xem chi tiết CSV:
   notepad {os.path.join(args.output_dir, f"{args.model_id}_results.csv")}

3. Load predictions trong Python:
   import numpy as np
   preds = np.load('{os.path.join(args.output_dir, f"{args.model_id}_pred.npy")}')
   trues = np.load('{os.path.join(args.output_dir, f"{args.model_id}_true.npy")}')

4. Visualize:
   python visualize_predictions.py --model_id {args.model_id}
""")

if __name__ == '__main__':
    main()
