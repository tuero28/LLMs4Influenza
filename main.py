from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali
from utils.tools_improved import test_improved as test  # Use improved version
from data_provider.data_augmentation import TimeSeriesAugmentation  # Add augmentation

from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.Llama2 import Llama2
from models.Gemma2 import Gemma2
from models.Llama3 import Llama3
# from ensemble_predictor import EnsemblePredictor, create_ensemble_from_checkpoints, create_ensemble_model  # Tạm thời bỏ qua

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import gc
from torch.cuda.amp import GradScaler, autocast

import os
import time
import psutil

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fix_seed', type=int, default=2021)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='NorthChina_diff.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=13)
    parser.add_argument('--label_len', type=int, default=18)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    # Đổi mặc định từ 10 thành 0 để tương thích tốt với Windows
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--llama_layers', type=int, default=32)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--cos', type=int, default=0)
    
    # Data augmentation arguments
    parser.add_argument('--use_augmentation', type=int, default=0, help='Enable data augmentation (0=disabled, 1=enabled)')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability of applying augmentation')

    # Ensemble predictor arguments
    parser.add_argument('--ensemble_weights', type=str, default='0.3,0.3,0.4',
                        help='Comma-separated ensemble weights for DLinear, PatchTST, Llama2 (default: 0.3,0.3,0.4)')
    parser.add_argument('--checkpoint_paths', type=str, default='',
                        help='Comma-separated checkpoint paths for ensemble models (optional)')
    parser.add_argument('--ensemble_iterations', type=int, default=0,
                        help='Enable iteration ensemble (0=disabled, 1=enabled)')
    parser.add_argument('--visualize', type=int, default=0,
                        help='Enable visualization of predictions (0=disabled, 1=enabled)')

    parser.add_argument('--plt', type=int, default=0)
    parser.add_argument('--read_model', type=int, default=0)
    parser.add_argument('--write_model', type=int, default=0)
    parser.add_argument('--if_inverse', type=int, default=0)
    parser.add_argument('--order', type=int, default=0)
    parser.add_argument('--fc_layer', type=int, default=512)

    args = parser.parse_args()

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    mses = []
    maes = []
    corr_s, corr_p = [], []
    # Store epidemic metrics across iterations
    all_epidemic_metrics = []

    process = psutil.Process()

    model = None  # Giữ tham chiếu cho write_model phía sau
    # ❌ REMOVED BROKEN PRE-LOADING
    # The pre-loading optimization was causing iterations to NOT be independent
    # Each iteration was training from the previous iteration's weights
    # This made mean/std calculations meaningless
    
    for ii in range(args.itr):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        start_time = time.time()
        cpu_mem_start = process.memory_info().rss / (1024 * 1024)
        if torch.cuda.is_available():
            gpu_mem_start_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
            gpu_mem_start_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        else:
            gpu_mem_start_alloc, gpu_mem_start_reserved = 0, 0

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len,
                                                                                 args.pred_len,
                                                                                 args.d_model, args.n_heads,
                                                                                 args.e_layers, args.gpt_layers,
                                                                                 args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        time_now = time.time()
        train_steps = len(train_loader)

        # Create model FRESH for each iteration (ensures independence)
        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'Llama2':
            model = Llama2(args, device)
        elif args.model == 'Gemma2':
            model = Gemma2(args, device)
        elif args.model == 'Llama3':
            model = Llama3(args, device)
        # elif args.model == 'ensemble':
        #     # Create ensemble predictor
        #     model = create_ensemble_model(args, device)
        else:
            model = GPT4TS(args, device)

        if args.read_model:
            pre_train_path = './PretrainModel/weather2/temp.pth'
            state_dict_model = torch.load(pre_train_path)

            unexpected_keys = [k for k in state_dict_model.keys() if k not in model.state_dict()]
            if unexpected_keys:
                print(f"Unexpected keys in state_dict: {unexpected_keys}")
            else:
                print("MATCH")

            filtered_state_dict = {k: v for k, v in state_dict_model.items() if k in model.state_dict()}
            model.load_state_dict(filtered_state_dict, strict=False)
            print('Loading completed')

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        # Skip training for ensemble model (uses pre-trained checkpoints)
        # if args.model != 'ensemble':
        if True:  # Always train (ensemble disabled)
            # Initialize augmentation if enabled
            augmenter = None
            if args.use_augmentation:
                augmenter = TimeSeriesAugmentation(
                    p=args.aug_prob,
                    jitter_sigma=0.03,
                    scaling_sigma=0.1,
                    rotation_sigma=0.05
                )
                print(f"[OK] Data augmentation ENABLED (p={args.aug_prob})")
            
            scaler = GradScaler()
            accumulation_steps = 1  # Set to 1 for small datasets with few batches
            for epoch in range(args.train_epochs):
                train_loss = []
                epoch_time = time.time()
                
                model.train()
                model_optim.zero_grad()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    
                    # Apply augmentation if enabled (only on training data)
                    if augmenter is not None:
                        batch_x = augmenter.batch_augment(batch_x, techniques=['jitter', 'scaling'])

                    with autocast():
                        outputs = model(batch_x, ii)
                        outputs = outputs[:, -args.pred_len:, :]
                        batch_y = batch_y[:, -args.pred_len:, :]
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())
                    scaler.scale(loss / accumulation_steps).backward()

                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                        scaler.step(model_optim)
                        scaler.update()
                        model_optim.zero_grad()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                # Calculate average training loss for this epoch
                avg_train_loss = np.average(train_loss)
                vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, avg_train_loss, vali_loss), flush=True)
                
                # Learning rate adjustment
                if args.cos:
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(model_optim, epoch + 1, args)
                
                # Early stopping - ENABLED for saving checkpoints
                early_stopping(vali_loss, model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # Only exits the epoch loop
    
        # Load best model for testing
        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from {best_model_path}")
        
        print("------------------------------------")
        # Test with improved metrics
        result = test(model, test_data, test_loader, args, device, ii)
        if len(result) == 5:
            mse, mae, corr1, corr2, epidemic_metrics_dict = result
            all_epidemic_metrics.append(epidemic_metrics_dict)
        else:
            mse, mae, corr1, corr2 = result
        
        # Store metrics for summary across iterations
        mses.append(mse)
        maes.append(mae)
        corr_s.append(corr1)  # Spearman correlation
        corr_p.append(corr2)  # Pearson correlation
        # Note: p-values are not returned by test() function

        end_time = time.time()
        cpu_mem_end = process.memory_info().rss / (1024 * 1024)
        if torch.cuda.is_available():
            gpu_mem_end_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
            gpu_mem_end_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        else:
            gpu_mem_end_alloc, gpu_mem_end_reserved = 0, 0

        print(f"Iteration {ii + 1}/{args.itr} completed in {end_time - start_time:.2f} seconds")
        print(f"CPU Memory Usage: Start {cpu_mem_start:.2f} MB, End {cpu_mem_end:.2f} MB")
        print(f"GPU Memory Usage: Allocated Start {gpu_mem_start_alloc:.2f} MB, End {gpu_mem_end_alloc:.2f} MB")
        print(f"GPU Memory Usage: Reserved Start {gpu_mem_start_reserved:.2f} MB, End {gpu_mem_end_reserved:.2f} MB")

        # Save model BEFORE deleting (if needed)
        if args.write_model and model is not None:
            save_path = './PretrainModel/' + args.model_id.split('_')[0]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + '/' + args.model + '-' + 'temp.pth')
            print(f"Model saved to {save_path}/{args.model}-temp.pth")

        # Clean up after saving
        del model_optim
        del test_loader
        # Clean up model after each iteration
        if model is not None:
            del model
            model = None  # Explicitly set to None
        gc.collect()
        torch.cuda.empty_cache()

    mses = np.array(mses)
    maes = np.array(maes)
    corr_s = np.array(corr_s)
    corr_p = np.array(corr_p)

    if len(mses) > 0:
        print("\n" + "="*60)
        print("SUMMARY ACROSS {} ITERATIONS".format(args.itr))
        print("="*60)
        print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
        print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
        print("spearmanR_mean = {:.4f}, spearmanR_std = {:.4f}".format(np.mean(corr_s), np.std(corr_s)))
        print("pearsonR_mean = {:.4f}, pearsonR_std = {:.4f}".format(np.mean(corr_p), np.std(corr_p)))
        
        # Print epidemic metrics summary if available
        if len(all_epidemic_metrics) > 0:
            print("\n" + "-"*60)
            print("EPIDEMIC METRICS SUMMARY")
            print("-"*60)
            
            # Aggregate metrics using correct keys
            peak_timing_errors = [m.get('peak_timing_error_weeks', 0) for m in all_epidemic_metrics if m.get('peak_timing_error_weeks', 0) != 0]
            peak_intensity_rel_errors = [m.get('peak_intensity_rel_error_pct', 0) for m in all_epidemic_metrics if m.get('peak_intensity_rel_error_pct', 0) != 0]
            direction_accuracies = [m.get('direction_accuracy_pct', 0) for m in all_epidemic_metrics if m.get('direction_accuracy_pct', 0) != 0]
            
            if len(peak_timing_errors) > 0:
                print("Peak Timing Error: mean = {:.2f} weeks, std = {:.2f}".format(
                    np.mean(peak_timing_errors), np.std(peak_timing_errors)))
            if len(peak_intensity_rel_errors) > 0:
                print("Peak Intensity Error: mean = {:.2f}%, std = {:.2f}%".format(
                    np.mean(peak_intensity_rel_errors), np.std(peak_intensity_rel_errors)))
            if len(direction_accuracies) > 0:
                print("Direction Accuracy: mean = {:.2f}%, std = {:.2f}%".format(
                    np.mean(direction_accuracies), np.std(direction_accuracies)))
        
        print("="*60 + "\n")


# BẮT BUỘC TRÊN WINDOWS:
if __name__ == '__main__':
    main()