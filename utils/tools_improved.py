"""
IMPROVED VERSION of utils/tools.py with:
1. Epidemic metrics
2. Post-processing (clipping)

This is a modified version - you can either:
- Option A: Replace the test() function in tools.py with this version
- Option B: Use this as a reference to modify tools.py manually
"""

# Import everything from original tools.py
from utils.tools import *

# Import new improvements
from utils.epidemic_metrics import epidemic_metrics, print_epidemic_metrics
from utils.post_processing import clip_predictions, analyze_clipping_impact, print_clipping_analysis


def test_improved(model, test_data, test_loader, args, device, itr, train_data=None):
    """
    Improved test function with epidemic metrics and clipping.
    
    Args:
        model: Model to test
        test_data: Test dataset
        test_loader: Test data loader
        args: Arguments
        device: Device (CPU/GPU)
        itr: Iteration number
        train_data: Training data (optional, for outbreak threshold learning)
    
    Returns:
        mse, mae, corr1, corr2, epidemic_metrics_dict
    """
    preds = []
    trues = []
    X = []
    Date = []
    
    # Load data for inverse transform if needed
    if args.if_inverse == 1:
        data = pd.read_excel('/data_disk/lichx/CN_CDC/data.xlsx', sheet_name=args.order)
        df = change_date(data)
        diff1 = df.diff(1).dropna()
        diff2 = diff1.diff(52).dropna()
    elif args.if_inverse == 2:
        df = pd.read_csv('/data_disk/lichx/CQ_CDC/2010-2023流感数据/Weekly/Weekly_pre.csv',
                        parse_dates=['date'], index_col='date')
    elif args.if_inverse == 3:
        df = pd.read_csv('/data_disk/lichx/CQ_CDC/2010-2023流感数据/positive_rate_pre.csv',
                        parse_dates=['date'], index_col='date')
        diff1 = df.diff(1).dropna()
        diff2 = diff1.diff(52).dropna()
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, date) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.to(device)
            
            x = batch_x.detach().cpu().numpy()
            date = date.detach().cpu().numpy()

            with autocast():
                outputs = model(batch_x, itr)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            X.append(x)
            Date.append(date)

    # Concatenate all predictions
    preds = np.array(preds)
    trues = np.array(trues)
    X = np.array(X)
    Date = np.array(Date)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    X = np.concatenate(X, axis=0)
    Date = np.concatenate(Date, axis=0)

    # Store original predictions before any processing
    preds_original = preds.copy()
    
    # Inverse transform if needed
    corr_sperman, corr_pearson = [], []
    for j in range(preds.shape[0]):
        dt = [f"{year}-{month}-{day}" for year, month, day in Date[j, :, :].astype(int)]
        start = dt[-args.pred_len]
        end = dt[-1]
        
        if args.if_inverse == 1 or args.if_inverse == 3:
            preds[j,:,0] = inverse_diff2(preds[j,:,0], start=start, end=end, _diff1=diff1, _df=df).values.reshape(-1)
            trues[j,:,0] = inverse_diff2(trues[j,:,0], start=start, end=end, _diff1=diff1, _df=df).values.reshape(-1)
            X[j,:,0] = inverse_diff2(X[j,:,0], start=dt[0], end=dt[-args.pred_len-1], _diff1=diff1, _df=df).values.reshape(-1)
        elif args.if_inverse == 2:
            preds[j,:,0] = inverse_diff1(preds[j,:,0], start=start, end=end, _df=df).values.reshape(-1)
            trues[j,:,0] = inverse_diff1(trues[j,:,0], start=start, end=end, _df=df).values.reshape(-1)
            X[j,:,0] = inverse_diff1(X[j,:,0], start=dt[0], end=dt[-args.pred_len-1], _df=df).values.reshape(-1)

        # Correlation for individual samples
        if args.pred_len > 1:
            corr_s, _ = stats.spearmanr(preds[j, :, :].reshape(-1), trues[j, :, :].reshape(-1))
            corr_p, _ = stats.pearsonr(preds[j, :, :].reshape(-1), trues[j, :, :].reshape(-1))
            corr_sperman.append(corr_s)
            corr_pearson.append(corr_p)
    
    corr1 = np.nanmean(corr_sperman)
    corr2 = np.nanmean(corr_pearson)
    
    # ========================================================================
    # 🆕 NEW: POST-PROCESSING WITH CLIPPING
    # ========================================================================
    print("\n" + "="*70)
    print("POST-PROCESSING: CLIPPING ANALYSIS")
    print("="*70)
    
    pred_before_clip = preds.copy()
    pred_clipped = clip_predictions(preds, min_value=0.0)
    
    # Analyze clipping impact
    clipping_analysis = analyze_clipping_impact(
        pred_before_clip, pred_clipped, trues
    )
    print_clipping_analysis(clipping_analysis)
    
    # ========================================================================
    # COMPUTE STANDARD METRICS FOR BOTH VERSIONS
    # ========================================================================
    print("\n" + "="*70)
    print("STANDARD METRICS - BEFORE CLIPPING")
    print("="*70)
    mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig, smape_orig, nd_orig = metric(pred_before_clip, trues)
    print(f'mse:{mse_orig:.6f}, mae:{mae_orig:.6f}, rmse:{rmse_orig:.6f}')
    print(f'mape:{mape_orig:.6f}, smape:{smape_orig:.6f}')
    
    print("\n" + "="*70)
    print("STANDARD METRICS - AFTER CLIPPING (RECOMMENDED)")
    print("="*70)
    mae, mse, rmse, mape, mspe, smape, nd = metric(pred_clipped, trues)
    print(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}')
    print(f'mape:{mape:.6f}, smape:{smape:.6f}')
    
    # Use clipped version for subsequent analysis
    preds = pred_clipped
    
    # ========================================================================
    # 🆕 NEW: EPIDEMIC-SPECIFIC METRICS
    # ========================================================================
    print("\n" + "="*70)
    print("EPIDEMIC-SPECIFIC METRICS")
    print("="*70)
    
    epidemic_metrics_dict = None
    try:
        # Reshape for epidemic metrics (need 2D: [n_samples, n_timesteps])
        preds_2d = preds.squeeze(-1) if preds.ndim == 3 else preds
        trues_2d = trues.squeeze(-1) if trues.ndim == 3 else trues
        
        epidemic_metrics_dict = epidemic_metrics(
            pred=preds_2d,
            true=trues_2d,
            train_data=train_data,
        )
        
        print_epidemic_metrics(epidemic_metrics_dict)
        
    except Exception as e:
        print(f"⚠️ Warning: Could not compute epidemic metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f'seq_len: {args.seq_len}, if_inverse: {args.if_inverse}')
    print(f'spearmanR: {corr1:.4f}, pearsonR: {corr2:.4f}')
    print(f'mae: {mae:.4f}, mse: {mse:.4f}, mape: {mape:.4f}, smape: {smape:.4f}')
    
    if epidemic_metrics_dict:
        print(f"\nEpidemic Metrics:")
        print(f'  Peak Timing Error: {epidemic_metrics_dict["peak_timing_error_weeks"]:.2f} weeks')
        print(f'  Peak Intensity Error: {epidemic_metrics_dict["peak_intensity_abs_error"]:.2f}%')
        print(f'  Direction Accuracy: {epidemic_metrics_dict["direction_accuracy_pct"]:.1f}%')
    
    print("="*70 + "\n")
    
    # ========================================================================
    # VISUALIZATION (if enabled)
    # ========================================================================
    if args.plt != 0:
        folder_path = f"/data_disk/lichx/NeurIPS2023-One-Fits-All/Long-term_Forecasting/Visual_relu_new/{args.data_path.split('.')[0]}/{args.model}/"
        
        for k in range(0, preds.shape[0], max(1, preds.shape[0] // 10)):
            dt2 = [f"{year}-{month}-{day}" for year, month, day in Date[k, :, :].astype(int)]
            dt2 = pd.to_datetime(dt2)
            GT = np.concatenate((X[k, :, 0], trues[k, :, 0]), axis=0).reshape(-1,1)
            PD = np.concatenate((X[k, :, 0], preds[k, :, 0]), axis=0).reshape(-1,1)
            
            visual(dt2, GT, PD, corr_sperman[k], corr_pearson[k], 
                   os.path.join(folder_path, f'if_inverse-{args.if_inverse}_pred_len-{args.pred_len}_epochs-{args.train_epochs}_{k}.pdf'))
    
    return mse, mae, corr1, corr2, epidemic_metrics_dict


# Wrapper to maintain backward compatibility
def test_with_improvements(model, test_data, test_loader, args, device, itr, train_data=None):
    """
    Wrapper function that can be used as a drop-in replacement for test().
    """
    return test_improved(model, test_data, test_loader, args, device, itr, train_data)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    IMPROVED TOOLS.PY WITH NEW FEATURES                       ║
║                                                                              ║
║  This file provides an improved test() function with:                       ║
║  1. Epidemic-specific metrics                                               ║
║  2. Post-processing with clipping                                           ║
║  3. Detailed analysis and reporting                                         ║
║                                                                              ║
║  HOW TO USE:                                                                 ║
║  Option A: Import test_improved() and use instead of test()                 ║
║  Option B: Copy the improvements into your original tools.py                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
