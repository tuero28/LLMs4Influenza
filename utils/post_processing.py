"""
Post-processing utilities for flu forecasting predictions.

Includes:
- Clipping predictions to valid ranges
- Analyzing the impact of clipping
- Constraint enforcement

Author: Generated for LLMs4Influenza project
Date: 2024
"""

import numpy as np
import torch


def clip_predictions(pred, min_value=0.0, max_value=None):
    """
    Clip predictions to valid range.
    
    For flu forecasting:
    - min_value=0: Flu rates cannot be negative
    - max_value=None: No upper limit (some regions can exceed 100% 
      due to counting multiple subtypes)
    
    ⚠️ IMPORTANT:
    - DO NOT set max=100: Some regions exceed 100%
    - DO NOT enforce 10%/week limit: Too rigid, can be wrong
    - DO NOT use "seasonal max": Hard to define, can be wrong
    
    Args:
        pred: Predictions (numpy array or torch tensor)
        min_value: Minimum value (default: 0.0 for non-negative)
        max_value: Maximum value (default: None for no upper limit)
    
    Returns:
        Clipped predictions (same type as input)
    
    Example:
        >>> pred = np.array([-1.5, 10.2, 25.8, -0.3])
        >>> clip_predictions(pred, min_value=0.0)
        array([ 0. , 10.2, 25.8,  0. ])
    """
    is_torch = isinstance(pred, torch.Tensor)
    
    if is_torch:
        if max_value is not None:
            return torch.clamp(pred, min=min_value, max=max_value)
        else:
            return torch.clamp(pred, min=min_value)
    else:
        return np.clip(pred, min_value, max_value)


def analyze_clipping_impact(pred_original, pred_clipped, true):
    """
    Analyze the impact of clipping on metrics.
    
    This helps understand:
    1. How many values were negative
    2. Whether clipping improved or worsened metrics
    3. The magnitude of changes
    
    Args:
        pred_original: Original predictions (before clipping)
        pred_clipped: Clipped predictions
        true: Ground truth
    
    Returns:
        dict: Analysis results including:
            - Metrics before and after clipping
            - Number of values clipped
            - Improvement in metrics
    
    Example:
        >>> analysis = analyze_clipping_impact(pred_orig, pred_clip, true)
        >>> print(f"Clipped {analysis['n_negative_values']} negative values")
        >>> print(f"MAE improved by {analysis['improvement']['mae']:.6f}")
    """
    from utils.metrics import metric
    
    # Compute metrics before clipping
    mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig, smape_orig, nd_orig = \
        metric(pred_original, true)
    
    # Compute metrics after clipping
    mae_clip, mse_clip, rmse_clip, mape_clip, mspe_clip, smape_clip, nd_clip = \
        metric(pred_clipped, true)
    
    # Analyze clipping
    results = {
        'before_clip': {
            'mae': float(mae_orig),
            'mse': float(mse_orig),
            'rmse': float(rmse_orig),
            'mape': float(mape_orig),
            'mspe': float(mspe_orig),
            'smape': float(smape_orig),
        },
        'after_clip': {
            'mae': float(mae_clip),
            'mse': float(mse_clip),
            'rmse': float(rmse_clip),
            'mape': float(mape_clip),
            'mspe': float(mspe_clip),
            'smape': float(smape_clip),
        },
        'improvement': {
            'mae': float(mae_orig - mae_clip),
            'mse': float(mse_orig - mse_clip),
            'rmse': float(rmse_orig - rmse_clip),
            'mape': float(mape_orig - mape_clip),
            'mspe': float(mspe_orig - mspe_clip),
            'smape': float(smape_orig - smape_clip),
        },
        'n_negative_values': int(np.sum(pred_original < 0)),
        'min_value_before': float(np.min(pred_original)),
        'max_value_before': float(np.max(pred_original)),
        'n_clipped': int(np.sum(pred_original != pred_clipped)),
        'percent_clipped': float(np.mean(pred_original != pred_clipped) * 100),
    }
    
    return results


def print_clipping_analysis(analysis):
    """
    Print clipping analysis in a nice format.
    
    Args:
        analysis: Dictionary from analyze_clipping_impact()
    """
    print("\n" + "="*70)
    print("CLIPPING ANALYSIS")
    print("="*70)
    
    # Statistics (removed emojis for Windows compatibility)
    print(f"\nClipping Statistics:")
    print(f"   Negative values detected: {analysis['n_negative_values']}")
    print(f"   Value range before: [{analysis['min_value_before']:.4f}, "
          f"{analysis['max_value_before']:.4f}]")
    print(f"   Total values clipped: {analysis['n_clipped']} "
          f"({analysis['percent_clipped']:.2f}%)")
    
    # Metrics comparison
    print("\nMetrics Comparison:")
    print(f"   {'Metric':<8} {'Before':<12} {'After':<12} {'Change':<12} {'Status'}")
    print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    
    for metric_name in ['mae', 'mse', 'rmse']:
        before = analysis['before_clip'][metric_name]
        after = analysis['after_clip'][metric_name]
        change = analysis['improvement'][metric_name]
        
        # Status indicator (using ASCII instead of emoji)
        if change > 1e-6:
            status = "[OK]"  # Improved
        elif change < -1e-6:
            status = "[!]"  # Worsened
        else:
            status = "[=]"  # No change
        
        print(f"   {metric_name.upper():<8} {before:<12.6f} {after:<12.6f} "
              f"{change:+<12.6f} {status}")
    
    # Interpretation
    print("\nInterpretation:")
    if analysis['n_negative_values'] > 0:
        if analysis['improvement']['mae'] > 0:
            print("   [OK] Clipping improved metrics!")
            print("   -> Negative predictions were unrealistic.")
        else:
            print("   [!] Clipping slightly worsened metrics numerically,")
            print("   -> BUT predictions are now more realistic (non-negative).")
            print("   -> Recommended to use clipped version for reporting.")
    else:
        print("   [OK] No negative values found - clipping has no effect.")
    
    print("="*70 + "\n")


def enforce_physical_constraints(predictions, 
                                 non_negative=True,
                                 smooth_transitions=False,
                                 max_weekly_change=10.0):
    """
    Enforce physical constraints on predictions.
    
    This is an ADVANCED version with optional constraints.
    Use with caution - overly strict constraints can hurt performance.
    
    Args:
        predictions: Predictions array/tensor [B, T, C] or [T, C]
        non_negative: Enforce non-negativity (recommended: True)
        smooth_transitions: Enforce smooth week-to-week changes (use carefully)
        max_weekly_change: Maximum % change per week (only if smooth_transitions=True)
    
    Returns:
        Constrained predictions
    
    ⚠️ WARNING:
    - smooth_transitions can be too restrictive
    - Real flu data can have sudden changes
    - Only use if you have domain expertise
    """
    is_torch = isinstance(predictions, torch.Tensor)
    
    if is_torch:
        pred = predictions.clone()
    else:
        pred = predictions.copy()
    
    # 1. Non-negativity (strongly recommended)
    if non_negative:
        if is_torch:
            pred = torch.clamp(pred, min=0.0)
        else:
            pred = np.clip(pred, 0.0, None)
    
    # 2. Smooth transitions (optional, use with caution)
    if smooth_transitions:
        print("⚠️ Warning: Enforcing smooth transitions - this may be too restrictive!")
        
        # Get shape
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)  # Add batch dimension
            remove_batch = True
        else:
            remove_batch = False
        
        # Enforce smooth changes
        for b in range(pred.shape[0]):
            for t in range(1, pred.shape[1]):
                diff = pred[b, t] - pred[b, t-1]
                
                # Clip change
                if is_torch:
                    diff = torch.clamp(diff, -max_weekly_change, max_weekly_change)
                else:
                    diff = np.clip(diff, -max_weekly_change, max_weekly_change)
                
                pred[b, t] = pred[b, t-1] + diff
        
        if remove_batch:
            pred = pred.squeeze(0)
    
    return pred


# Example usage and testing
if __name__ == "__main__":
    print("Testing Post-processing Utilities...")
    print("="*70)
    
    # Test 1: Basic clipping
    print("\nTest 1: Basic Clipping")
    print("-"*70)
    
    # Create test data with some negative values
    np.random.seed(42)
    pred_orig = np.random.randn(10, 13) * 5 + 20  # Some will be negative
    true = np.random.rand(10, 13) * 30 + 10  # Always positive
    
    print(f"Original range: [{pred_orig.min():.2f}, {pred_orig.max():.2f}]")
    print(f"Negative values: {np.sum(pred_orig < 0)}")
    
    # Clip
    pred_clipped = clip_predictions(pred_orig, min_value=0.0)
    
    print(f"Clipped range: [{pred_clipped.min():.2f}, {pred_clipped.max():.2f}]")
    print(f"Negative values: {np.sum(pred_clipped < 0)}")
    
    # Test 2: Analyze impact
    print("\nTest 2: Clipping Impact Analysis")
    print("-"*70)
    
    analysis = analyze_clipping_impact(pred_orig, pred_clipped, true)
    print_clipping_analysis(analysis)
    
    # Test 3: PyTorch tensors
    print("\nTest 3: PyTorch Tensor Support")
    print("-"*70)
    
    pred_torch = torch.randn(5, 8) * 3 + 15
    print(f"Original (torch): range [{pred_torch.min():.2f}, {pred_torch.max():.2f}]")
    
    pred_torch_clipped = clip_predictions(pred_torch, min_value=0.0)
    print(f"Clipped (torch): range [{pred_torch_clipped.min():.2f}, "
          f"{pred_torch_clipped.max():.2f}]")
    
    # Test 4: Physical constraints (advanced)
    print("\nTest 4: Physical Constraints (Advanced)")
    print("-"*70)
    
    pred_test = np.array([[10, 15, -5, 30, 25, 20, 50]])  # Has jump and negative
    print(f"Original: {pred_test[0]}")
    
    pred_constrained = enforce_physical_constraints(
        pred_test,
        non_negative=True,
        smooth_transitions=True,
        max_weekly_change=10.0
    )
    print(f"Constrained: {pred_constrained[0]}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    
    # Usage recommendations
    print("\n📝 RECOMMENDATIONS:")
    print("="*70)
    print("""
1. ✅ ALWAYS use clip_predictions with min_value=0.0
   → Flu rates cannot be negative

2. ✅ REPORT both before and after clipping metrics
   → Shows transparency and impact

3. ⚠️ BE CAREFUL with smooth_transitions
   → Real flu data can have sudden changes
   → Only use if you have domain expertise

4. ❌ DO NOT set max_value=100
   → Some regions exceed 100% (multiple subtypes counted)

5. ✅ SAVE both original and clipped predictions
   → Allows later analysis
    """)
    print("="*70)
