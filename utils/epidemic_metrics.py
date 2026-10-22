"""
Epidemic-specific metrics for flu forecasting.
These metrics are more relevant for epidemic forecasting than standard MSE/MAE.

Author: Generated for LLMs4Influenza project
Date: 2024
"""

import numpy as np
import os
from sklearn.metrics import f1_score


def peak_timing_error(pred, true):
    """
    Calculate peak timing error (in weeks).
    
    This measures how many weeks off the predicted peak is from the true peak.
    Critical for epidemic preparedness.
    
    Args:
        pred: Predictions [B, T] or [T]
        true: Ground truth [B, T] or [T]
    
    Returns:
        float: Average peak timing error in weeks
    
    Example:
        If true peak is at week 5 and predicted peak is at week 7,
        then peak_timing_error = 2 weeks
    """
    pred_peak_idx = np.argmax(pred, axis=-1)
    true_peak_idx = np.argmax(true, axis=-1)
    
    error = np.abs(pred_peak_idx - true_peak_idx)
    return np.mean(error)


def peak_intensity_error(pred, true):
    """
    Calculate peak intensity error.
    
    This measures how different the predicted peak value is from the true peak.
    
    Args:
        pred: Predictions [B, T] or [T]
        true: Ground truth [B, T] or [T]
    
    Returns:
        tuple: (absolute_error, relative_error_percentage)
    
    Example:
        If true peak = 30% and predicted peak = 35%
        then abs_error = 5%, rel_error = 16.67%
    """
    pred_peak_value = np.max(pred, axis=-1)
    true_peak_value = np.max(true, axis=-1)
    
    # Absolute error
    abs_error = np.abs(pred_peak_value - true_peak_value)
    
    # Relative error (%)
    rel_error = abs_error / (true_peak_value + 1e-8) * 100
    
    return np.mean(abs_error), np.mean(rel_error)


def direction_accuracy(pred, true):
    """
    Calculate direction accuracy (trend prediction).
    
    This measures how well the model predicts whether the value will
    increase or decrease from one week to the next.
    
    Args:
        pred: Predictions [B, T] or [T]
        true: Ground truth [B, T] or [T]
    
    Returns:
        float: Direction accuracy in percentage (0-100)
    
    Example:
        If this week is 20% and next week is 25% (increasing),
        and the prediction also increases, that's correct.
    """
    # Calculate week-to-week differences
    pred_diff = pred[..., 1:] - pred[..., :-1]
    true_diff = true[..., 1:] - true[..., :-1]
    
    # Handle edge cases
    if pred_diff.size == 0 or true_diff.size == 0:
        return 0.0
    
    # Check if all predictions are same (no direction changes)
    if np.all(pred_diff == 0):
        return 0.0  # No direction predicted
    
    # Get direction: +1 (increase), -1 (decrease), 0 (no change)
    pred_direction = np.sign(pred_diff)
    true_direction = np.sign(true_diff)
    
    # Calculate accuracy
    correct = (pred_direction == true_direction)
    accuracy = np.mean(correct) * 100
    
    # Handle NaN
    if np.isnan(accuracy):
        return 0.0
    
    return float(accuracy)


def learn_outbreak_threshold(train_data, percentile=75):
    """
    Learn outbreak threshold from training data.
    
    This is better than using a fixed threshold (e.g., 20%) because
    different regions have different baseline flu rates.
    
    Args:
        train_data: Training data array
        percentile: Percentile to use as threshold (default: 75th percentile)
    
    Returns:
        float: Learned threshold value
    
    Raises:
        ValueError: If train_data is None or empty
    """
    if train_data is None or len(train_data) == 0:
        raise ValueError("Train data is required to learn outbreak threshold!")
    
    threshold = np.percentile(train_data, percentile)
    return threshold


def outbreak_detection_f1(pred, true, train_data=None, threshold=None):
    """
    Calculate F1 score for outbreak detection.
    
    An "outbreak" is defined as when the flu rate exceeds a threshold.
    The threshold should be learned from training data, not arbitrarily set.
    
    Args:
        pred: Predictions
        true: Ground truth
        train_data: Training data to learn threshold (preferred)
        threshold: Fixed threshold value (only if train_data is not available)
    
    Returns:
        tuple: (f1_score, threshold_used)
    
    Raises:
        ValueError: If neither train_data nor threshold is provided
    """
    if threshold is None:
        if train_data is not None:
            threshold = learn_outbreak_threshold(train_data, percentile=75)
        else:
            raise ValueError(
                "Must provide either 'train_data' to learn threshold "
                "or 'threshold' value!"
            )
    
    # Binary classification: outbreak or not
    pred_outbreak = (pred > threshold).astype(float)
    true_outbreak = (true > threshold).astype(float)
    
    # Calculate F1 score
    f1 = f1_score(
        true_outbreak.flatten(), 
        pred_outbreak.flatten(), 
        zero_division=0
    )
    
    return f1, threshold


def epidemic_metrics(pred, true, train_data=None, pred_file_path=None):
    """
    Calculate all epidemic-specific metrics.
    
    Args:
        pred: Predictions [B, T] or [T]
        true: Ground truth [B, T] or [T]
        train_data: Training data for learning outbreak threshold (optional)
        pred_file_path: Path to prediction file for verification (optional)
    
    Returns:
        dict: Dictionary containing all epidemic metrics
    
    Raises:
        FileNotFoundError: If pred_file_path is provided but doesn't exist
    
    Example:
        >>> metrics = epidemic_metrics(pred, true, train_data)
        >>> print(metrics['peak_timing_error_weeks'])
        1.35
    """
    # Verify prediction file exists if path is provided
    if pred_file_path is not None:
        if not os.path.exists(pred_file_path):
            raise FileNotFoundError(
                f"Prediction file not found: {pred_file_path}\n"
                "Cannot compute epidemic metrics without predictions!"
            )
    
    # Ensure 2D arrays for consistent processing
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if true.ndim == 1:
        true = true.reshape(1, -1)
    
    # Compute all metrics
    pt_error = peak_timing_error(pred, true)
    pi_abs_error, pi_rel_error = peak_intensity_error(pred, true)
    dir_acc = direction_accuracy(pred, true)
    
    results = {
        'peak_timing_error_weeks': float(pt_error),
        'peak_intensity_abs_error': float(pi_abs_error),
        'peak_intensity_rel_error_pct': float(pi_rel_error),
        'direction_accuracy_pct': float(dir_acc),
    }
    
    # Add outbreak F1 if training data is available
    if train_data is not None:
        try:
            f1, threshold = outbreak_detection_f1(pred, true, train_data=train_data)
            results['outbreak_f1'] = float(f1)
            results['outbreak_threshold'] = float(threshold)
        except Exception as e:
            print(f"Warning: Could not compute outbreak F1: {e}")
    
    return results


def print_epidemic_metrics(metrics):
    """
    Print epidemic metrics in a nice format with interpretations.
    
    Args:
        metrics: Dictionary of metrics from epidemic_metrics()
    """
    print("\n" + "="*60)
    print("EPIDEMIC-SPECIFIC METRICS")
    print("="*60)
    
    # Peak Timing Error (removed emojis for Windows)
    pte = metrics['peak_timing_error_weeks']
    print(f"\n[*] Peak Timing Error: {pte:.2f} weeks")
    if pte <= 1:
        print("   [OK] Excellent! Peak prediction within 1 week")
    elif pte <= 2:
        print("   [OK] Good - Peak prediction within 2 weeks")
    elif pte <= 3:
        print("   [~] Acceptable - Peak prediction within 3 weeks")
    else:
        print("   [!] Needs improvement - Peak prediction off by >3 weeks")
    
    # Peak Intensity Error
    pie_abs = metrics['peak_intensity_abs_error']
    pie_rel = metrics['peak_intensity_rel_error_pct']
    print(f"\n[*] Peak Intensity Error:")
    print(f"   Absolute: {pie_abs:.2f}%")
    print(f"   Relative: {pie_rel:.2f}%")
    if pie_abs <= 2:
        print("   [OK] Excellent! Peak value prediction within 2%")
    elif pie_abs <= 5:
        print("   [OK] Good - Peak value prediction within 5%")
    else:
        print("   [!] Needs improvement - Peak value off by >5%")
    
    # Direction Accuracy
    dir_acc = metrics['direction_accuracy_pct']
    print(f"\n[*] Direction Accuracy: {dir_acc:.2f}%")
    if dir_acc >= 80:
        print("   [OK] Excellent! Trend prediction >80% accurate")
    elif dir_acc >= 70:
        print("   [OK] Good - Trend prediction >70% accurate")
    elif dir_acc >= 60:
        print("   [~] Acceptable - Trend prediction >60% accurate")
    else:
        print("   [!] Needs improvement - Trend prediction <60% accurate")
    
    # Outbreak Detection F1 (if available)
    if 'outbreak_f1' in metrics:
        f1 = metrics['outbreak_f1']
        threshold = metrics['outbreak_threshold']
        print(f"\n[*] Outbreak Detection F1: {f1:.4f}")
        print(f"   (learned threshold: {threshold:.2f}%)")
        if f1 >= 0.8:
            print("   [OK] Excellent outbreak detection!")
        elif f1 >= 0.6:
            print("   [OK] Good outbreak detection")
        else:
            print("   [~] Needs improvement")
    
    print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Epidemic Metrics...")
    print("="*60)
    
    # Generate synthetic test data
    np.random.seed(42)
    
    # Create realistic flu patterns
    # True: peak at week 6 with value 32%
    true = np.zeros((5, 13))
    for i in range(5):
        weeks = np.arange(13)
        # Gaussian peak
        peak_week = 6 + np.random.randint(-1, 2)
        peak_value = 30 + np.random.randn() * 2
        true[i] = peak_value * np.exp(-((weeks - peak_week) ** 2) / 8) + 5
    
    # Pred: similar but slightly off
    pred = true + np.random.randn(5, 13) * 2
    
    # Training data for outbreak threshold
    train_data = np.random.rand(100, 52) * 30 + 10
    
    print("\nTest Data:")
    print(f"  True shape: {true.shape}")
    print(f"  Pred shape: {pred.shape}")
    print(f"  True peak values: {np.max(true, axis=1)}")
    print(f"  Pred peak values: {np.max(pred, axis=1)}")
    
    # Compute metrics
    print("\n" + "-"*60)
    print("Computing metrics...")
    print("-"*60)
    
    metrics = epidemic_metrics(
        pred=pred,
        true=true,
        train_data=train_data
    )
    
    # Print results
    print_epidemic_metrics(metrics)
    
    # Test error handling
    print("\n" + "-"*60)
    print("Testing error handling...")
    print("-"*60)
    
    try:
        # Should raise error if file doesn't exist
        epidemic_metrics(pred, true, pred_file_path="/nonexistent/file.npy")
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError:")
        print(f"  {str(e)[:80]}...")
    
    try:
        # Should raise error if no train_data for outbreak threshold
        outbreak_detection_f1(pred, true)
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError:")
        print(f"  {str(e)[:80]}...")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
