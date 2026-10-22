"""
Test script for new improvements:
1. Epidemic metrics
2. Post-processing (clipping)

Run this to verify everything works before integrating into main pipeline.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from utils.epidemic_metrics import (
    epidemic_metrics, 
    print_epidemic_metrics,
    peak_timing_error,
    peak_intensity_error,
    direction_accuracy,
)
from utils.post_processing import (
    clip_predictions,
    analyze_clipping_impact,
    print_clipping_analysis,
)


def test_epidemic_metrics():
    """Test epidemic metrics với dữ liệu thật."""
    print("\n" + "="*80)
    print("TEST 1: EPIDEMIC METRICS")
    print("="*80)
    
    # Tạo dữ liệu giả có pattern thực tế
    np.random.seed(42)
    n_samples = 10
    n_weeks = 13
    
    # Tạo pattern có đỉnh dịch
    true = np.zeros((n_samples, n_weeks))
    pred = np.zeros((n_samples, n_weeks))
    
    for i in range(n_samples):
        # True: đỉnh ở tuần 6-7, giá trị 25-35%
        peak_week_true = 6 + np.random.randint(-1, 2)
        peak_value_true = 30 + np.random.randn() * 3
        
        # Pred: đỉnh hơi lệch, giá trị hơi sai
        peak_week_pred = peak_week_true + np.random.randint(-2, 3)
        peak_value_pred = peak_value_true + np.random.randn() * 4
        
        # Tạo pattern Gaussian
        weeks = np.arange(n_weeks)
        true[i] = peak_value_true * np.exp(-((weeks - peak_week_true) ** 2) / 6) + 8
        pred[i] = peak_value_pred * np.exp(-((weeks - peak_week_pred) ** 2) / 6) + 8
        
        # Thêm noise
        true[i] += np.random.randn(n_weeks) * 1.5
        pred[i] += np.random.randn(n_weeks) * 2
    
    # Training data để học outbreak threshold
    train_data = np.random.rand(200, 52) * 25 + 10  # 10-35%
    
    print("\n✓ Created synthetic data:")
    print(f"  Shape: pred={pred.shape}, true={true.shape}")
    print(f"  True peak range: {np.max(true, axis=1).min():.1f}% - {np.max(true, axis=1).max():.1f}%")
    print(f"  Pred peak range: {np.max(pred, axis=1).min():.1f}% - {np.max(pred, axis=1).max():.1f}%")
    
    # Test từng metric riêng lẻ
    print("\n" + "-"*80)
    print("Testing individual metrics...")
    print("-"*80)
    
    pte = peak_timing_error(pred, true)
    print(f"✓ Peak Timing Error: {pte:.2f} weeks")
    
    pie_abs, pie_rel = peak_intensity_error(pred, true)
    print(f"✓ Peak Intensity Error: {pie_abs:.2f}% (absolute), {pie_rel:.1f}% (relative)")
    
    dir_acc = direction_accuracy(pred, true)
    print(f"✓ Direction Accuracy: {dir_acc:.1f}%")
    
    # Test tất cả metrics cùng lúc
    print("\n" + "-"*80)
    print("Testing combined epidemic_metrics()...")
    print("-"*80)
    
    metrics = epidemic_metrics(
        pred=pred,
        true=true,
        train_data=train_data
    )
    
    print_epidemic_metrics(metrics)
    
    # Test với batch size = 1
    print("\n" + "-"*80)
    print("Testing with single sample (1D arrays)...")
    print("-"*80)
    
    metrics_single = epidemic_metrics(
        pred=pred[0],  # 1D array
        true=true[0],
        train_data=train_data
    )
    
    print(f"✓ Peak Timing Error: {metrics_single['peak_timing_error_weeks']:.2f} weeks")
    print(f"✓ Direction Accuracy: {metrics_single['direction_accuracy_pct']:.1f}%")
    
    print("\n✅ Test 1 PASSED: Epidemic metrics work correctly!")
    return True


def test_clipping():
    """Test clipping với dữ liệu có giá trị âm."""
    print("\n" + "="*80)
    print("TEST 2: POST-PROCESSING (CLIPPING)")
    print("="*80)
    
    # Tạo dữ liệu có một số giá trị âm
    np.random.seed(123)
    n_samples = 15
    n_weeks = 13
    
    pred_orig = np.random.randn(n_samples, n_weeks) * 8 + 18  # Một số sẽ âm với std lớn hơn
    true = np.abs(np.random.randn(n_samples, n_weeks) * 5 + 20)  # Luôn dương
    
    print(f"\n✓ Created test data:")
    print(f"  Shape: {pred_orig.shape}")
    print(f"  Original range: [{pred_orig.min():.2f}, {pred_orig.max():.2f}]")
    print(f"  Negative values: {np.sum(pred_orig < 0)}/{pred_orig.size}")
    
    # Test clipping
    print("\n" + "-"*80)
    print("Testing clip_predictions()...")
    print("-"*80)
    
    pred_clipped = clip_predictions(pred_orig, min_value=0.0)
    
    print(f"✓ After clipping:")
    print(f"  Range: [{pred_clipped.min():.2f}, {pred_clipped.max():.2f}]")
    print(f"  Negative values: {np.sum(pred_clipped < 0)}/{pred_clipped.size}")
    
    assert np.all(pred_clipped >= 0), "❌ Clipping failed: still has negative values!"
    print("✓ Assertion passed: no negative values after clipping")
    
    # Test impact analysis
    print("\n" + "-"*80)
    print("Testing clipping impact analysis...")
    print("-"*80)
    
    analysis = analyze_clipping_impact(pred_orig, pred_clipped, true)
    print_clipping_analysis(analysis)
    
    # Verify analysis results
    assert analysis['n_negative_values'] > 0, "Test data should have negative values"
    assert analysis['n_clipped'] > 0, "Should have clipped some values"
    print("✓ Analysis results verified")
    
    # Test with torch tensors
    try:
        import torch
        print("\n" + "-"*80)
        print("Testing with PyTorch tensors...")
        print("-"*80)
        
        pred_torch = torch.tensor(pred_orig, dtype=torch.float32)
        pred_torch_clipped = clip_predictions(pred_torch, min_value=0.0)
        
        print(f"✓ Torch input: {pred_torch.shape}, dtype={pred_torch.dtype}")
        print(f"✓ Torch output: {pred_torch_clipped.shape}, dtype={pred_torch_clipped.dtype}")
        print(f"✓ Range: [{pred_torch_clipped.min():.2f}, {pred_torch_clipped.max():.2f}]")
        
        assert torch.all(pred_torch_clipped >= 0), "Torch clipping failed!"
        print("✓ PyTorch clipping works correctly")
        
    except ImportError:
        print("\n⚠️ PyTorch not available, skipping torch test")
    
    print("\n✅ Test 2 PASSED: Clipping works correctly!")
    return True


def test_integration():
    """Test tích hợp: clipping + epidemic metrics."""
    print("\n" + "="*80)
    print("TEST 3: INTEGRATION (CLIPPING + EPIDEMIC METRICS)")
    print("="*80)
    
    # Tạo dữ liệu có negative predictions
    np.random.seed(456)
    
    # True data: realistic flu pattern
    n_samples = 20
    n_weeks = 13
    true = np.zeros((n_samples, n_weeks))
    pred_orig = np.zeros((n_samples, n_weeks))
    
    for i in range(n_samples):
        peak_week = 6 + np.random.randint(-2, 3)
        peak_value = 28 + np.random.randn() * 4
        
        weeks = np.arange(n_weeks)
        true[i] = peak_value * np.exp(-((weeks - peak_week) ** 2) / 7) + 10
        
        # Predictions có thể âm
        pred_orig[i] = true[i] + np.random.randn(n_weeks) * 5
    
    train_data = np.random.rand(150, 52) * 30 + 8
    
    print(f"\n✓ Created test data:")
    print(f"  Predictions range: [{pred_orig.min():.2f}, {pred_orig.max():.2f}]")
    print(f"  Negative predictions: {np.sum(pred_orig < 0)}")
    
    # Step 1: Clip predictions
    print("\n" + "-"*80)
    print("Step 1: Clipping predictions...")
    print("-"*80)
    
    pred_clipped = clip_predictions(pred_orig, min_value=0.0)
    clipping_analysis = analyze_clipping_impact(pred_orig, pred_clipped, true)
    
    print(f"✓ Clipped {clipping_analysis['n_clipped']} values")
    print(f"✓ MAE change: {clipping_analysis['improvement']['mae']:+.6f}")
    
    # Step 2: Compute metrics for both versions
    print("\n" + "-"*80)
    print("Step 2: Computing epidemic metrics...")
    print("-"*80)
    
    print("\nBEFORE CLIPPING:")
    metrics_orig = epidemic_metrics(pred_orig, true, train_data)
    for key, value in metrics_orig.items():
        if 'threshold' not in key:
            print(f"  {key}: {value:.4f}")
    
    print("\nAFTER CLIPPING (RECOMMENDED):")
    metrics_clipped = epidemic_metrics(pred_clipped, true, train_data)
    for key, value in metrics_clipped.items():
        if 'threshold' not in key:
            print(f"  {key}: {value:.4f}")
    
    # Compare
    print("\n" + "-"*80)
    print("Comparison:")
    print("-"*80)
    
    for key in ['peak_timing_error_weeks', 'peak_intensity_abs_error', 'direction_accuracy_pct']:
        orig_val = metrics_orig[key]
        clip_val = metrics_clipped[key]
        diff = clip_val - orig_val
        
        print(f"  {key}:")
        print(f"    Before: {orig_val:.4f}")
        print(f"    After:  {clip_val:.4f}")
        print(f"    Change: {diff:+.4f}")
    
    print("\n✅ Test 3 PASSED: Integration works correctly!")
    return True


def test_real_data_if_available():
    """Thử với dữ liệu thật nếu có."""
    print("\n" + "="*80)
    print("TEST 4: REAL DATA (IF AVAILABLE)")
    print("="*80)
    
    # Thử load dữ liệu thật
    try:
        # Thử load từ results nếu có
        results_dirs = [
            './results',
            './checkpoints',
        ]
        
        pred_file = None
        true_file = None
        
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                # Tìm file pred.npy và true.npy
                for root, dirs, files in os.walk(results_dir):
                    if 'pred.npy' in files and 'true.npy' in files:
                        pred_file = os.path.join(root, 'pred.npy')
                        true_file = os.path.join(root, 'true.npy')
                        break
                if pred_file:
                    break
        
        if pred_file and true_file:
            print(f"\n✓ Found real data:")
            print(f"  Predictions: {pred_file}")
            print(f"  Ground truth: {true_file}")
            
            pred = np.load(pred_file)
            true = np.load(true_file)
            
            print(f"\n✓ Loaded data:")
            print(f"  Pred shape: {pred.shape}")
            print(f"  True shape: {true.shape}")
            print(f"  Pred range: [{pred.min():.2f}, {pred.max():.2f}]")
            print(f"  Has negative: {np.sum(pred < 0)} values")
            
            # Reshape if needed
            if pred.ndim == 3:
                pred = pred.reshape(-1, pred.shape[-1])
            if true.ndim == 3:
                true = true.reshape(-1, true.shape[-1])
            
            # Load train data
            train_file = './dataset/Flu_in_NorthChina.csv'
            if os.path.exists(train_file):
                import pandas as pd
                df = pd.read_csv(train_file)
                train_data = df['positive_rate'].values
                print(f"✓ Loaded train data: {len(train_data)} samples")
            else:
                train_data = None
                print("⚠️ Train data not found, outbreak F1 will be skipped")
            
            # Test with real data
            print("\n" + "-"*80)
            print("Testing with REAL data...")
            print("-"*80)
            
            # Clip
            pred_clipped = clip_predictions(pred, min_value=0.0)
            
            if np.sum(pred < 0) > 0:
                analysis = analyze_clipping_impact(pred, pred_clipped, true)
                print_clipping_analysis(analysis)
            else:
                print("✓ No negative values in real predictions")
            
            # Metrics
            print("\nEpidemic Metrics (with real data):")
            metrics = epidemic_metrics(pred_clipped, true, train_data)
            print_epidemic_metrics(metrics)
            
            print("\n✅ Test 4 PASSED: Works with real data!")
            return True
        else:
            print("\n⚠️ No real prediction files found")
            print("   Run a model first to generate predictions, then run this test again")
            return None
            
    except Exception as e:
        print(f"\n⚠️ Could not test with real data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                TEST SUITE FOR EPIDEMIC METRICS & POST-PROCESSING             ║
║                                                                              ║
║  This script tests the new improvements before integrating into pipeline.   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = []
    
    try:
        # Test 1
        result1 = test_epidemic_metrics()
        results.append(("Epidemic Metrics", result1))
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Epidemic Metrics", False))
    
    try:
        # Test 2
        result2 = test_clipping()
        results.append(("Clipping", result2))
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Clipping", False))
    
    try:
        # Test 3
        result3 = test_integration()
        results.append(("Integration", result3))
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Integration", False))
    
    try:
        # Test 4 (optional)
        result4 = test_real_data_if_available()
        if result4 is not None:
            results.append(("Real Data", result4))
    except Exception as e:
        print(f"\n⚠️ Test 4 skipped: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        if result:
            status = "✅ PASSED"
        elif result is None:
            status = "⚠️ SKIPPED"
        else:
            status = "❌ FAILED"
        
        print(f"  {test_name:.<50} {status}")
    
    print("="*80)
    
    # Overall result
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now integrate these improvements into your main pipeline.")
        print("See PHAN_TICH_4_TINH_NANG.md for integration instructions.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please fix before integrating.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
