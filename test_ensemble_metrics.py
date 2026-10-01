"""
Test suite for ensemble metrics and visualization functionality

Tests the performance evaluation and visualization components of the ensemble predictor.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ensemble_predictor import (
    EnsembleMetrics,
    compute_improvement_percentage,
    print_comparison_table,
    visualize_predictions,
    save_visualization
)
import os
import tempfile


def test_compute_improvement_percentage():
    """Test improvement percentage calculation"""
    print("\n=== Testing compute_improvement_percentage ===")
    
    # Test improvement (lower is better for MSE/MAE)
    improvement = compute_improvement_percentage(0.050, 0.045)
    assert abs(improvement - 10.0) < 0.01, f"Expected 10.0%, got {improvement}%"
    print(f"✓ Improvement calculation: {improvement:.2f}%")
    
    # Test degradation
    degradation = compute_improvement_percentage(0.050, 0.055)
    assert abs(degradation - (-10.0)) < 0.01, f"Expected -10.0%, got {degradation}%"
    print(f"✓ Degradation calculation: {degradation:.2f}%")
    
    # Test no change
    no_change = compute_improvement_percentage(0.050, 0.050)
    assert abs(no_change - 0.0) < 0.01, f"Expected 0.0%, got {no_change}%"
    print(f"✓ No change calculation: {no_change:.2f}%")
    
    # Test zero baseline
    zero_baseline = compute_improvement_percentage(0.0, 0.045)
    assert zero_baseline == 0.0, "Should return 0 for zero baseline"
    print("✓ Zero baseline handled correctly")
    
    print("✓ All compute_improvement_percentage tests passed!")


def test_ensemble_metrics():
    """Test EnsembleMetrics dataclass"""
    print("\n=== Testing EnsembleMetrics ===")
    
    # Create metrics instance
    metrics = EnsembleMetrics(
        ensemble_mse=0.045,
        ensemble_mae=0.15,
        dlinear_mse=0.052,
        dlinear_mae=0.17,
        patchtst_mse=0.048,
        patchtst_mae=0.16,
        llama2_mse=0.050,
        llama2_mae=0.165
    )
    
    # Calculate improvements
    metrics.mse_improvement['DLinear'] = compute_improvement_percentage(
        metrics.dlinear_mse, metrics.ensemble_mse
    )
    metrics.mae_improvement['DLinear'] = compute_improvement_percentage(
        metrics.dlinear_mae, metrics.ensemble_mae
    )
    
    metrics.mse_improvement['PatchTST'] = compute_improvement_percentage(
        metrics.patchtst_mse, metrics.ensemble_mse
    )
    metrics.mae_improvement['PatchTST'] = compute_improvement_percentage(
        metrics.patchtst_mae, metrics.ensemble_mae
    )
    
    metrics.mse_improvement['Llama2'] = compute_improvement_percentage(
        metrics.llama2_mse, metrics.ensemble_mse
    )
    metrics.mae_improvement['Llama2'] = compute_improvement_percentage(
        metrics.llama2_mae, metrics.ensemble_mae
    )
    
    print(f"✓ Ensemble MSE: {metrics.ensemble_mse}")
    print(f"✓ Ensemble MAE: {metrics.ensemble_mae}")
    print(f"✓ MSE improvements: {metrics.mse_improvement}")
    print(f"✓ MAE improvements: {metrics.mae_improvement}")
    
    # Verify improvements are positive (ensemble should be better)
    assert all(imp > 0 for imp in metrics.mse_improvement.values()), \
        "All MSE improvements should be positive"
    assert all(imp > 0 for imp in metrics.mae_improvement.values()), \
        "All MAE improvements should be positive"
    
    print("✓ All EnsembleMetrics tests passed!")
    
    return metrics


def test_print_comparison_table(metrics):
    """Test comparison table printing"""
    print("\n=== Testing print_comparison_table ===")
    
    # This will print to console
    print_comparison_table(metrics)
    
    print("✓ Comparison table printed successfully!")


def test_visualize_predictions():
    """Test visualization generation"""
    print("\n=== Testing visualize_predictions ===")
    
    # Create dummy predictions
    batch_size = 5
    pred_len = 8
    features = 1
    
    ground_truth = np.random.randn(batch_size, pred_len, features) * 0.1 + 0.5
    
    predictions = {
        'ensemble': np.random.randn(batch_size, pred_len, features) * 0.08 + 0.5,
        'DLinear': np.random.randn(batch_size, pred_len, features) * 0.12 + 0.5,
        'PatchTST': np.random.randn(batch_size, pred_len, features) * 0.10 + 0.5,
        'Llama2': np.random.randn(batch_size, pred_len, features) * 0.09 + 0.5
    }
    
    # Create dummy args
    class DummyArgs:
        pass
    
    args = DummyArgs()
    args.pred_len = pred_len
    
    # Generate visualization
    fig = visualize_predictions(
        predictions=predictions,
        ground_truth=ground_truth,
        args=args,
        output_path=None,
        sample_idx=0
    )
    
    assert fig is not None, "Figure should not be None"
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    
    print("✓ Visualization generated successfully")
    print(f"✓ Figure size: {fig.get_size_inches()}")
    
    # Close figure to free memory
    plt.close(fig)
    
    print("✓ All visualize_predictions tests passed!")


def test_save_visualization():
    """Test visualization saving"""
    print("\n=== Testing save_visualization ===")
    
    # Create a simple figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([1, 2, 3], [1, 2, 3], 'b-', label='Test')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Figure')
    ax.legend()
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test_plot.png')
        save_visualization(fig, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path), f"Output file not created: {output_path}"
        
        # Verify file has content
        file_size = os.path.getsize(output_path)
        assert file_size > 0, f"Output file is empty: {output_path}"
        
        print(f"✓ Visualization saved to: {output_path}")
        print(f"✓ File size: {file_size} bytes")
    
    plt.close(fig)
    
    print("✓ All save_visualization tests passed!")


def test_end_to_end_workflow():
    """Test complete workflow: metrics -> table -> visualization"""
    print("\n=== Testing end-to-end workflow ===")
    
    # 1. Create metrics
    metrics = EnsembleMetrics(
        ensemble_mse=0.042,
        ensemble_mae=0.145,
        dlinear_mse=0.050,
        dlinear_mae=0.165,
        patchtst_mse=0.047,
        patchtst_mae=0.158,
        llama2_mse=0.048,
        llama2_mae=0.160
    )
    
    # 2. Calculate improvements
    for model, mse in [('DLinear', 0.050), ('PatchTST', 0.047), ('Llama2', 0.048)]:
        metrics.mse_improvement[model] = compute_improvement_percentage(
            mse, metrics.ensemble_mse
        )
    
    for model, mae in [('DLinear', 0.165), ('PatchTST', 0.158), ('Llama2', 0.160)]:
        metrics.mae_improvement[model] = compute_improvement_percentage(
            mae, metrics.ensemble_mae
        )
    
    # 3. Print comparison table
    print_comparison_table(metrics)
    
    # 4. Create visualization
    batch_size = 3
    pred_len = 13
    
    ground_truth = np.random.randn(batch_size, pred_len, 1) * 0.05 + 0.3
    predictions = {
        'Ensemble': np.random.randn(batch_size, pred_len, 1) * 0.04 + 0.3,
        'DLinear': np.random.randn(batch_size, pred_len, 1) * 0.06 + 0.3,
        'PatchTST': np.random.randn(batch_size, pred_len, 1) * 0.055 + 0.3,
        'Llama2': np.random.randn(batch_size, pred_len, 1) * 0.05 + 0.3
    }
    
    class DummyArgs:
        pass
    
    dummy_args = DummyArgs()
    dummy_args.pred_len = pred_len
    
    # 5. Generate and save visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'ensemble_comparison.png')
        fig = visualize_predictions(
            predictions=predictions,
            ground_truth=ground_truth,
            args=dummy_args,
            output_path=output_path,
            sample_idx=0
        )
        
        assert os.path.exists(output_path), "Visualization file not created"
        print(f"✓ End-to-end workflow completed, output: {output_path}")
        
        plt.close(fig)
    
    print("✓ All end-to-end workflow tests passed!")


def main():
    """Run all tests"""
    print("=" * 70)
    print("ENSEMBLE METRICS AND VISUALIZATION TESTS")
    print("=" * 70)
    
    try:
        test_compute_improvement_percentage()
        metrics = test_ensemble_metrics()
        test_print_comparison_table(metrics)
        test_visualize_predictions()
        test_save_visualization()
        test_end_to_end_workflow()
        
        print("\n" + "=" * 70)
        print("✓ ALL METRICS AND VISUALIZATION TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
