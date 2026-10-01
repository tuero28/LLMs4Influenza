"""
Basic test for EnsemblePredictor class

This script tests the core functionality of the EnsemblePredictor without
requiring actual checkpoint files, using mock objects for quick validation.
"""

import torch
import torch.nn as nn
from ensemble_predictor import (
    EnsemblePredictor,
    weighted_average,
    discover_checkpoints,
    parse_ensemble_weights,
    validate_weights,
    normalize_weights,
    adjust_weights_for_missing_models
)


class DummyArgs:
    """Dummy args object for testing"""
    def __init__(self):
        self.ensemble_weights = "0.3,0.3,0.4"
        self.checkpoints = "./checkpoints/"
        self.seq_len = 336
        self.pred_len = 8
        self.label_len = 18
        self.enc_in = 1
        self.c_out = 1
        self.d_model = 768
        self.n_heads = 16
        self.e_layers = 3
        self.d_ff = 512
        self.dropout = 0.2
        self.embed = 'timeF'
        self.freq = 1
        self.patch_size = 16
        self.stride = 8
        self.kernel_size = 25
        self.is_gpt = 1
        self.pretrain = 0
        self.freeze = 0
        self.llama_layers = 32


def test_weighted_average():
    """Test weighted_average function"""
    print("\n=== Testing weighted_average ===")
    
    # Create sample predictions
    pred1 = torch.randn(2, 8, 1)  # [batch=2, pred_len=8, features=1]
    pred2 = torch.randn(2, 8, 1)
    pred3 = torch.randn(2, 8, 1)
    
    weights = [0.3, 0.3, 0.4]
    
    # Test weighted average
    ensemble = weighted_average([pred1, pred2, pred3], weights)
    
    # Verify shape
    assert ensemble.shape == pred1.shape, f"Shape mismatch: {ensemble.shape} vs {pred1.shape}"
    print(f"✓ Shape correct: {ensemble.shape}")
    
    # Verify computation (manual check)
    expected = 0.3 * pred1 + 0.3 * pred2 + 0.4 * pred3
    assert torch.allclose(ensemble, expected), "Weighted average computation incorrect"
    print("✓ Weighted average computed correctly")
    
    # Test error cases
    try:
        # Mismatched shapes
        weighted_average([pred1, torch.randn(2, 10, 1)], [0.5, 0.5])
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        print(f"✓ Shape mismatch error caught: {str(e)[:50]}...")
    
    try:
        # Wrong number of weights
        weighted_average([pred1, pred2], [0.5, 0.3, 0.2])
        assert False, "Should have raised ValueError for weight count mismatch"
    except ValueError as e:
        print(f"✓ Weight count mismatch error caught: {str(e)[:50]}...")
    
    print("✓ All weighted_average tests passed!")


def test_weight_functions():
    """Test weight validation and normalization"""
    print("\n=== Testing weight functions ===")
    
    # Test validate_weights
    assert validate_weights([0.3, 0.3, 0.4]) == True
    print("✓ Valid weights pass validation")
    
    assert validate_weights([0.33, 0.33, 0.34]) == True
    print("✓ Weights within tolerance pass validation")
    
    try:
        validate_weights([0.3, -0.1, 0.8])
        assert False, "Should reject negative weights"
    except ValueError:
        print("✓ Negative weights rejected")
    
    try:
        validate_weights([0.0, 0.0, 0.0])
        assert False, "Should reject zero sum"
    except ValueError:
        print("✓ Zero sum weights rejected")
    
    # Test normalize_weights
    normalized = normalize_weights([0.3, 0.3, 0.6])
    expected = [0.25, 0.25, 0.5]
    assert all(abs(a - b) < 0.001 for a, b in zip(normalized, expected))
    print(f"✓ Normalization works: {normalized}")
    
    # Test adjust_weights_for_missing_models
    original = {'DLinear': 0.3, 'PatchTST': 0.3, 'Llama2': 0.4}
    available = ['DLinear', 'Llama2']
    adjusted = adjust_weights_for_missing_models(original, available)
    
    assert set(adjusted.keys()) == set(available)
    assert abs(sum(adjusted.values()) - 1.0) < 0.001
    assert abs(adjusted['DLinear'] / adjusted['Llama2'] - 0.3 / 0.4) < 0.001
    print(f"✓ Weight adjustment works: {adjusted}")
    
    print("✓ All weight function tests passed!")


def test_parse_ensemble_weights():
    """Test parse_ensemble_weights function"""
    print("\n=== Testing parse_ensemble_weights ===")
    
    # Valid inputs
    weights = parse_ensemble_weights("0.3,0.3,0.4")
    assert weights == [0.3, 0.3, 0.4]
    print("✓ Basic parsing works")
    
    weights = parse_ensemble_weights(" 0.25 , 0.35 , 0.40 ")
    assert weights == [0.25, 0.35, 0.40]
    print("✓ Whitespace handling works")
    
    # Invalid inputs
    try:
        parse_ensemble_weights("0.5,0.5")
        assert False, "Should reject wrong count"
    except ValueError:
        print("✓ Wrong count rejected")
    
    try:
        parse_ensemble_weights("0.3,abc,0.4")
        assert False, "Should reject non-numeric"
    except ValueError:
        print("✓ Non-numeric rejected")
    
    try:
        parse_ensemble_weights("0.3,-0.1,0.8")
        assert False, "Should reject negative"
    except ValueError:
        print("✓ Negative values rejected")
    
    print("✓ All parse_ensemble_weights tests passed!")


def test_ensemble_predictor_init():
    """Test EnsemblePredictor initialization"""
    print("\n=== Testing EnsemblePredictor initialization ===")
    
    args = DummyArgs()
    device = torch.device('cpu')
    
    # Test initialization
    predictor = EnsemblePredictor(args, device)
    
    assert predictor.device == device
    assert predictor.weights == {'DLinear': 0.3, 'PatchTST': 0.3, 'Llama2': 0.4}
    assert len(predictor.models) == 0  # No models loaded yet
    print("✓ Initialization works")
    
    # Test with custom weights
    args.ensemble_weights = "0.25,0.35,0.40"
    predictor = EnsemblePredictor(args, device)
    assert predictor.weights == {'DLinear': 0.25, 'PatchTST': 0.35, 'Llama2': 0.40}
    print("✓ Custom weights work")
    
    # Test weight normalization
    args.ensemble_weights = "0.3,0.3,0.6"
    predictor = EnsemblePredictor(args, device)
    expected = {'DLinear': 0.25, 'PatchTST': 0.25, 'Llama2': 0.5}
    for key in expected:
        assert abs(predictor.weights[key] - expected[key]) < 0.001
    print("✓ Weight normalization works")
    
    print("✓ All EnsemblePredictor initialization tests passed!")


def test_discover_checkpoints():
    """Test checkpoint discovery function"""
    print("\n=== Testing discover_checkpoints ===")
    
    # This will fail for missing checkpoints but should not crash
    model_types = ['DLinear', 'PatchTST', 'Llama2']
    paths = discover_checkpoints('./checkpoints/', model_types, iteration=0)
    
    print(f"✓ Discovered {len(paths)} checkpoints: {list(paths.keys())}")
    
    # Verify path format
    for model_type, path in paths.items():
        assert 'checkpoint.pth' in path
        assert model_type in path
        print(f"  - {model_type}: {path}")
    
    print("✓ Checkpoint discovery test passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("BASIC ENSEMBLE PREDICTOR TESTS")
    print("=" * 60)
    
    try:
        test_weighted_average()
        test_weight_functions()
        test_parse_ensemble_weights()
        test_ensemble_predictor_init()
        test_discover_checkpoints()
        
        print("\n" + "=" * 60)
        print("✓ ALL BASIC TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
