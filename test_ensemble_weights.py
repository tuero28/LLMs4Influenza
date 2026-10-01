"""
Test suite for parse_ensemble_weights function

This module tests the weight parsing and validation functionality for the ensemble
predictor, ensuring compliance with Requirements 9.2, 9.3, 9.4, 9.5.

Tests cover:
- Valid weight strings with exactly 3 non-negative floats
- Invalid weight counts (too few, too many)
- Invalid weight values (non-numeric, negative)
- Edge cases (empty strings, whitespace handling, zero values)
"""

import pytest
from ensemble_predictor import parse_ensemble_weights


class TestParseEnsembleWeights:
    """Test suite for parse_ensemble_weights function.
    
    **Validates: Requirements 9.2, 9.3, 9.4, 9.5**
    """
    
    # ========================================================================
    # Valid Input Tests (Requirement 9.2)
    # ========================================================================
    
    def test_parse_valid_default_weights(self):
        """Test parsing default weight string '0.3,0.3,0.4'."""
        result = parse_ensemble_weights("0.3,0.3,0.4")
        assert result == [0.3, 0.3, 0.4], "Should parse default weights correctly"
    
    def test_parse_valid_custom_weights(self):
        """Test parsing custom weight string '0.25,0.35,0.40'."""
        result = parse_ensemble_weights("0.25,0.35,0.40")
        assert result == [0.25, 0.35, 0.40], "Should parse custom weights correctly"
    
    def test_parse_valid_equal_weights(self):
        """Test parsing equal weights '0.333,0.333,0.334'."""
        result = parse_ensemble_weights("0.333,0.333,0.334")
        assert result == [0.333, 0.333, 0.334], "Should parse equal weights correctly"
    
    def test_parse_weights_with_whitespace(self):
        """Test parsing weights with extra whitespace."""
        result = parse_ensemble_weights(" 0.3 , 0.3 , 0.4 ")
        assert result == [0.3, 0.3, 0.4], "Should handle whitespace correctly"
    
    def test_parse_weights_with_leading_trailing_spaces(self):
        """Test parsing weights with leading/trailing spaces."""
        result = parse_ensemble_weights("  0.25,0.35,0.40  ")
        assert result == [0.25, 0.35, 0.40], "Should trim leading/trailing spaces"
    
    def test_parse_integer_weights(self):
        """Test parsing integer values (which are valid floats)."""
        result = parse_ensemble_weights("1,0,0")
        assert result == [1.0, 0.0, 0.0], "Should parse integer weights as floats"
    
    def test_parse_zero_weights(self):
        """Test parsing zero values (valid but will need normalization)."""
        result = parse_ensemble_weights("0,0.5,0.5")
        assert result == [0.0, 0.5, 0.5], "Should accept zero weights"
    
    def test_parse_scientific_notation(self):
        """Test parsing weights in scientific notation."""
        result = parse_ensemble_weights("3e-1,3e-1,4e-1")
        assert result == [0.3, 0.3, 0.4], "Should parse scientific notation"
    
    def test_parse_very_small_weights(self):
        """Test parsing very small positive weights."""
        result = parse_ensemble_weights("0.0001,0.0002,0.9997")
        assert result == [0.0001, 0.0002, 0.9997], "Should parse very small weights"
    
    def test_parse_weights_summing_above_one(self):
        """Test parsing weights that sum above 1.0 (will be normalized later)."""
        result = parse_ensemble_weights("0.5,0.5,0.5")
        assert result == [0.5, 0.5, 0.5], "Should parse weights summing above 1.0"
    
    def test_parse_weights_summing_below_one(self):
        """Test parsing weights that sum below 1.0 (will be normalized later)."""
        result = parse_ensemble_weights("0.1,0.1,0.1")
        assert result == [0.1, 0.1, 0.1], "Should parse weights summing below 1.0"
    
    # ========================================================================
    # Invalid Count Tests (Requirement 9.4)
    # ========================================================================
    
    def test_parse_too_few_weights_one(self):
        """Test error when only 1 weight provided (Requirement 9.4)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.5")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg
        assert "got 1" in error_msg
    
    def test_parse_too_few_weights_two(self):
        """Test error when only 2 weights provided (Requirement 9.4)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.5,0.5")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg
        assert "got 2" in error_msg
    
    def test_parse_too_many_weights_four(self):
        """Test error when 4 weights provided (Requirement 9.4)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.25,0.25,0.25,0.25")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg
        assert "got 4" in error_msg
    
    def test_parse_too_many_weights_five(self):
        """Test error when 5 weights provided (Requirement 9.4)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.2,0.2,0.2,0.2,0.2")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg
        assert "got 5" in error_msg
    
    def test_parse_empty_string(self):
        """Test error when empty string provided."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg or "could not convert" in error_msg
    
    # ========================================================================
    # Invalid Value Tests (Requirement 9.3)
    # ========================================================================
    
    def test_parse_non_numeric_first_position(self):
        """Test error with non-numeric value in first position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("abc,0.3,0.4")
        
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg
        assert "abc" in error_msg
        assert "could not convert string to float" in error_msg
    
    def test_parse_non_numeric_second_position(self):
        """Test error with non-numeric value in second position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,xyz,0.4")
        
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg
        assert "xyz" in error_msg
        assert "could not convert string to float" in error_msg
    
    def test_parse_non_numeric_third_position(self):
        """Test error with non-numeric value in third position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,0.3,invalid")
        
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg
        assert "invalid" in error_msg
        assert "could not convert string to float" in error_msg
    
    def test_parse_negative_first_position(self):
        """Test error with negative value in first position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("-0.1,0.5,0.6")
        
        error_msg = str(exc_info.value)
        assert "must be non-negative" in error_msg
        assert "-0.1" in error_msg
    
    def test_parse_negative_second_position(self):
        """Test error with negative value in second position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,-0.2,0.9")
        
        error_msg = str(exc_info.value)
        assert "must be non-negative" in error_msg
        assert "-0.2" in error_msg
    
    def test_parse_negative_third_position(self):
        """Test error with negative value in third position (Requirement 9.3)."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.4,0.4,-0.3")
        
        error_msg = str(exc_info.value)
        assert "must be non-negative" in error_msg
        assert "-0.3" in error_msg
    
    def test_parse_multiple_negative_values(self):
        """Test error with multiple negative values."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("-0.1,-0.2,-0.3")
        
        error_msg = str(exc_info.value)
        assert "must be non-negative" in error_msg
    
    def test_parse_special_characters(self):
        """Test error with special characters."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,@#$,0.4")
        
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg
        assert "could not convert string to float" in error_msg
    
    def test_parse_mixed_valid_invalid(self):
        """Test error with mix of valid and invalid values."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,hello,0.4")
        
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg
        assert "hello" in error_msg
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_parse_all_zeros(self):
        """Test parsing all zero weights (valid but may cause issues in normalization)."""
        result = parse_ensemble_weights("0,0,0")
        assert result == [0.0, 0.0, 0.0], "Should parse all zeros"
    
    def test_parse_very_large_weights(self):
        """Test parsing very large weight values."""
        result = parse_ensemble_weights("1000,2000,3000")
        assert result == [1000.0, 2000.0, 3000.0], "Should parse large weights"
    
    def test_parse_decimal_precision(self):
        """Test parsing weights with high decimal precision."""
        result = parse_ensemble_weights("0.123456789,0.234567890,0.345678901")
        assert result == [0.123456789, 0.234567890, 0.345678901], "Should preserve decimal precision"
    
    def test_parse_missing_comma(self):
        """Test error when comma separator is missing."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3 0.3 0.4")
        
        # This should fail because "0.3 0.3 0.4" will be treated as a single non-numeric value
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg or "could not convert" in error_msg
    
    def test_parse_extra_commas(self):
        """Test error when extra commas are present."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,,0.3,0.4")
        
        # Empty string between commas will either fail count validation or float conversion
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg or "could not convert" in error_msg
    
    def test_parse_trailing_comma(self):
        """Test error when trailing comma is present."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,0.3,0.4,")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg or "could not convert" in error_msg
    
    def test_parse_leading_comma(self):
        """Test error when leading comma is present."""
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights(",0.3,0.3,0.4")
        
        error_msg = str(exc_info.value)
        assert "Expected exactly 3 weights" in error_msg or "could not convert" in error_msg
    
    # ========================================================================
    # Requirement Validation Tests
    # ========================================================================
    
    def test_requirement_9_2_returns_list_of_floats(self):
        """Test that function returns list of floats (Requirement 9.2)."""
        result = parse_ensemble_weights("0.3,0.3,0.4")
        
        assert isinstance(result, list), "Should return a list"
        assert len(result) == 3, "Should return exactly 3 elements"
        assert all(isinstance(w, float) for w in result), "All elements should be floats"
    
    def test_requirement_9_3_validates_non_negative(self):
        """Test that function validates non-negative values (Requirement 9.3)."""
        # Valid: all non-negative
        result = parse_ensemble_weights("0.3,0.0,0.7")
        assert all(w >= 0 for w in result), "Should accept non-negative values"
        
        # Invalid: negative value
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,-0.1,0.8")
        assert "non-negative" in str(exc_info.value)
    
    def test_requirement_9_4_validates_exactly_three_weights(self):
        """Test that function requires exactly 3 weights (Requirement 9.4)."""
        # Valid: exactly 3
        result = parse_ensemble_weights("0.3,0.3,0.4")
        assert len(result) == 3, "Should accept exactly 3 weights"
        
        # Invalid: 2 weights
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.5,0.5")
        assert "Expected exactly 3 weights" in str(exc_info.value)
        
        # Invalid: 4 weights
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.25,0.25,0.25,0.25")
        assert "Expected exactly 3 weights" in str(exc_info.value)
    
    def test_requirement_9_5_provides_descriptive_errors(self):
        """Test that function provides descriptive error messages (Requirement 9.5)."""
        # Test error for wrong count
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.5,0.5")
        error_msg = str(exc_info.value)
        assert "3 weights" in error_msg, "Should mention expected count"
        assert "DLinear, PatchTST, Llama2" in error_msg, "Should mention model names"
        
        # Test error for non-numeric
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,abc,0.4")
        error_msg = str(exc_info.value)
        assert "Invalid weight value" in error_msg, "Should identify invalid value"
        assert "abc" in error_msg, "Should show the problematic value"
        assert "could not convert" in error_msg, "Should explain the issue"
        
        # Test error for negative
        with pytest.raises(ValueError) as exc_info:
            parse_ensemble_weights("0.3,-0.1,0.8")
        error_msg = str(exc_info.value)
        assert "non-negative" in error_msg, "Should mention non-negativity requirement"
        assert "-0.1" in error_msg, "Should show the negative value"


# ========================================================================
# Run Tests
# ========================================================================

if __name__ == '__main__':
    # Run with pytest if available, otherwise run basic tests
    try:
        import sys
        pytest.main([__file__, '-v', '--tb=short'])
    except ImportError:
        print("pytest not found, running basic tests...")
        
        test_class = TestParseEnsembleWeights()
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                failed += 1
        
        print(f"\n{passed} passed, {failed} failed out of {len(test_methods)} tests")
