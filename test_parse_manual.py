"""Quick manual test of parse_ensemble_weights function"""
from ensemble_predictor import parse_ensemble_weights

print("=" * 60)
print("Manual Tests for parse_ensemble_weights")
print("=" * 60)

# Test 1: Valid default weights
print("\n1. Test valid default weights '0.3,0.3,0.4':")
result = parse_ensemble_weights('0.3,0.3,0.4')
print(f"   Result: {result}")
print(f"   Type: {type(result)}, Length: {len(result)}")

# Test 2: Valid with spaces
print("\n2. Test valid with spaces ' 0.25 , 0.35 , 0.40 ':")
result = parse_ensemble_weights(' 0.25 , 0.35 , 0.40 ')
print(f"   Result: {result}")

# Test 3: Error - wrong count (2 weights)
print("\n3. Test error with 2 weights '0.5,0.5':")
try:
    result = parse_ensemble_weights('0.5,0.5')
    print(f"   ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"   ✓ Caught ValueError: {e}")

# Test 4: Error - wrong count (4 weights)
print("\n4. Test error with 4 weights '0.25,0.25,0.25,0.25':")
try:
    result = parse_ensemble_weights('0.25,0.25,0.25,0.25')
    print(f"   ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"   ✓ Caught ValueError: {e}")

# Test 5: Error - negative weight
print("\n5. Test error with negative weight '0.3,-0.1,0.8':")
try:
    result = parse_ensemble_weights('0.3,-0.1,0.8')
    print(f"   ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"   ✓ Caught ValueError: {e}")

# Test 6: Error - non-numeric value
print("\n6. Test error with non-numeric 'abc,0.3,0.4':")
try:
    result = parse_ensemble_weights('abc,0.3,0.4')
    print(f"   ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"   ✓ Caught ValueError: {e}")

# Test 7: Zero values (valid)
print("\n7. Test zero values '0,0.5,0.5':")
result = parse_ensemble_weights('0,0.5,0.5')
print(f"   Result: {result}")

# Test 8: Integer values
print("\n8. Test integer values '1,0,0':")
result = parse_ensemble_weights('1,0,0')
print(f"   Result: {result}")
print(f"   All floats: {all(isinstance(w, float) for w in result)}")

print("\n" + "=" * 60)
print("All manual tests completed successfully!")
print("=" * 60)
