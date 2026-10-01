"""
Quick test to train DLinear for 1 epoch and verify checkpoint saving
"""

import subprocess
import sys
import os
import time

print("="*70)
print("QUICK TRAINING TEST - DLinear (1 epoch)")
print("="*70)
print()

# Configuration for quick test
cmd = [
    sys.executable,
    'main.py',
    '--model', 'DLinear',
    '--model_id', 'test_DLinear',
    '--data_path', 'NorthChina_diff.csv',
    '--pred_len', '8',
    '--train_epochs', '1',  # Just 1 epoch for quick test
    '--itr', '1',
    '--batch_size', '32',
    '--patience', '3'
]

print("Running command:")
print(" ".join(cmd))
print()
print("-"*70)

start = time.time()

try:
    # Run training
    result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    
    elapsed = time.time() - start
    print()
    print("-"*70)
    print(f"✓ Training completed in {elapsed:.1f}s")
    
    # Check if checkpoint was created
    checkpoint_path = "checkpoints/test_DLinear_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth"
    
    print()
    print("Checking checkpoint...")
    print(f"Expected path: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path)
        size_mb = size / (1024 * 1024)
        print(f"✓ Checkpoint FOUND! ({size_mb:.2f} MB)")
        print()
        print("="*70)
        print("SUCCESS! Checkpoint saving is working correctly.")
        print("="*70)
        print()
        print("You can now train the full models:")
        print("  python train_models_for_ensemble.py")
    else:
        print("✗ Checkpoint NOT FOUND!")
        print()
        print("Checking directory...")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            print(f"✓ Directory exists: {checkpoint_dir}")
            files = os.listdir(checkpoint_dir)
            if files:
                print(f"  Files in directory: {files}")
            else:
                print("  Directory is empty")
        else:
            print(f"✗ Directory doesn't exist: {checkpoint_dir}")
        
        print()
        print("="*70)
        print("ISSUE: Checkpoint was not saved")
        print("="*70)
        print()
        print("Possible causes:")
        print("1. Early stopping callback not being called")
        print("2. Validation loss not decreasing")
        print("3. Permission issues writing to checkpoint directory")
        print()
        print("Please check the training log above for errors.")
    
except subprocess.CalledProcessError as e:
    print()
    print("="*70)
    print(f"✗ Training FAILED with exit code {e.returncode}")
    print("="*70)
    print()
    print("Please check error messages above.")

except KeyboardInterrupt:
    print()
    print("Training interrupted by user.")

except Exception as e:
    print()
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
