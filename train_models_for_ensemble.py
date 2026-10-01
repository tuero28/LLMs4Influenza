"""
Helper script to train models for ensemble predictor

This script trains DLinear, PatchTST, and optionally Llama2 models,
ensuring checkpoints are saved correctly for ensemble usage.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def check_checkpoint_exists(model_name, iteration=0):
    """Check if checkpoint file exists for a model"""
    if model_name == 'DLinear':
        checkpoint_dir = f"checkpoints/DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    elif model_name == 'PatchTST':
        checkpoint_dir = f"checkpoints/PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    elif model_name == 'Llama2':
        checkpoint_dir = f"checkpoints/Llama2_full_sl336_ll18_pl13_dm4096_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    else:
        return False
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    return os.path.exists(checkpoint_path)


def train_model(model_name, data_path='NorthChina_diff.csv', pred_len=8, epochs=10, itr=1):
    """Train a single model"""
    
    print("\n" + "="*70)
    print(f"TRAINING {model_name}")
    print("="*70)
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        'main.py',
        '--model', model_name,
        '--model_id', f'{model_name}_full',
        '--data_path', data_path,
        '--pred_len', str(pred_len),
        '--train_epochs', str(epochs),
        '--itr', str(itr),
        '--batch_size', '32',
        '--patience', '3'
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        elapsed = time.time() - start_time
        
        print(f"\n✓ {model_name} training completed in {elapsed:.1f}s")
        
        # Verify checkpoint was saved
        if check_checkpoint_exists(model_name, iteration=0):
            print(f"✓ Checkpoint verified for {model_name}")
            return True
        else:
            print(f"✗ WARNING: Checkpoint not found for {model_name}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error training {model_name}: {e}")
        return False


def verify_all_checkpoints():
    """Verify all required checkpoints exist"""
    print("\n" + "="*70)
    print("VERIFYING CHECKPOINTS")
    print("="*70)
    
    models = ['DLinear', 'PatchTST', 'Llama2']
    results = {}
    
    for model in models:
        exists = check_checkpoint_exists(model, iteration=0)
        results[model] = exists
        
        status = "✓" if exists else "✗"
        print(f"{status} {model}: {'Found' if exists else 'Missing'}")
    
    return results


def main():
    """Main training pipeline"""
    print("="*70)
    print("ENSEMBLE MODEL TRAINING SCRIPT")
    print("="*70)
    print("\nThis script will train models required for ensemble predictor:")
    print("  1. DLinear")
    print("  2. PatchTST")
    print("  3. Llama2 (optional - requires strong GPU)")
    print()
    
    # Check current directory
    if not os.path.exists('main.py'):
        print("ERROR: main.py not found!")
        print("Please run this script from the LLMs4Influenza directory")
        return
    
    # Configuration
    DATA_PATH = 'NorthChina_diff.csv'
    PRED_LEN = 8
    EPOCHS = 10
    ITERATIONS = 1
    
    print(f"Configuration:")
    print(f"  - Dataset: {DATA_PATH}")
    print(f"  - Prediction Length: {PRED_LEN}")
    print(f"  - Training Epochs: {EPOCHS}")
    print(f"  - Iterations: {ITERATIONS}")
    print()
    
    # Ask for confirmation
    response = input("Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Train models
    results = {}
    
    # 1. Train DLinear
    print("\n" + "="*70)
    print("STEP 1/3: Training DLinear")
    print("="*70)
    results['DLinear'] = train_model('DLinear', DATA_PATH, PRED_LEN, EPOCHS, ITERATIONS)
    
    # 2. Train PatchTST
    print("\n" + "="*70)
    print("STEP 2/3: Training PatchTST")
    print("="*70)
    results['PatchTST'] = train_model('PatchTST', DATA_PATH, PRED_LEN, EPOCHS, ITERATIONS)
    
    # 3. Ask about Llama2 (requires more resources)
    print("\n" + "="*70)
    print("STEP 3/3: Llama2 Training (Optional)")
    print("="*70)
    print("Llama2 requires:")
    print("  - Strong GPU (8GB+ VRAM)")
    print("  - More training time (~3-5x longer)")
    print("  - pred_len=13 (different from DLinear/PatchTST)")
    print()
    
    response = input("Train Llama2? (y/n): ").strip().lower()
    if response == 'y':
        results['Llama2'] = train_model('Llama2', DATA_PATH, pred_len=13, epochs=EPOCHS, itr=ITERATIONS)
    else:
        print("Skipping Llama2 training.")
        print("Note: Ensemble can work with just DLinear + PatchTST")
        results['Llama2'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model:<12} {status}")
    
    # Verify checkpoints
    checkpoint_status = verify_all_checkpoints()
    
    # Final status
    available_models = sum(1 for v in checkpoint_status.values() if v)
    
    print("\n" + "="*70)
    print("ENSEMBLE READINESS")
    print("="*70)
    
    if available_models >= 2:
        print(f"✓ READY: {available_models}/3 models available")
        print("\nYou can now run ensemble:")
        print("  python main.py --model ensemble --model_id ensemble_test --data_path NorthChina_diff.csv --pred_len 8 --itr 1")
    else:
        print(f"✗ NOT READY: Only {available_models}/3 models available")
        print("  Ensemble requires at least 2 models")
        print("  Please retry training for failed models")
    
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
