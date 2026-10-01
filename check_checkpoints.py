"""
Quick script to check checkpoint status for ensemble predictor
"""

import os
from pathlib import Path


def check_checkpoint(model_name, iteration=0):
    """Check if checkpoint exists and get file info"""
    
    # Build checkpoint path based on model naming convention
    if model_name == 'DLinear':
        checkpoint_dir = f"checkpoints/DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    elif model_name == 'PatchTST':
        checkpoint_dir = f"checkpoints/PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    elif model_name == 'Llama2':
        checkpoint_dir = f"checkpoints/Llama2_full_sl336_ll18_pl13_dm4096_nh16_el3_gl6_df512_ebtimeF_itr{iteration}"
    else:
        return None
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    # Check existence
    if os.path.exists(checkpoint_path):
        file_size = os.path.getsize(checkpoint_path)
        file_size_mb = file_size / (1024 * 1024)
        return {
            'exists': True,
            'path': checkpoint_path,
            'size': file_size,
            'size_mb': file_size_mb
        }
    else:
        # Check if directory exists
        dir_exists = os.path.exists(checkpoint_dir)
        return {
            'exists': False,
            'path': checkpoint_path,
            'dir_exists': dir_exists
        }


def main():
    print("="*70)
    print("CHECKPOINT STATUS FOR ENSEMBLE PREDICTOR")
    print("="*70)
    print()
    
    models = ['DLinear', 'PatchTST', 'Llama2']
    available_count = 0
    
    for model in models:
        print(f"\n{model}:")
        print("-" * 40)
        
        # Check all iterations
        for itr in range(3):
            info = check_checkpoint(model, itr)
            
            if info and info['exists']:
                available_count += 1
                print(f"  itr{itr}: ✓ Found ({info['size_mb']:.1f} MB)")
                print(f"         {info['path']}")
            else:
                if info and info.get('dir_exists'):
                    print(f"  itr{itr}: ✗ Directory exists but checkpoint.pth missing")
                else:
                    print(f"  itr{itr}: ✗ Not found")
                if info:
                    print(f"         Expected: {info['path']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Check itr0 specifically (required for ensemble)
    itr0_status = {}
    for model in models:
        info = check_checkpoint(model, 0)
        itr0_status[model] = info and info['exists']
    
    available_itr0 = sum(1 for v in itr0_status.values() if v)
    
    print(f"\nModels available (itr0): {available_itr0}/3")
    for model, available in itr0_status.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {model:<12} {status}")
    
    # Ensemble readiness
    print("\n" + "="*70)
    print("ENSEMBLE READINESS")
    print("="*70)
    
    if available_itr0 >= 2:
        print(f"✓ READY: {available_itr0}/3 models available for ensemble")
        print("\nYou can run:")
        print("  python main.py --model ensemble --model_id ensemble_test \\")
        print("      --data_path NorthChina_diff.csv --pred_len 8 --itr 1")
        
        # List which models will be used
        available_models = [m for m, v in itr0_status.items() if v]
        print(f"\nEnsemble will use: {', '.join(available_models)}")
        
        if available_itr0 < 3:
            missing = [m for m, v in itr0_status.items() if not v]
            print(f"Missing: {', '.join(missing)} (weights will be auto-adjusted)")
    
    elif available_itr0 == 1:
        print(f"⚠ PARTIAL: Only {available_itr0}/3 model available")
        print("  Ensemble requires at least 2 models")
        available = [m for m, v in itr0_status.items() if v][0]
        missing = [m for m, v in itr0_status.items() if not v]
        print(f"\n  Available: {available}")
        print(f"  Need to train: {', '.join(missing)}")
        print("\nTo train models:")
        print("  python train_models_for_ensemble.py")
    
    else:
        print("✗ NOT READY: No models available")
        print("\nTo train models:")
        print("  python train_models_for_ensemble.py")
        print("\nOr train individually:")
        for model in models:
            print(f"  python main.py --model {model} --model_id {model}_full --train_epochs 10")
    
    print()


if __name__ == '__main__':
    main()
