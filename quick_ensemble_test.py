"""
Quick ensemble test - simplified version
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from ensemble_predictor import discover_checkpoints, setup_logger

def main():
    logger = setup_logger()
    
    # Test 1: Check if checkpoints can be discovered
    logger.info("=" * 70)
    logger.info("TEST 1: Checkpoint Discovery")
    logger.info("=" * 70)
    
    checkpoints_dir = './checkpoints/'
    model_types = ['DLinear', 'PatchTST', 'Llama2']
    iteration = 0
    
    checkpoint_paths = discover_checkpoints(checkpoints_dir, model_types, iteration)
    
    logger.info(f"\n✓ Found {len(checkpoint_paths)}/{len(model_types)} checkpoints:")
    for model_name, path in checkpoint_paths.items():
        file_size = os.path.getsize(path) / (1024**2)  # MB
        logger.info(f"  {model_name}: {path}")
        logger.info(f"    Size: {file_size:.1f} MB")
    
    if len(checkpoint_paths) == 0:
        logger.error("❌ No checkpoints found! Cannot proceed.")
        return
    
    # Test 2: Check if checkpoints can be loaded
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Checkpoint Loading")
    logger.info("=" * 70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Import models
    try:
        from models.DLinear import DLinear
        from models.PatchTST import PatchTST  
        from models.Llama2 import Llama2
        logger.info("✓ Model imports successful")
    except Exception as e:
        logger.error(f"❌ Failed to import models: {e}")
        return
    
    # Create minimal args for model instantiation
    class Args:
        def __init__(self):
            # Sequence settings
            self.seq_len = 336
            self.label_len = 18
            self.pred_len = 8
            
            # Model architecture
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.d_model = 768
            self.n_heads = 16
            self.e_layers = 3
            self.d_layers = 1
            self.d_ff = 512
            self.dropout = 0.1
            self.patch_size = 16
            self.stride = 8
            
            # Llama settings
            self.llama_layers = 32
            self.llama_model = './data_disk/lichx/Model_from_HF/LLAMA2'
            self.d_llm = 4096
            
            # Other
            self.freq = 'W'
            self.embed = 'timeF'
    
    args = Args()
    
    # Try loading each checkpoint
    loaded_models = {}
    for model_name, checkpoint_path in checkpoint_paths.items():
        try:
            logger.info(f"\nLoading {model_name}...")
            
            # Instantiate model
            if model_name == 'DLinear':
                model = DLinear(args, device)
                model.to(device)
            elif model_name == 'PatchTST':
                model = PatchTST(args, device)
                model.to(device)
            elif model_name == 'Llama2':
                # Adjust args for Llama2
                args.pred_len = 13  # Llama2 was trained with pl=13
                model = Llama2(args, device)
                args.pred_len = 8  # Reset
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            
            # Set to eval mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            loaded_models[model_name] = model
            logger.info(f"✓ {model_name} loaded successfully")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Checkpoints discovered: {len(checkpoint_paths)}/3")
    logger.info(f"Models loaded successfully: {len(loaded_models)}/3")
    
    if len(loaded_models) >= 2:
        logger.info("\n✅ SUCCESS! Ensemble is ready to use.")
        logger.info(f"   Available models: {', '.join(loaded_models.keys())}")
        logger.info("\nNext steps:")
        logger.info("1. Use main.py with --model ensemble")
        logger.info("2. Make sure to set correct --target and other args")
        logger.info("3. Example command:")
        logger.info("   python main.py --model ensemble --model_id test \\")
        logger.info("     --data_path NorthChina_diff.csv --target positive_rate \\")
        logger.info("     --pred_len 8 --itr 1")
    else:
        logger.error("\n❌ FAILED! Need at least 2 models for ensemble.")
    
if __name__ == '__main__':
    main()
