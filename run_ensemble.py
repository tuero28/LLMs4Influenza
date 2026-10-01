"""
Quick script to run ensemble prediction
"""
import sys
sys.path.append('.')

# Override args để bypass training
class Args:
    def __init__(self):
        # Model settings
        self.model = 'ensemble'
        self.model_id = 'ensemble_test'
        
        # Data settings
        self.data = 'custom'
        self.root_path = './dataset/'
        self.data_path = 'NorthChina_diff.csv'
        self.features = 'S'  # Single feature
        self.target = 'positive_rate'
        self.freq = 'W'  # Weekly
        self.embed = 'timeF'
        self.percent = 100  # Use 100% of data
        self.max_len = -1  # No max length limit
        
        # Sequence settings
        self.seq_len = 52  # ~1 năm for weekly data
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
        
        # Ensemble settings
        self.ensemble_weights = '0.3,0.3,0.4'  # DLinear, PatchTST, Llama2
        self.checkpoint_paths = ''  # Auto-discover
        self.ensemble_iterations = False
        
        # Training settings (not used for inference)
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.train_epochs = 10
        self.patience = 3
        self.lradj = 'type1'
        
        # Other
        self.checkpoints = './checkpoints/'
        self.itr = 1
        self.visualize = 1
        self.use_gpu = True
        self.gpu = 0
        self.devices = '0'
        
        # Llama specific
        self.llama_layers = 32
        self.llama_model = './data_disk/lichx/Model_from_HF/LLAMA2'

def main():
    import torch
    from ensemble_predictor import EnsemblePredictor, setup_logger
    from data_provider.data_factory import data_provider
    import numpy as np
    
    # Setup
    args = Args()
    logger = setup_logger()
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    logger.info("Loading test data...")
    try:
        test_data, test_loader = data_provider(args, 'test')
        logger.info(f"Test data loaded: {len(test_data)} samples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return
    
    # Create ensemble predictor
    logger.info("Creating ensemble predictor...")
    try:
        predictor = EnsemblePredictor(args, device)
        logger.info("Ensemble predictor created")
    except Exception as e:
        logger.error(f"Failed to create predictor: {e}")
        return
    
    # Load checkpoints
    logger.info("Loading checkpoints...")
    try:
        predictor.load_checkpoints()
        logger.info(f"Loaded {len(predictor.models)} models")
    except Exception as e:
        logger.error(f"Failed to load checkpoints: {e}")
        return
    
    # Run inference
    logger.info("Running inference...")
    try:
        preds = []
        trues = []
        
        predictor.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # Forward pass
                outputs = predictor(batch_x, iteration=0)
                
                # Extract predictions
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(test_loader)} batches")
        
        # Concatenate results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        logger.info(f"Predictions shape: {preds.shape}")
        logger.info(f"Ground truth shape: {trues.shape}")
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(trues.reshape(-1), preds.reshape(-1))
        mae = mean_absolute_error(trues.reshape(-1), preds.reshape(-1))
        
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        logger.info("✓ Ensemble inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
