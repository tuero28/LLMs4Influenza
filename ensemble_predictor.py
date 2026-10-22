"""
Ensemble Predictor for Flu Forecasting
Combines predictions from multiple models (Llama2, Llama3, Gemma2, PatchTST, DLinear)
to improve robustness and accuracy.

Ensemble Strategies:
1. Simple Average
2. Weighted Average (learned weights)
3. Median Ensemble
4. Stacking (meta-model on predictions)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import os


class EnsemblePredictor(nn.Module):
    """
    Ensemble predictor that combines multiple trained models.
    
    Args:
        models: List of model instances [model1, model2, ...]
        model_names: List of model names ['Llama2', 'PatchTST', ...]
        strategy: Ensemble strategy ('average', 'weighted', 'median', 'stacking')
        learnable_weights: If True, learn weights during training (for 'weighted')
    """
    
    def __init__(self, models, model_names=None, strategy='average', learnable_weights=True):
        super(EnsemblePredictor, self).__init__()
        
        self.models = nn.ModuleList(models) if not isinstance(models, nn.ModuleList) else models
        self.num_models = len(models)
        self.strategy = strategy
        self.learnable_weights = learnable_weights
        
        if model_names is None:
            self.model_names = [f"model_{i}" for i in range(self.num_models)]
        else:
            self.model_names = model_names
        
        # Initialize weights for weighted ensemble
        if strategy == 'weighted' and learnable_weights:
            # Initialize with equal weights
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        elif strategy == 'weighted' and not learnable_weights:
            # Fixed equal weights
            self.register_buffer('weights', torch.ones(self.num_models) / self.num_models)
        
        # Stacking meta-model (simple linear layer)
        if strategy == 'stacking':
            self.meta_model = nn.Sequential(
                nn.Linear(self.num_models, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, itr=0, mask=None):
        """
        Forward pass through ensemble.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, features]
            x_mark_enc: Encoder time marks
            x_dec: Decoder input
            x_mark_dec: Decoder time marks
            itr: Iteration number (for some models)
            mask: Optional mask
        
        Returns:
            Ensemble prediction [batch_size, pred_len, features]
        """
        predictions = []
        
        # Collect predictions from all models
        for i, model in enumerate(self.models):
            try:
                with torch.no_grad() if not self.training else torch.enable_grad():
                    # Different models have different forward signatures
                    if hasattr(model, 'forward') and 'itr' in model.forward.__code__.co_varnames:
                        # LLM-based models (Llama, Gemma)
                        pred = model(x_enc, itr)
                    else:
                        # Transformer-based models (PatchTST, DLinear)
                        if mask is not None:
                            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                        else:
                            try:
                                pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                            except:
                                # Some models might not need all inputs
                                pred = model(x_enc)
                    
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {self.model_names[i]} failed: {e}")
                # Use zero prediction as fallback
                if len(predictions) > 0:
                    predictions.append(torch.zeros_like(predictions[0]))
                else:
                    raise e
        
        # Stack predictions [num_models, batch_size, pred_len, features]
        predictions = torch.stack(predictions, dim=0)
        
        # Apply ensemble strategy
        if self.strategy == 'average':
            output = torch.mean(predictions, dim=0)
        
        elif self.strategy == 'weighted':
            # Normalize weights with softmax
            weights = F.softmax(self.weights, dim=0)
            # Weighted sum
            weights = weights.view(-1, 1, 1, 1)  # [num_models, 1, 1, 1]
            output = torch.sum(predictions * weights, dim=0)
        
        elif self.strategy == 'median':
            output = torch.median(predictions, dim=0)[0]
        
        elif self.strategy == 'stacking':
            # Reshape for meta-model
            batch_size, pred_len, features = predictions.shape[1:]
            # [num_models, B, T, F] -> [B, T, F, num_models]
            predictions_transposed = predictions.permute(1, 2, 3, 0)
            # Apply meta-model
            output = self.meta_model(predictions_transposed).squeeze(-1)
            # [B, T, F, 1] -> [B, T, F]
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return output
    
    def get_model_weights(self):
        """Get current ensemble weights."""
        if self.strategy == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return {name: w.item() for name, w in zip(self.model_names, weights)}
        return None
    
    def set_model_weights(self, weights_dict):
        """
        Set ensemble weights manually.
        
        Args:
            weights_dict: Dict mapping model names to weights
                         e.g., {'Llama2': 0.3, 'PatchTST': 0.4, 'DLinear': 0.3}
        """
        if self.strategy != 'weighted':
            raise ValueError("Can only set weights for 'weighted' strategy")
        
        weights = []
        for name in self.model_names:
            if name in weights_dict:
                weights.append(weights_dict[name])
            else:
                weights.append(0.0)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize
        
        if self.learnable_weights:
            self.weights.data = weights.to(self.weights.device)
        else:
            self.weights = weights.to(self.weights.device)


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns to weight models based on input characteristics.
    Different models may be better at different time periods or patterns.
    """
    
    def __init__(self, models, model_names=None, input_dim=768):
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models) if not isinstance(models, nn.ModuleList) else models
        self.num_models = len(models)
        
        if model_names is None:
            self.model_names = [f"model_{i}" for i in range(self.num_models)]
        else:
            self.model_names = model_names
        
        # Weight prediction network
        # Takes input features and predicts weights for each model
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, itr=0, mask=None):
        """
        Forward pass with adaptive weighting.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, features]
            x_mark_enc: Encoder time marks
            x_dec: Decoder input
            x_mark_dec: Decoder time marks
            itr: Iteration number
            mask: Optional mask
        
        Returns:
            Ensemble prediction [batch_size, pred_len, features]
        """
        predictions = []
        
        # Collect predictions from all models
        for model in self.models:
            try:
                with torch.no_grad() if not self.training else torch.enable_grad():
                    if hasattr(model, 'forward') and 'itr' in model.forward.__code__.co_varnames:
                        pred = model(x_enc, itr)
                    else:
                        if mask is not None:
                            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                        else:
                            try:
                                pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                            except:
                                pred = model(x_enc)
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model failed: {e}")
                if len(predictions) > 0:
                    predictions.append(torch.zeros_like(predictions[0]))
                else:
                    raise e
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, T, F]
        
        # Predict weights based on input
        # Use mean of input sequence as feature
        input_features = x_enc.mean(dim=1)  # [B, F]
        weights = self.weight_network(input_features)  # [B, num_models]
        
        # Apply adaptive weights
        weights = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_models, 1]
        predictions = predictions.permute(1, 2, 0, 3)  # [B, T, num_models, F]
        
        output = torch.sum(predictions * weights, dim=2)  # [B, T, F]
        
        return output


def load_ensemble_models(configs, checkpoint_paths):
    """
    Load multiple models from checkpoints for ensemble.
    
    Args:
        configs: List of config objects for each model
        checkpoint_paths: List of checkpoint file paths
    
    Returns:
        models: List of loaded models
        model_names: List of model names
    """
    models = []
    model_names = []
    
    for config, ckpt_path in zip(configs, checkpoint_paths):
        # Import model class dynamically
        model_name = config.model
        model_names.append(model_name)
        
        # Load model
        if model_name == 'Llama2':
            from models.Llama2 import Model
        elif model_name == 'Llama3':
            from models.Llama3 import Model
        elif model_name == 'Gemma2':
            from models.Gemma2 import Model
        elif model_name == 'PatchTST':
            from models.PatchTST import Model
        elif model_name == 'DLinear':
            from models.DLinear import Model
        elif model_name == 'GPT4TS':
            from models.GPT4TS import Model
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Initialize model
        model = Model(config)
        
        # Load checkpoint if provided
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded {model_name} from {ckpt_path}")
        else:
            print(f"⚠ No checkpoint for {model_name}, using random initialization")
        
        model.eval()
        models.append(model)
    
    return models, model_names


def create_ensemble_from_checkpoints(checkpoint_dir, model_configs, strategy='weighted'):
    """
    Create ensemble from trained model checkpoints.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        model_configs: List of config objects
        strategy: Ensemble strategy
    
    Returns:
        ensemble: EnsemblePredictor instance
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find checkpoint files
    checkpoint_paths = []
    for config in model_configs:
        # Look for checkpoint pattern matching the config
        pattern = f"{config.model}_*_itr{config.itr}/checkpoint.pth"
        matches = list(checkpoint_dir.glob(pattern))
        
        if matches:
            checkpoint_paths.append(str(matches[0]))
        else:
            checkpoint_paths.append(None)
            print(f"⚠ No checkpoint found for {config.model}")
    
    # Load models
    models, model_names = load_ensemble_models(model_configs, checkpoint_paths)
    
    # Create ensemble
    ensemble = EnsemblePredictor(models, model_names, strategy=strategy)
    
    return ensemble


class EnsembleConfig:
    """Configuration for ensemble predictor."""
    
    def __init__(self):
        # Model selection
        self.models = ['Llama2', 'PatchTST', 'DLinear']  # Models to ensemble
        self.strategy = 'weighted'  # 'average', 'weighted', 'median', 'stacking'
        self.learnable_weights = True
        
        # Initial weights (if not learnable)
        self.initial_weights = {
            'Llama2': 0.4,
            'Llama3': 0.3,
            'Gemma2': 0.1,
            'PatchTST': 0.1,
            'DLinear': 0.1,
            'GPT4TS': 0.0
        }
        
        # Checkpoint directory
        self.checkpoint_dir = './checkpoints'
    
    def save(self, filepath):
        """Save configuration to JSON file."""
        config_dict = {
            'models': self.models,
            'strategy': self.strategy,
            'learnable_weights': self.learnable_weights,
            'initial_weights': self.initial_weights,
            'checkpoint_dir': self.checkpoint_dir
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load(filepath):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = EnsembleConfig()
        config.models = config_dict['models']
        config.strategy = config_dict['strategy']
        config.learnable_weights = config_dict.get('learnable_weights', True)
        config.initial_weights = config_dict.get('initial_weights', {})
        config.checkpoint_dir = config_dict.get('checkpoint_dir', './checkpoints')
        
        return config


def test_ensemble():
    """Test ensemble predictor with dummy models."""
    
    print("Testing Ensemble Predictor...")
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale
        
        def forward(self, x, itr=0):
            return x * self.scale
    
    models = [
        DummyModel(scale=1.0),
        DummyModel(scale=1.2),
        DummyModel(scale=0.8)
    ]
    
    model_names = ['Model1', 'Model2', 'Model3']
    
    # Test different strategies
    strategies = ['average', 'weighted', 'median']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} ensemble...")
        ensemble = EnsemblePredictor(models, model_names, strategy=strategy)
        
        # Create dummy input
        batch_size, seq_len, features = 2, 52, 1
        x = torch.randn(batch_size, seq_len, features)
        
        # Forward pass
        output = ensemble(x, None, None, None, itr=0)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        
        if strategy == 'weighted':
            weights = ensemble.get_model_weights()
            print(f"  Model weights: {weights}")
    
    print("\n✅ All ensemble tests passed!")


if __name__ == "__main__":
    test_ensemble()


# Convenience function for main.py
def create_ensemble_model(args, device):
    """
    Create ensemble model from args (convenience wrapper).
    
    This is a simple wrapper for compatibility with main.py.
    For more advanced usage, use create_ensemble_from_checkpoints directly.
    """
    # Parse model names from args
    if hasattr(args, 'ensemble_models'):
        model_names = args.ensemble_models.split(',')
    else:
        model_names = ['DLinear', 'PatchTST']  # Default
    
    # Import model classes
    from models.DLinear import DLinear
    from models.PatchTST import PatchTST
    from models.Llama2 import Llama2
    
    model_classes = {
        'DLinear': DLinear,
        'PatchTST': PatchTST,
        'Llama2': Llama2,
    }
    
    # Create model instances
    models = []
    for name in model_names:
        if name in model_classes:
            model = model_classes[name](args).to(device)
            models.append(model)
        else:
            print(f"Warning: Model {name} not found, skipping")
    
    if not models:
        raise ValueError(f"No valid models found in: {model_names}")
    
    # Create ensemble
    strategy = getattr(args, 'ensemble_strategy', 'weighted')
    ensemble = EnsemblePredictor(
        models=models,
        model_names=model_names,
        strategy=strategy,
        learnable_weights=True
    ).to(device)
    
    print(f"✓ Created ensemble with {len(models)} models: {model_names}")
    return ensemble
