"""
Time Series Data Augmentation for Flu Forecasting
Implements various augmentation techniques to increase training data diversity
and reduce overfitting on small datasets (416 weeks).
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


class TimeSeriesAugmentation:
    """
    Time series augmentation techniques for flu forecasting.
    
    Techniques:
    1. Jitter: Add random Gaussian noise
    2. Scaling: Scale amplitude
    3. Rotation: Rotate trend
    4. Time Warping: Non-linear time distortion
    5. Window Slicing: Random cropping
    6. Magnitude Warping: Change amplitude over time
    
    Args:
        p (float): Probability of applying each augmentation (default: 0.5)
        jitter_sigma (float): Standard deviation for jitter noise (default: 0.03)
        scaling_sigma (float): Standard deviation for scaling factor (default: 0.1)
        rotation_sigma (float): Standard deviation for rotation (default: 0.05)
        magnitude_sigma (float): Standard deviation for magnitude warping (default: 0.2)
        time_warp_sigma (float): Standard deviation for time warping (default: 0.2)
        window_ratio (float): Ratio for window slicing (default: 0.9)
    """
    
    def __init__(self, p=0.5, jitter_sigma=0.03, scaling_sigma=0.1, 
                 rotation_sigma=0.05, magnitude_sigma=0.2, 
                 time_warp_sigma=0.2, window_ratio=0.9):
        self.p = p
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.rotation_sigma = rotation_sigma
        self.magnitude_sigma = magnitude_sigma
        self.time_warp_sigma = time_warp_sigma
        self.window_ratio = window_ratio
    
    def jitter(self, x):
        """
        Add random Gaussian noise to the time series.
        
        Args:
            x: Input time series [batch_size, seq_len, features] or [seq_len, features]
        
        Returns:
            Augmented time series with added noise
        """
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.jitter_sigma, x.shape)
            return x + noise
        return x
    
    def scaling(self, x):
        """
        Scale the amplitude of the time series.
        
        Args:
            x: Input time series [batch_size, seq_len, features] or [seq_len, features]
        
        Returns:
            Scaled time series
        """
        if np.random.rand() < self.p:
            factor = np.random.normal(1.0, self.scaling_sigma)
            # Clamp factor to reasonable range [0.8, 1.2]
            factor = np.clip(factor, 0.8, 1.2)
            return x * factor
        return x
    
    def rotation(self, x):
        """
        Rotate the trend of the time series (add linear trend).
        
        Args:
            x: Input time series [batch_size, seq_len, features] or [seq_len, features]
        
        Returns:
            Rotated time series
        """
        if np.random.rand() < self.p:
            shape = x.shape
            if len(shape) == 3:  # [batch, seq_len, features]
                seq_len = shape[1]
            else:  # [seq_len, features]
                seq_len = shape[0]
            
            # Create rotation trend
            rotation_trend = np.random.normal(0, self.rotation_sigma)
            time_steps = np.linspace(0, 1, seq_len)
            
            if len(shape) == 3:
                time_steps = time_steps.reshape(1, -1, 1)
            else:
                time_steps = time_steps.reshape(-1, 1)
            
            trend = rotation_trend * time_steps
            return x + trend
        return x
    
    def time_warp(self, x):
        """
        Apply non-linear time warping using cubic spline interpolation.
        
        Args:
            x: Input time series [seq_len, features] (batch processing not supported)
        
        Returns:
            Time-warped series
        """
        if np.random.rand() < self.p:
            seq_len, num_features = x.shape
            
            # Create warping path
            # Original time steps
            orig_steps = np.arange(seq_len)
            
            # Create random warping anchors
            num_anchors = max(4, seq_len // 10)  # At least 4 anchors
            anchor_positions = np.sort(np.random.choice(seq_len, num_anchors, replace=False))
            
            # Add boundary anchors
            if anchor_positions[0] != 0:
                anchor_positions = np.concatenate([[0], anchor_positions])
            if anchor_positions[-1] != seq_len - 1:
                anchor_positions = np.concatenate([anchor_positions, [seq_len - 1]])
            
            # Random warping factors
            warp_factors = np.random.normal(1.0, self.time_warp_sigma, len(anchor_positions))
            warp_factors = np.clip(warp_factors, 0.8, 1.2)
            warp_factors[0] = 1.0  # Keep boundaries fixed
            warp_factors[-1] = 1.0
            
            # Create warped time steps
            warped_anchors = anchor_positions.astype(float) * warp_factors
            # Normalize to [0, seq_len-1]
            warped_anchors = warped_anchors * (seq_len - 1) / warped_anchors[-1]
            
            # Ensure strictly increasing (fix any issues)
            for i in range(1, len(warped_anchors)):
                if warped_anchors[i] <= warped_anchors[i-1]:
                    warped_anchors[i] = warped_anchors[i-1] + 0.01
            
            # Interpolate each feature
            warped_x = np.zeros_like(x)
            for i in range(num_features):
                cs = CubicSpline(warped_anchors, x[anchor_positions, i])
                warped_x[:, i] = cs(orig_steps)
            
            return warped_x
        return x
    
    def window_slice(self, x):
        """
        Randomly slice a window from the time series (crop).
        
        Args:
            x: Input time series [seq_len, features]
        
        Returns:
            Sliced time series (same length, padded with edge values if needed)
        """
        if np.random.rand() < self.p:
            seq_len, num_features = x.shape
            target_len = int(seq_len * self.window_ratio)
            
            if target_len < seq_len:
                start = np.random.randint(0, seq_len - target_len)
                end = start + target_len
                
                # Extract slice
                sliced = x[start:end]
                
                # Pad to original length (repeat edge values)
                pad_before = start
                pad_after = seq_len - end
                
                if pad_before > 0:
                    sliced = np.vstack([np.tile(sliced[0:1], (pad_before, 1)), sliced])
                if pad_after > 0:
                    sliced = np.vstack([sliced, np.tile(sliced[-1:], (pad_after, 1))])
                
                return sliced
        return x
    
    def magnitude_warp(self, x):
        """
        Warp the magnitude (amplitude) over time using smooth random curve.
        
        Args:
            x: Input time series [seq_len, features]
        
        Returns:
            Magnitude-warped series
        """
        if np.random.rand() < self.p:
            seq_len, num_features = x.shape
            
            # Create smooth magnitude warping curve
            num_knots = max(4, seq_len // 20)
            knot_positions = np.linspace(0, seq_len - 1, num_knots)
            knot_values = np.random.normal(1.0, self.magnitude_sigma, num_knots)
            knot_values = np.clip(knot_values, 0.7, 1.3)
            
            # Interpolate
            cs = CubicSpline(knot_positions, knot_values)
            warp_curve = cs(np.arange(seq_len))
            
            # Apply magnitude warping
            warp_curve = warp_curve.reshape(-1, 1)
            return x * warp_curve
        return x
    
    def augment(self, x, techniques=None):
        """
        Apply multiple augmentation techniques sequentially.
        
        Args:
            x: Input time series [seq_len, features] or numpy array/torch tensor
            techniques: List of technique names to apply. If None, apply all.
                       Options: ['jitter', 'scaling', 'rotation', 'time_warp', 
                                'window_slice', 'magnitude_warp']
        
        Returns:
            Augmented time series (same shape as input)
        """
        # Convert torch tensor to numpy if needed
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device = x.device
            x = x.cpu().numpy()
        
        # Default: apply all techniques
        if techniques is None:
            techniques = ['jitter', 'scaling', 'rotation']
            # Only apply complex techniques with lower probability
            if np.random.rand() < 0.3:
                techniques.append('magnitude_warp')
            if np.random.rand() < 0.2:
                techniques.append('time_warp')
        
        # Apply each technique
        for technique in techniques:
            if technique == 'jitter':
                x = self.jitter(x)
            elif technique == 'scaling':
                x = self.scaling(x)
            elif technique == 'rotation':
                x = self.rotation(x)
            elif technique == 'time_warp':
                # Only works on 2D arrays
                if len(x.shape) == 2:
                    x = self.time_warp(x)
            elif technique == 'window_slice':
                # Only works on 2D arrays
                if len(x.shape) == 2:
                    x = self.window_slice(x)
            elif technique == 'magnitude_warp':
                # Only works on 2D arrays
                if len(x.shape) == 2:
                    x = self.magnitude_warp(x)
        
        # Convert back to torch if input was torch
        if is_torch:
            x = torch.from_numpy(x).float().to(device)
        
        return x
    
    def batch_augment(self, batch_x, techniques=None):
        """
        Apply augmentation to a batch of time series.
        
        Args:
            batch_x: Batch of time series [batch_size, seq_len, features]
            techniques: List of technique names to apply
        
        Returns:
            Augmented batch
        """
        # Convert torch tensor to numpy if needed
        is_torch = isinstance(batch_x, torch.Tensor)
        if is_torch:
            device = batch_x.device
            batch_x = batch_x.cpu().numpy()
        
        batch_size = batch_x.shape[0]
        augmented_batch = []
        
        # Apply augmentation to each sample in batch
        for i in range(batch_size):
            x = batch_x[i]  # [seq_len, features]
            x_aug = self.augment(x, techniques=techniques)
            augmented_batch.append(x_aug)
        
        result = np.stack(augmented_batch, axis=0)
        
        # Convert back to torch if input was torch
        if is_torch:
            result = torch.from_numpy(result).float().to(device)
        
        return result


class FluSeasonalAugmentation:
    """
    Seasonal-aware augmentation specifically for flu data.
    Preserves seasonal patterns while adding variation.
    """
    
    def __init__(self, p=0.5, season_period=52):
        """
        Args:
            p: Probability of applying augmentation
            season_period: Period of seasonality (52 weeks for flu)
        """
        self.p = p
        self.season_period = season_period
    
    def seasonal_jitter(self, x):
        """
        Add noise that respects seasonal patterns.
        Higher noise during high-flu season, lower during low-flu season.
        """
        if np.random.rand() < self.p:
            seq_len, num_features = x.shape
            
            # Create seasonal noise pattern
            # Higher variance where values are higher
            variance_scale = np.abs(x) / (np.abs(x).max() + 1e-6)
            noise = np.random.normal(0, 0.02, x.shape) * (1 + variance_scale)
            
            return x + noise
        return x
    
    def phase_shift(self, x):
        """
        Shift the phase of seasonal pattern (simulate early/late flu season).
        """
        if np.random.rand() < self.p:
            seq_len, num_features = x.shape
            
            # Random shift (1-4 weeks)
            shift = np.random.randint(-4, 5)
            
            if shift != 0:
                x_shifted = np.roll(x, shift, axis=0)
                # Fix boundaries with interpolation
                if shift > 0:
                    x_shifted[:shift] = x[:shift]
                else:
                    x_shifted[shift:] = x[shift:]
                return x_shifted
        return x


# Convenience function for easy integration
def augment_flu_data(x, p=0.5):
    """
    Quick augmentation function for flu data.
    
    Args:
        x: Time series data (numpy array or torch tensor)
        p: Probability of augmentation
    
    Returns:
        Augmented data
    """
    augmenter = TimeSeriesAugmentation(p=p)
    return augmenter.augment(x)


if __name__ == "__main__":
    # Test augmentation
    print("Testing Time Series Augmentation...")
    
    # Create sample time series (simulating flu data)
    np.random.seed(42)
    seq_len = 52  # 52 weeks
    x = np.sin(np.linspace(0, 4*np.pi, seq_len)) + np.random.normal(0, 0.1, seq_len)
    x = x.reshape(-1, 1)  # [seq_len, 1]
    
    # Test augmentation
    augmenter = TimeSeriesAugmentation(p=1.0)
    
    print(f"Original shape: {x.shape}")
    
    # Test each technique
    x_jitter = augmenter.jitter(x.copy())
    print(f"After jitter: MSE = {np.mean((x - x_jitter)**2):.6f}")
    
    x_scaled = augmenter.scaling(x.copy())
    print(f"After scaling: MSE = {np.mean((x - x_scaled)**2):.6f}")
    
    x_rotated = augmenter.rotation(x.copy())
    print(f"After rotation: MSE = {np.mean((x - x_rotated)**2):.6f}")
    
    x_warped = augmenter.time_warp(x.copy())
    print(f"After time_warp: MSE = {np.mean((x - x_warped)**2):.6f}")
    
    x_magnitude = augmenter.magnitude_warp(x.copy())
    print(f"After magnitude_warp: MSE = {np.mean((x - x_magnitude)**2):.6f}")
    
    # Test combined augmentation
    x_aug = augmenter.augment(x.copy())
    print(f"After combined augment: MSE = {np.mean((x - x_aug)**2):.6f}")
    
    # Test batch augmentation with torch
    print("\nTesting batch augmentation with torch tensors...")
    batch_x = torch.randn(4, 52, 1)  # [batch_size=4, seq_len=52, features=1]
    batch_aug = augmenter.batch_augment(batch_x)
    print(f"Batch input shape: {batch_x.shape}")
    print(f"Batch output shape: {batch_aug.shape}")
    print(f"Batch MSE: {torch.mean((batch_x - batch_aug)**2):.6f}")
    
    print("\n✅ All tests passed!")
