# Implementation Plan: Ensemble Predictor for Flu Forecasting

## Overview

This plan implements a hybrid ensemble predictor that combines predictions from three pre-trained models (DLinear, PatchTST, Llama2) using weighted averaging to achieve 8-15% performance improvement. The implementation follows a modular approach with sequential model loading to manage GPU memory efficiently, integrates with the existing test pipeline, and includes comprehensive property-based testing for mathematical correctness.

## Tasks

- [x] 1. Set up ensemble predictor module structure and core utilities
  - Create `ensemble_predictor.py` module in project root
  - Define `EnsembleConfig` dataclass for configuration management
  - Implement logging utilities with progress indicators
  - Set up constants for default checkpoint paths and weights
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 12.5_

- [ ] 2. Implement checkpoint loading and model instantiation
  - [-] 2.1 Implement `discover_checkpoints` function
    - Accept checkpoints directory, model types list, and iteration number
    - Build checkpoint paths following existing naming convention
    - Return dictionary mapping model names to checkpoint paths
    - Handle missing checkpoint directories gracefully
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [-] 2.2 Implement `load_checkpoint` function
    - Accept model type, checkpoint path, device, and args
    - Instantiate model based on model type (DLinear, PatchTST, Llama2)
    - Load state dict from checkpoint file
    - Set model to evaluation mode and disable gradients
    - Log warnings for missing or corrupted checkpoints
    - Return loaded model instance or None on failure
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.4, 8.4_
  
  - [-] 2.3 Write property test for checkpoint loading error handling
    - **Property: Graceful degradation for missing checkpoints**
    - **Validates: Requirements 1.4, 8.4**
    - Test that missing checkpoints return None without crashing
    - Test that warning messages are logged correctly

- [ ] 3. Implement weight validation and normalization
  - [-] 3.1 Implement `validate_weights` function
    - Check that all weights are non-negative floats
    - Check that weights sum to 1.0 within tolerance (0.001)
    - Raise ValueError for negative weights or zero sum
    - Return validation result
    - _Requirements: 3.4, 9.3_
  
  - [-] 3.2 Implement `normalize_weights` function
    - Accept list of weights and return normalized weights
    - Compute sum and divide each weight by sum
    - Handle edge case of zero sum by raising ValueError
    - Log warning when normalization is applied
    - _Requirements: 3.5_
  
  - [-] 3.3 Write property tests for weight operations
    - **Property 3: Weight Sum Validation**
    - **Validates: Requirement 3.4**
    - Test validation function with various weight sums
    - **Property 4: Weight Normalization Idempotence**
    - **Validates: Requirement 3.5**
    - Test normalization produces sum of 1.0 and is idempotent
  
  - [-] 3.4 Implement `adjust_weights_for_missing_models` function
    - Accept original weights dict and list of available model names
    - Filter weights to only available models
    - Normalize remaining weights to sum to 1.0
    - Log which models are missing and adjusted weights
    - _Requirements: 8.2, 8.5_
  
  - [ ] 3.5 Write property test for weight adjustment
    - **Property 6: Weight Adjustment for Missing Models**
    - **Validates: Requirement 8.2**
    - Test adjustment for various subsets of available models

- [ ] 4. Implement weighted averaging computation
  - [~] 4.1 Implement `weighted_average` function
    - Accept list of prediction tensors and list of weights
    - Validate that all tensors have identical shapes
    - Compute weighted sum: sum(w_i * pred_i)
    - Return ensemble prediction tensor
    - Raise ValueError for shape mismatches
    - _Requirements: 3.2_
  
  - [~] 4.2 Write property test for weighted average correctness
    - **Property 2: Weighted Average Correctness**
    - **Validates: Requirement 3.2**
    - Generate random predictions and weights
    - Verify ensemble equals manual weighted sum

- [ ] 5. Implement EnsemblePredictor class core functionality
  - [~] 5.1 Implement `__init__` method
    - Accept args and device parameters
    - Initialize models dictionary (empty initially)
    - Parse and validate ensemble weights from args
    - Store device and configuration
    - Initialize logging
    - _Requirements: 3.1, 3.3, 9.1, 9.2_
  
  - [~] 5.2 Implement `load_checkpoints` method
    - Discover checkpoint paths for three models
    - Loop through models and load each sequentially
    - Store loaded models in self.models dictionary
    - Adjust weights for any missing models
    - Log summary of loaded models and final weights
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 8.1, 8.3_
  
  - [~] 5.3 Implement `forward` method with sequential inference
    - Accept input tensor batch and iteration number
    - Initialize predictions dictionary
    - Loop through available models sequentially
    - For each model: load to GPU, run inference, store prediction
    - Clear GPU cache after each model inference
    - Compute ensemble prediction using weighted_average
    - Return ensemble prediction tensor
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.2, 11.1, 11.2, 11.5_
  
  - [~] 5.4 Write property test for prediction shape consistency
    - **Property 1: Prediction Shape Consistency**
    - **Validates: Requirements 2.1, 2.2, 2.3, 4.1**
    - Generate random input batches with various shapes
    - Verify output shape matches [batch_size, pred_len, features]
  
  - [~] 5.5 Write unit tests for EnsemblePredictor class
    - Test initialization with default and custom weights
    - Test evaluation mode for loaded models
    - Test fallback behavior for one or two missing models

- [~] 6. Checkpoint - Ensure core ensemble logic tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement iteration ensemble functionality (optional feature)
  - [~] 7.1 Implement `load_iteration_checkpoints` method
    - Accept model type and number of iterations
    - Load checkpoints for itr0, itr1, itr2
    - Return list of loaded model instances
    - Skip missing iterations with warning
    - Log how many iterations loaded per model
    - _Requirements: 10.1, 10.4, 10.5_
  
  - [~] 7.2 Implement `average_iterations` function
    - Accept list of prediction tensors from multiple iterations
    - Compute element-wise mean across iterations
    - Return averaged prediction tensor
    - _Requirements: 10.2_
  
  - [~] 7.3 Write property tests for iteration ensemble
    - **Property 8: Iteration Averaging Correctness**
    - **Validates: Requirement 10.2**
    - Test averaging with multiple iteration predictions
    - **Property 9: Two-Stage Ensemble Composition**
    - **Validates: Requirement 10.3**
    - Verify iteration averaging happens before model weighting
  
  - [~] 7.4 Integrate iteration ensemble into `forward` method
    - Add conditional logic for args.ensemble_iterations flag
    - Load all iterations for each model when enabled
    - Average iteration predictions before model weighting
    - _Requirements: 10.1, 10.3_

- [ ] 8. Implement command-line argument parsing
  - [x] 8.1 Extend main.py argument parser
    - Add `--model ensemble` option to model choices
    - Add `--ensemble_weights` argument (comma-separated floats)
    - Add `--checkpoint_paths` argument (comma-separated paths)
    - Add `--ensemble_iterations` flag (boolean)
    - Add `--visualize` flag (boolean)
    - _Requirements: 4.5, 5.3, 5.4, 7.1, 9.1, 9.2_
  
  - [x] 8.2 Implement `parse_ensemble_weights` function
    - Parse comma-separated weight string into list of floats
    - Validate that exactly 3 weights are provided
    - Validate that all values are non-negative floats
    - Raise descriptive errors for invalid inputs
    - _Requirements: 9.2, 9.3, 9.4, 9.5_
  
  - [~] 8.3 Write property test for weight input validation
    - **Property 7: Weight Input Validation**
    - **Validates: Requirement 9.3**
    - Test valid and invalid weight string formats

- [ ] 9. Integrate EnsemblePredictor with existing test pipeline
  - [~] 9.1 Create `create_ensemble_model` factory function
    - Accept args and device parameters
    - Instantiate EnsemblePredictor
    - Call load_checkpoints method
    - Return ensemble predictor instance
    - _Requirements: 4.5_
  
  - [~] 9.2 Add ensemble model case to main.py model instantiation
    - Add `elif args.model == 'ensemble':` branch
    - Call create_ensemble_model factory function
    - Ensure ensemble predictor follows same interface as other models
    - _Requirements: 4.5_
  
  - [~] 9.3 Verify ensemble predictor works with existing test function
    - Ensure forward method signature matches expected interface
    - Ensure output tensor format matches [batch_size, pred_len, features]
    - Test that MSE and MAE metrics compute correctly
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [~] 10. Checkpoint - Ensure integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement performance evaluation and comparison
  - [~] 11.1 Create `EnsembleMetrics` dataclass
    - Define fields for individual model metrics (MSE, MAE)
    - Define fields for ensemble metrics
    - Define fields for improvement percentages
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [~] 11.2 Implement `compute_improvement_percentage` function
    - Accept baseline metric and ensemble metric
    - Calculate percentage improvement using formula: ((baseline - ensemble) / baseline) * 100
    - Handle division by zero edge case
    - Return improvement percentage
    - _Requirements: 6.4_
  
  - [~] 11.3 Write property test for improvement calculation
    - **Property 5: Percentage Improvement Calculation**
    - **Validates: Requirement 6.4**
    - Test calculation with various metric values
  
  - [~] 11.4 Implement `evaluate_ensemble_performance` function
    - Accept model, test_data, test_loader, args, device
    - Run inference on test set and compute MSE, MAE for ensemble
    - Optionally compute metrics for individual models
    - Populate EnsembleMetrics instance
    - Compute improvement percentages
    - Check if target 8-15% improvement is met
    - _Requirements: 6.1, 6.2, 6.3, 6.5_
  
  - [~] 11.5 Implement `print_comparison_table` function
    - Accept EnsembleMetrics instance
    - Format and print table with columns: Model_Name, MSE, MAE, MSE_Improvement%, MAE_Improvement%
    - Highlight whether target improvement range is achieved
    - _Requirements: 6.3, 6.5_

- [ ] 12. Implement visualization support
  - [~] 12.1 Implement `visualize_predictions` function
    - Accept predictions dict (ensemble + individual models), ground truth, args
    - Create line plot with time on x-axis, ILI percentage on y-axis
    - Plot multiple lines: ground truth, ensemble, DLinear, PatchTST, Llama2
    - Add legend identifying each line
    - Set figure size to at least 1200x800 pixels
    - _Requirements: 7.1, 7.3, 7.4_
  
  - [~] 12.2 Implement `save_visualization` function
    - Accept matplotlib figure and output path
    - Save plot as PNG with at least 1200x800 resolution
    - Create output directory if doesn't exist
    - Log saved file path
    - _Requirements: 7.2, 7.5_
  
  - [~] 12.3 Integrate visualization into test pipeline
    - Add conditional check for args.visualize flag
    - Call visualize_predictions with sample test sequence
    - Save visualization to checkpoints directory
    - _Requirements: 7.1, 7.2_

- [ ] 13. Implement standalone ensemble_predictor.py script
  - [~] 13.1 Create standalone script structure
    - Set up argument parser with required arguments
    - Implement main function for standalone execution
    - _Requirements: 5.1, 5.2_
  
  - [~] 13.2 Implement standalone prediction workflow
    - Parse checkpoint paths and weights from arguments
    - Initialize EnsemblePredictor with parsed arguments
    - Load test data using data_provider
    - Run inference and compute ensemble predictions
    - Save predictions to disk in numpy format
    - _Requirements: 5.2, 5.3, 5.4, 5.5_
  
  - [~] 13.3 Write integration test for standalone script
    - Test end-to-end execution with valid arguments
    - Verify output file is created with correct format

- [ ] 14. Implement memory management and performance monitoring
  - [~] 14.1 Add GPU memory logging
    - Log GPU memory allocated/reserved after loading each model
    - Log GPU memory after clearing cache
    - Use torch.cuda.memory_allocated() and torch.cuda.memory_reserved()
    - _Requirements: 11.4_
  
  - [~] 14.2 Add execution time tracking
    - Log start time when ensemble predictor initializes
    - Log elapsed time after checkpoint loading phase
    - Log elapsed time after inference phase
    - Log elapsed time after evaluation phase
    - Display total execution time at completion
    - _Requirements: 11.3, 12.1, 12.2, 12.3, 12.4_
  
  - [~] 14.3 Implement progress indicators for long operations
    - Add progress bar for checkpoint loading
    - Add progress bar for test set inference
    - Add progress indicator for visualization generation
    - Use tqdm library (already in project)
    - _Requirements: 12.5_

- [ ] 15. Error handling and edge cases
  - [~] 15.1 Implement comprehensive error handling for checkpoint loading
    - Handle FileNotFoundError with descriptive message
    - Handle RuntimeError for incompatible checkpoints
    - Handle memory errors with fallback strategies
    - Log all errors with context
    - _Requirements: 1.4, 8.3, 8.4_
  
  - [~] 15.2 Implement error handling for weight configuration
    - Validate weight count matches number of models
    - Validate no negative weights
    - Validate non-zero sum
    - Provide descriptive error messages
    - _Requirements: 3.4, 3.5, 9.3, 9.4, 9.5_
  
  - [~] 15.3 Implement numerical stability checks
    - Check for NaN and Inf values in predictions
    - Add epsilon to prevent division by zero
    - Clamp extreme values if necessary
    - Log warnings for numerical issues
    - _Requirements: 6.4_
  
  - [~] 15.4 Write unit tests for error handling
    - Test missing checkpoint handling
    - Test invalid weight configurations
    - Test numerical edge cases

- [ ] 16. Final integration and end-to-end testing
  - [~] 16.1 Test complete workflow with all three models
    - Run ensemble predictor with default weights
    - Verify MSE and MAE metrics are computed
    - Verify improvement percentage is calculated
    - Check console output for comparison table
    - _Requirements: 4.2, 4.3, 6.1, 6.2, 6.3, 6.4_
  
  - [~] 16.2 Test custom weight configurations
    - Run with various weight combinations
    - Verify custom weights are applied correctly
    - Test weight normalization warning
    - _Requirements: 3.3, 9.1, 9.2_
  
  - [~] 16.3 Test fallback scenarios
    - Test with one missing checkpoint
    - Test with two missing checkpoints
    - Verify weight adjustment and warnings
    - _Requirements: 8.1, 8.2, 8.3, 8.5_
  
  - [~] 16.4 Test optional iteration ensemble
    - Enable --ensemble_iterations flag
    - Verify iteration averaging occurs before model weighting
    - Verify correct logging of iteration counts
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [~] 16.5 Test visualization generation
    - Enable --visualize flag
    - Verify PNG file is created in checkpoints directory
    - Verify plot contains all expected lines
    - Check resolution meets 1200x800 requirement
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [~] 16.6 Write integration tests for complete workflows
    - Test end-to-end with mocked models
    - Test standalone script execution
    - Verify output format compatibility

- [ ] 17. Documentation and final validation
  - [~] 17.1 Add docstrings to all functions and classes
    - Document parameters, return values, and exceptions
    - Include usage examples for key functions
    - Follow Google-style docstring format
  
  - [~] 17.2 Validate execution time budget
    - Measure total time for implementation tasks
    - Measure test execution time on full dataset
    - Measure visualization generation time
    - Verify total time is within 1.5 hours budget
    - _Requirements: 12.1, 12.2, 12.3_
  
  - [~] 17.3 Verify 8-15% improvement target
    - Run ensemble on full test dataset
    - Compare against individual model baselines
    - Document actual improvement percentages
    - Log whether target range is achieved
    - _Requirements: 6.5_

- [~] 18. Final checkpoint - Complete validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP implementation
- The implementation uses Python with PyTorch, matching the existing codebase
- Sequential model loading (tasks 5.3, 11.1) is critical for GPU memory management
- Property-based tests use `hypothesis` library with minimum 100 iterations per property
- Each task references specific requirements for traceability back to the requirements document
- Checkpoints at tasks 6, 10, and 18 provide natural stopping points for user feedback
- The design includes 9 correctness properties that are implemented as property-based tests
- Integration with existing pipeline (tasks 9.x) ensures the ensemble predictor works seamlessly with current infrastructure
- Memory management (task 14) ensures the ensemble runs within resource constraints
- Error handling (task 15) ensures graceful degradation and helpful error messages
- The standalone script (task 13) provides an alternative execution path independent of main.py

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "8.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2", "8.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.3", "3.4", "3.5", "4.1", "8.3"] },
    { "id": 3, "tasks": ["4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "11.1", "11.2"] },
    { "id": 5, "tasks": ["5.3", "5.4", "5.5", "11.3"] },
    { "id": 6, "tasks": ["7.1", "7.2", "9.1"] },
    { "id": 7, "tasks": ["7.3", "7.4", "9.2", "11.4"] },
    { "id": 8, "tasks": ["9.3", "11.5", "12.1"] },
    { "id": 9, "tasks": ["12.2", "12.3", "13.1"] },
    { "id": 10, "tasks": ["13.2", "13.3", "14.1", "14.2"] },
    { "id": 11, "tasks": ["14.3", "15.1", "15.2"] },
    { "id": 12, "tasks": ["15.3", "15.4", "16.1", "16.2"] },
    { "id": 13, "tasks": ["16.3", "16.4", "16.5", "16.6"] },
    { "id": 14, "tasks": ["17.1", "17.2", "17.3"] }
  ]
}
```
