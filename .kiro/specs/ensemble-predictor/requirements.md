# Requirements Document

## Introduction

This document specifies requirements for a Hybrid Model Ensemble predictor for flu forecasting. The system combines predictions from three trained model checkpoints (DLinear, PatchTST, Llama2) using weighted averaging to achieve 8-15% performance improvement without retraining models. The solution must be implemented within 1.5 hours total time constraint (30 min code + 30 min test + 30 min visualize).

## Glossary

- **Ensemble_Predictor**: The hybrid system that loads multiple model checkpoints and combines their predictions using weighted averaging
- **Checkpoint**: A saved model state containing trained parameters stored in the checkpoints directory
- **Model_Type**: One of three supported model architectures (DLinear, PatchTST, Llama2)
- **Weight_Vector**: A set of numerical weights summing to 1.0 used for weighted averaging of model predictions
- **Prediction_Tensor**: A PyTorch tensor containing flu forecasting predictions for the next 8-13 weeks
- **Test_Pipeline**: The existing evaluation infrastructure that computes metrics (MSE, MAE) on test datasets
- **ILI_Dataset**: Influenza-like illness dataset containing 468 weeks of data for North China, South China, and USA regions
- **Iteration_Checkpoint**: A checkpoint from a specific training iteration (itr0, itr1, or itr2)

## Requirements

### Requirement 1: Load Model Checkpoints

**User Story:** As a researcher, I want to load trained model checkpoints from the filesystem, so that I can use their predictions without retraining.

#### Acceptance Criteria

1. WHEN a checkpoint path is provided for DLinear itr0, THE Ensemble_Predictor SHALL load the DLinear model state from `checkpoints/DLinear_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth`
2. WHEN a checkpoint path is provided for PatchTST itr0, THE Ensemble_Predictor SHALL load the PatchTST model state from `checkpoints/PatchTST_full_sl336_ll18_pl8_dm768_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth`
3. WHEN a checkpoint path is provided for Llama2 itr0, THE Ensemble_Predictor SHALL load the Llama2 model state from `checkpoints/Llama2_full_sl336_ll18_pl13_dm4096_nh16_el3_gl6_df512_ebtimeF_itr0/checkpoint.pth`
4. IF a checkpoint file does not exist, THEN THE Ensemble_Predictor SHALL log a warning message and continue with remaining available models
5. WHEN all three model checkpoints are loaded, THE Ensemble_Predictor SHALL initialize each model with its corresponding architecture configuration

### Requirement 2: Generate Individual Model Predictions

**User Story:** As a researcher, I want each loaded model to generate predictions independently, so that I can combine them into an ensemble prediction.

#### Acceptance Criteria

1. WHEN the ILI_Dataset test data is provided to a loaded DLinear model, THE Ensemble_Predictor SHALL generate a Prediction_Tensor of shape [batch_size, pred_len, features]
2. WHEN the ILI_Dataset test data is provided to a loaded PatchTST model, THE Ensemble_Predictor SHALL generate a Prediction_Tensor of shape [batch_size, pred_len, features]
3. WHEN the ILI_Dataset test data is provided to a loaded Llama2 model, THE Ensemble_Predictor SHALL generate a Prediction_Tensor of shape [batch_size, pred_len, features]
4. THE Ensemble_Predictor SHALL execute model inference in evaluation mode with gradients disabled
5. WHEN generating predictions, THE Ensemble_Predictor SHALL use the same data preprocessing pipeline as the existing Test_Pipeline

### Requirement 3: Weighted Average Combination

**User Story:** As a researcher, I want to combine predictions using weighted averaging, so that I can leverage the strengths of different models.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL combine predictions using a default Weight_Vector of [0.3, 0.3, 0.4] for [DLinear, PatchTST, Llama2]
2. WHEN computing the ensemble prediction, THE Ensemble_Predictor SHALL calculate weighted_prediction = w1 * pred1 + w2 * pred2 + w3 * pred3
3. WHERE custom weights are provided via command-line arguments, THE Ensemble_Predictor SHALL use the custom Weight_Vector instead of defaults
4. THE Ensemble_Predictor SHALL validate that all weights in the Weight_Vector sum to 1.0 within tolerance 0.001
5. IF the Weight_Vector sum is invalid, THEN THE Ensemble_Predictor SHALL normalize the weights to sum to 1.0 and log a warning

### Requirement 4: Integration with Existing Pipeline

**User Story:** As a researcher, I want the ensemble predictor to integrate seamlessly with the existing test pipeline, so that I can evaluate it using established metrics.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL output predictions in the same tensor format as single models [batch_size, pred_len, features]
2. WHEN invoked through the Test_Pipeline, THE Ensemble_Predictor SHALL compute MSE metric on ensemble predictions versus ground truth
3. WHEN invoked through the Test_Pipeline, THE Ensemble_Predictor SHALL compute MAE metric on ensemble predictions versus ground truth
4. THE Ensemble_Predictor SHALL accept the same command-line arguments as existing models (--seq_len, --pred_len, --batch_size, --data_path)
5. THE Ensemble_Predictor SHALL be invokable via a new --model ensemble argument in the existing main.py entry point

### Requirement 5: Standalone Prediction Script

**User Story:** As a researcher, I want a standalone script to run ensemble predictions, so that I can use it independently without modifying existing code.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL provide an ensemble_predictor.py script that can be executed directly
2. WHEN ensemble_predictor.py is executed with required arguments, THE Ensemble_Predictor SHALL load all three checkpoints, generate predictions, and output results
3. THE Ensemble_Predictor SHALL support a --checkpoint_paths argument accepting three comma-separated paths
4. THE Ensemble_Predictor SHALL support a --weights argument accepting three comma-separated float values
5. WHEN ensemble_predictor.py completes execution, THE Ensemble_Predictor SHALL save ensemble predictions to disk in numpy format

### Requirement 6: Performance Evaluation and Comparison

**User Story:** As a researcher, I want to compare ensemble performance against individual models, so that I can verify the 8-15% improvement target.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL compute MSE for the ensemble prediction on the test set
2. THE Ensemble_Predictor SHALL compute MAE for the ensemble prediction on the test set
3. WHEN evaluation completes, THE Ensemble_Predictor SHALL display a comparison table showing [Model_Name, MSE, MAE, MSE_Improvement%, MAE_Improvement%]
4. THE Ensemble_Predictor SHALL calculate percentage improvement as ((single_model_metric - ensemble_metric) / single_model_metric) * 100
5. THE Ensemble_Predictor SHALL log whether the ensemble achieved the target 8-15% improvement range

### Requirement 7: Visualization Support

**User Story:** As a researcher, I want to visualize ensemble predictions versus ground truth, so that I can qualitatively assess forecast quality.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL generate a line plot comparing ensemble predictions, ground truth, and individual model predictions for a sample test sequence
2. WHEN visualization is enabled via --visualize flag, THE Ensemble_Predictor SHALL save plots to the checkpoints directory
3. THE Ensemble_Predictor SHALL plot time (weeks) on the x-axis and ILI percentage on the y-axis
4. THE Ensemble_Predictor SHALL include a legend identifying each line (Ground Truth, Ensemble, DLinear, PatchTST, Llama2)
5. THE Ensemble_Predictor SHALL save visualization plots in PNG format with resolution at least 1200x800 pixels

### Requirement 8: Fallback Handling for Missing Checkpoints

**User Story:** As a researcher, I want the system to gracefully handle missing checkpoints, so that partial ensemble predictions can still be generated.

#### Acceptance Criteria

1. IF only two of three checkpoints are available, THEN THE Ensemble_Predictor SHALL compute ensemble using only the available models
2. WHEN fewer than three models are available, THE Ensemble_Predictor SHALL automatically adjust weights to sum to 1.0 across available models
3. IF only one checkpoint is available, THEN THE Ensemble_Predictor SHALL log an error message and use the single model prediction
4. IF zero checkpoints are available, THEN THE Ensemble_Predictor SHALL terminate with a clear error message indicating missing checkpoint paths
5. WHEN operating in fallback mode, THE Ensemble_Predictor SHALL log which models are missing and which weights are being used

### Requirement 9: Configurable Weights via Arguments

**User Story:** As a researcher, I want to experiment with different weight combinations, so that I can find the optimal ensemble configuration.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL accept a --ensemble_weights argument with three comma-separated float values
2. WHEN --ensemble_weights "0.25,0.35,0.40" is provided, THE Ensemble_Predictor SHALL use weights [0.25, 0.35, 0.40] for [DLinear, PatchTST, Llama2]
3. THE Ensemble_Predictor SHALL validate that provided weights are non-negative float values
4. THE Ensemble_Predictor SHALL validate that the number of provided weights matches the number of models (3)
5. IF weight validation fails, THEN THE Ensemble_Predictor SHALL terminate with a descriptive error message

### Requirement 10: Optional Iteration Ensemble

**User Story:** As a researcher, I want to optionally ensemble multiple iterations of the same model, so that I can further reduce variance if time permits.

#### Acceptance Criteria

1. WHERE --ensemble_iterations flag is enabled, THE Ensemble_Predictor SHALL load all three Iteration_Checkpoints (itr0, itr1, itr2) for each Model_Type
2. WHEN computing iteration ensemble for a single model type, THE Ensemble_Predictor SHALL average predictions from itr0, itr1, and itr2 with equal weights
3. THE Ensemble_Predictor SHALL compute the iteration-averaged prediction before applying the inter-model weighted average
4. IF any Iteration_Checkpoint is missing, THEN THE Ensemble_Predictor SHALL use only available iterations for that model
5. WHEN iteration ensemble is enabled, THE Ensemble_Predictor SHALL log how many iterations were averaged for each Model_Type

### Requirement 11: Memory and Performance Constraints

**User Story:** As a researcher, I want the ensemble predictor to fit within existing resource constraints, so that it can run on the same hardware as individual models.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL load at most one model into GPU memory at a time during inference
2. WHEN a model completes inference, THE Ensemble_Predictor SHALL release GPU memory before loading the next model
3. THE Ensemble_Predictor SHALL complete full test set inference within 30 minutes on existing hardware
4. THE Ensemble_Predictor SHALL log GPU memory usage after loading each model
5. THE Ensemble_Predictor SHALL use torch.cuda.empty_cache() after each model's inference completes

### Requirement 12: Execution Time Budget

**User Story:** As a researcher, I want to complete the entire ensemble implementation and evaluation within 1.5 hours, so that I can deliver results within the project timeline.

#### Acceptance Criteria

1. THE Ensemble_Predictor implementation SHALL be completable within 30 minutes of coding time
2. THE Ensemble_Predictor testing on the full ILI_Dataset SHALL complete within 30 minutes
3. THE Ensemble_Predictor visualization generation SHALL complete within 30 minutes
4. WHEN execution starts, THE Ensemble_Predictor SHALL log start time and display elapsed time after each major phase (loading, inference, evaluation)
5. THE Ensemble_Predictor SHALL provide progress indicators during long-running operations (loading checkpoints, generating predictions)
