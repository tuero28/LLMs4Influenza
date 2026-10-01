# Requirements Document

## Introduction

This document specifies requirements for implementing flu forecasting model improvements within a 5-hour time constraint. The improvements target four key areas: (1) data augmentation through time series transformations, (2) epidemic-specific evaluation metrics for better performance measurement, (3) temporal feature enhancement to capture seasonal patterns, and (4) hybrid model ensemble to combine predictions from multiple architectures. These improvements aim to achieve an 8-15% performance gain on a limited dataset (468 weeks of ILI data from three regions) without requiring external data collection or extensive retraining.

## Glossary

- **Data_Augmentation_Module**: Component that applies stochastic transformations to time series during training
- **Epidemic_Metrics_Module**: Component that computes domain-specific metrics for epidemic forecasting evaluation
- **Temporal_Features_Module**: Component that generates time-based features including Fourier and cyclic encodings
- **Training_Pipeline**: The existing data loading and model training process
- **Dataset_Loader**: Component in data_provider/data_loader.py that loads and prepares ILI data
- **Test_Function**: The test() function in main.py or exp_main.py that evaluates model performance
- **ILI_Dataset**: Influenza-Like Illness dataset containing 468 weeks of data from North China, South China, and USA
- **Training_Flag**: Boolean indicator distinguishing training mode from validation/test modes
- **Peak_Week**: The week index where ILI cases reach maximum value in a sequence
- **Peak_Value**: The maximum ILI case count in a sequence
- **Outbreak_Threshold**: ILI case count above which an outbreak is considered active (default 30000)
- **Fourier_Harmonic**: Sine/cosine pair representing a frequency component of seasonal patterns
- **Week_of_Year**: Integer from 1 to 52 representing the week position in annual cycle
- **Season_Indicator**: Binary flag indicating membership in a specific season
- **Ensemble_Predictor**: Component that combines predictions from multiple trained models
- **Model_Checkpoint**: Saved model weights from a completed training run
- **Hybrid_Ensemble**: Ensemble strategy combining architecturally diverse models (DLinear, PatchTST, Llama2)
- **Iteration_Ensemble**: Ensemble strategy combining multiple training iterations of the same model architecture
- **Ensemble_Weight**: Coefficient determining contribution of each model to final ensemble prediction
- **Checkpoint_Path**: File system location of a Model_Checkpoint

## Requirements

### Requirement 1: Time Series Data Augmentation

**User Story:** As a model trainer, I want stochastic augmentation applied to training data, so that models generalize better with limited data.

#### Acceptance Criteria

1. WHEN Training_Flag equals 'train', THE Data_Augmentation_Module SHALL apply jitter with Gaussian noise where sigma equals 0.03
2. WHEN Training_Flag equals 'train', THE Data_Augmentation_Module SHALL apply scaling with amplitude factor sampled from Gaussian distribution where sigma equals 0.1
3. WHEN Training_Flag equals 'train', THE Data_Augmentation_Module SHALL apply window slice with reduce_ratio equal to 0.9
4. WHEN Training_Flag equals 'val' OR Training_Flag equals 'test', THE Data_Augmentation_Module SHALL return data without augmentation
5. THE Data_Augmentation_Module SHALL integrate into Dataset_Loader within data_provider/data_loader.py
6. THE Data_Augmentation_Module SHALL preserve the temporal dimension of input sequences
7. THE Data_Augmentation_Module SHALL apply augmentations independently to each sequence in a batch

### Requirement 2: Epidemic-Specific Metrics Implementation

**User Story:** As a researcher, I want epidemic-specific metrics computed during evaluation, so that I can measure performance on domain-relevant criteria beyond MSE/MAE.

#### Acceptance Criteria

1. THE Epidemic_Metrics_Module SHALL compute Peak_Timing_Error as the absolute difference between predicted Peak_Week and ground truth Peak_Week
2. THE Epidemic_Metrics_Module SHALL compute Peak_Intensity_Error as the absolute difference between predicted Peak_Value and ground truth Peak_Value
3. THE Epidemic_Metrics_Module SHALL compute Direction_Accuracy as the proportion of time steps where sign of prediction change matches sign of ground truth change
4. THE Epidemic_Metrics_Module SHALL compute Outbreak_F1 as the F1 score for outbreak detection using Outbreak_Threshold equal to 30000
5. THE Epidemic_Metrics_Module SHALL implement in utils/epidemic_metrics.py
6. THE Test_Function SHALL invoke Epidemic_Metrics_Module and log results alongside existing MSE and MAE metrics
7. THE Epidemic_Metrics_Module SHALL handle sequences where no peak exceeds Outbreak_Threshold by returning null values for outbreak metrics

### Requirement 3: Temporal Features Enhancement

**User Story:** As a data engineer, I want advanced temporal features extracted from timestamps, so that models can better capture seasonal flu patterns.

#### Acceptance Criteria

1. THE Temporal_Features_Module SHALL generate Fourier_Harmonic pairs for k equal to 1, 2, 3, and 4
2. WHEN generating Fourier_Harmonic for frequency k, THE Temporal_Features_Module SHALL compute sin(2 * pi * k * Week_of_Year / 52) and cos(2 * pi * k * Week_of_Year / 52)
3. THE Temporal_Features_Module SHALL generate cyclic encodings week_sin, week_cos, month_sin, and month_cos using 2 * pi * value / period
4. THE Temporal_Features_Module SHALL generate Season_Indicator features is_winter, is_spring, is_summer, and is_fall based on month boundaries
5. WHEN month is 11, 12, 1, or 2, THE Temporal_Features_Module SHALL set is_winter to 1
6. WHEN month is 3, 4, or 5, THE Temporal_Features_Module SHALL set is_spring to 1
7. WHEN month is 6, 7, or 8, THE Temporal_Features_Module SHALL set is_summer to 1
8. WHEN month is 9, 10, THE Temporal_Features_Module SHALL set is_fall to 1
9. THE Temporal_Features_Module SHALL integrate into Dataset_Loader and concatenate generated features with existing temporal features
10. THE Temporal_Features_Module SHALL preserve backward compatibility by making enhanced features optional via configuration flag

### Requirement 4: Integration and Compatibility

**User Story:** As a developer, I want all improvements integrated without breaking existing functionality, so that the system remains stable during incremental enhancement.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL maintain existing command-line arguments and configuration parameters
2. THE Dataset_Loader SHALL preserve existing data loading behavior when augmentation is disabled
3. THE Test_Function SHALL maintain existing output format while adding epidemic metrics to logs
4. THE system SHALL support all existing models (Llama2, PatchTST, DLinear) without modification to model architectures
5. WHEN augmentation_enabled configuration flag is False, THE Data_Augmentation_Module SHALL bypass all transformations
6. WHEN epidemic_metrics_enabled configuration flag is False, THE Test_Function SHALL compute only existing MSE and MAE metrics
7. WHEN enhanced_temporal_features configuration flag is False, THE Temporal_Features_Module SHALL generate only existing temporal features

### Requirement 5: Performance and Efficiency

**User Story:** As an operator, I want all improvements to complete training and evaluation within the 5-hour time budget, so that rapid iteration remains feasible.

#### Acceptance Criteria

1. THE Data_Augmentation_Module SHALL add no more than 10% overhead to data loading time
2. THE Epidemic_Metrics_Module SHALL compute all metrics in O(n) time complexity where n is sequence length
3. THE Temporal_Features_Module SHALL compute features during dataset initialization rather than per batch
4. THE system SHALL maintain memory footprint within existing GPU constraints
5. WHEN all improvements are enabled, THE Training_Pipeline SHALL complete one full training run within 5 hours on existing hardware

### Requirement 6: Hybrid Model Ensemble

**User Story:** As a researcher, I want predictions from multiple trained models combined through ensemble, so that I can achieve 8-15% performance improvement without additional training.

#### Acceptance Criteria

1. THE Ensemble_Predictor SHALL load Model_Checkpoint from three distinct architectures: DLinear, PatchTST, and Llama2
2. THE Ensemble_Predictor SHALL locate Model_Checkpoint files using Checkpoint_Path pattern 'checkpoints/{model_name}_full_sl336_ll18_pl{pred_len}_dm{d_model}_nh16_el3_gl6_df512_ebtimeF_itr0'
3. THE Ensemble_Predictor SHALL verify Model_Checkpoint existence and load successfully before proceeding with ensemble prediction
4. WHEN generating predictions, THE Ensemble_Predictor SHALL invoke forward pass on all three loaded models with identical input data
5. THE Ensemble_Predictor SHALL combine individual model predictions using weighted average where Ensemble_Weight values sum to 1.0
6. THE Ensemble_Predictor SHALL initialize Ensemble_Weight values to [0.3, 0.3, 0.4] for DLinear, PatchTST, and Llama2 respectively
7. THE Ensemble_Predictor SHALL support configurable Ensemble_Weight values via command-line arguments or configuration file
8. THE Ensemble_Predictor SHALL implement as a standalone prediction script that loads checkpoints without modifying existing training code
9. THE Ensemble_Predictor SHALL output predictions in the same format as single models for compatibility with existing evaluation pipeline
10. THE Ensemble_Predictor SHALL complete inference on test set within 30 minutes on existing hardware
11. WHEN any Model_Checkpoint fails to load, THE Ensemble_Predictor SHALL log error and fall back to averaging only successfully loaded models
12. THE Ensemble_Predictor SHALL support optional Iteration_Ensemble mode where multiple training iterations of same architecture are averaged

