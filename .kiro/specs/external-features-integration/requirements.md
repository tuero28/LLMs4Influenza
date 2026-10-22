# Requirements Document

## Introduction

This document specifies requirements for integrating external features (exogenous data) into the LLMs4Influenza forecasting system. Currently, the system relies solely on historical flu positive_rate data for predictions. Research indicates that incorporating external features such as weather conditions, search trends, holidays, demographics, and vaccination data can improve forecasting accuracy by 15-25%.

The external features integration will extend the existing univariate time series approach to multivariate forecasting while maintaining backward compatibility with current models.

## Glossary

- **System**: The LLMs4Influenza forecasting system
- **External_Features**: Exogenous data sources not directly measuring flu incidence (weather, search trends, holidays, demographics, vaccination)
- **Feature_Collector**: Module responsible for gathering external data from various sources
- **Feature_Processor**: Module responsible for temporal and spatial alignment of external features
- **Feature_Engineer**: Module responsible for creating derived features (lags, rolling statistics, interactions)
- **Dataset_Custom**: Existing PyTorch Dataset class in data_loader.py for flu time series data
- **Historical_Flu_Data**: Weekly positive_rate time series from 2011-04-04 to 2020-03-23 (469 weeks)
- **Region**: Geographic area for flu forecasting (North China, South China, USA)
- **Temporal_Alignment**: Process of matching external feature timestamps to weekly flu data frequency
- **Spatial_Alignment**: Process of matching external features to specific regions
- **Data_Leakage**: Using future information to predict past values (testing error)
- **Ablation_Study**: Systematic removal of features to measure individual contribution

## Requirements

### Requirement 1: Weather Data Collection

**User Story:** As a data scientist, I want to collect weather data for each region, so that temperature and humidity patterns can inform flu forecasting.

#### Acceptance Criteria

1. WHEN a region name is provided, THE Feature_Collector SHALL retrieve temperature data for that region from 2011-04-04 to 2020-03-23
2. WHEN a region name is provided, THE Feature_Collector SHALL retrieve humidity data for that region from 2011-04-04 to 2020-03-23
3. WHEN a region name is provided, THE Feature_Collector SHALL retrieve precipitation data for that region from 2011-04-04 to 2020-03-23
4. WHERE weather data has missing values, THE Feature_Collector SHALL apply forward-fill interpolation
5. WHEN weather data is retrieved, THE Feature_Collector SHALL validate that temperature values are within -50°C to 50°C range
6. WHEN weather data is retrieved, THE Feature_Collector SHALL validate that humidity values are within 0% to 100% range
7. THE Feature_Collector SHALL support data sources including OpenWeatherMap API and NOAA Climate Data Online

### Requirement 2: Google Trends Data Collection

**User Story:** As a data scientist, I want to collect Google Trends search volume data, so that public interest in flu-related terms can inform outbreak predictions.

#### Acceptance Criteria

1. WHEN a region name is provided, THE Feature_Collector SHALL retrieve search volume for keyword "flu" from 2011-04-04 to 2020-03-23
2. WHEN a region name is provided, THE Feature_Collector SHALL retrieve search volume for keyword "fever" from 2011-04-04 to 2020-03-23
3. WHEN a region name is provided, THE Feature_Collector SHALL retrieve search volume for keyword "vaccine" from 2011-04-04 to 2020-03-23
4. WHERE region is North China OR South China, THE Feature_Collector SHALL retrieve search volume for keyword "cúm" (Vietnamese for flu)
5. WHEN Google Trends data is retrieved, THE Feature_Collector SHALL normalize search volume to 0-100 scale
6. WHERE Google Trends API rate limits are exceeded, THE Feature_Collector SHALL retry with exponential backoff up to 3 attempts
7. WHEN search volume data has gaps, THE Feature_Collector SHALL apply linear interpolation for gaps up to 4 weeks

### Requirement 3: Holiday Calendar Data Collection

**User Story:** As a data scientist, I want to collect holiday and vacation period data, so that seasonal travel patterns can inform flu transmission predictions.

#### Acceptance Criteria

1. WHEN a region name is provided, THE Feature_Collector SHALL retrieve major holidays for that region from 2011-04-04 to 2020-03-23
2. WHERE region is North China OR South China, THE Feature_Collector SHALL include Chinese New Year (Spring Festival) dates
3. WHERE region is USA, THE Feature_Collector SHALL include Thanksgiving and Christmas dates
4. THE Feature_Collector SHALL encode holidays as binary features (1 for holiday week, 0 otherwise)
5. THE Feature_Collector SHALL include a 2-week window around major holidays as extended holiday periods
6. WHEN holiday data is processed, THE Feature_Collector SHALL create separate binary features for pre-holiday, holiday, and post-holiday weeks

### Requirement 4: Demographics Data Collection

**User Story:** As a data scientist, I want to collect demographic data for each region, so that population characteristics can inform flu susceptibility patterns.

#### Acceptance Criteria

1. WHEN a region name is provided, THE Feature_Collector SHALL retrieve population density data
2. WHEN a region name is provided, THE Feature_Collector SHALL retrieve age distribution data (percentage in age groups: 0-18, 19-64, 65+)
3. WHERE demographics data is only available annually, THE Feature_Collector SHALL repeat values across all weeks within that year
4. WHEN demographics data is retrieved, THE Feature_Collector SHALL validate that age distribution percentages sum to 100% ± 1%
5. THE Feature_Collector SHALL normalize population density using z-score normalization across all regions

### Requirement 5: Vaccination Data Collection

**User Story:** As a data scientist, I want to collect vaccination coverage data, so that immunization levels can inform flu outbreak predictions.

#### Acceptance Criteria

1. WHEN a region name is provided, THE Feature_Collector SHALL attempt to retrieve vaccination coverage data from WHO or CDC sources
2. WHERE vaccination data is available, THE Feature_Collector SHALL extract flu vaccine coverage percentage by week
3. WHERE vaccination data is only available at seasonal granularity, THE Feature_Collector SHALL distribute coverage linearly across vaccination campaign weeks (typically weeks 40-52)
4. WHERE vaccination data is unavailable for a region, THE Feature_Collector SHALL set vaccination features to zero and log a warning
5. WHEN vaccination data is retrieved, THE Feature_Collector SHALL validate that coverage percentages are within 0% to 100% range

### Requirement 6: Temporal Alignment

**User Story:** As a data scientist, I want all external features aligned to weekly frequency, so that they match the flu data temporal resolution.

#### Acceptance Criteria

1. WHEN external features are collected at daily frequency, THE Feature_Processor SHALL aggregate to weekly frequency using mean
2. WHEN external features are collected at monthly frequency, THE Feature_Processor SHALL interpolate to weekly frequency using cubic spline
3. THE Feature_Processor SHALL align all feature timestamps to Monday of each week (ISO week start)
4. WHEN timestamps are aligned, THE Feature_Processor SHALL ensure exactly 469 weeks of data from 2011-04-04 to 2020-03-23
5. WHERE a feature has missing weeks after alignment, THE Feature_Processor SHALL apply forward-fill for up to 2 consecutive weeks
6. WHERE a feature has missing weeks exceeding 2 consecutive weeks, THE Feature_Processor SHALL raise an error with details on the gap

### Requirement 7: Spatial Alignment

**User Story:** As a data scientist, I want all external features matched to specific regions, so that geographic correspondence with flu data is ensured.

#### Acceptance Criteria

1. THE Feature_Processor SHALL support three regions: "North China", "South China", and "USA"
2. WHEN weather data is collected, THE Feature_Processor SHALL map weather stations to regions using geographic bounding boxes
3. WHEN Google Trends data is collected, THE Feature_Processor SHALL use region-specific search queries (e.g., "China" for Chinese regions, "United States" for USA)
4. WHERE multiple data sources overlap a region, THE Feature_Processor SHALL average values weighted by coverage area
5. WHEN spatial alignment is complete, THE Feature_Processor SHALL validate that each region has a complete feature set for all 469 weeks

### Requirement 8: Feature Normalization

**User Story:** As a data scientist, I want all features normalized to similar scales, so that models can learn effectively without scale bias.

#### Acceptance Criteria

1. WHEN continuous features are processed, THE Feature_Processor SHALL apply z-score normalization (mean=0, std=1)
2. THE Feature_Processor SHALL compute normalization statistics (mean, std) using only training data (first 70% of 469 weeks)
3. THE Feature_Processor SHALL apply training normalization statistics to validation and test data
4. WHEN binary features (holidays) are processed, THE Feature_Processor SHALL NOT apply normalization
5. THE Feature_Processor SHALL save normalization statistics to disk for inverse transformation of predictions
6. WHEN normalization is applied, THE Feature_Processor SHALL validate that normalized features have no NaN or infinite values

### Requirement 9: Lagged Features Engineering

**User Story:** As a data scientist, I want to create lagged versions of weather features, so that models can learn delayed effects of temperature and humidity on flu transmission.

#### Acceptance Criteria

1. WHEN weather features are processed, THE Feature_Engineer SHALL create lag-1 features (1 week prior)
2. WHEN weather features are processed, THE Feature_Engineer SHALL create lag-2 features (2 weeks prior)
3. WHEN weather features are processed, THE Feature_Engineer SHALL create lag-4 features (4 weeks prior)
4. WHEN lagged features are created, THE Feature_Engineer SHALL ensure no data leakage (lag values come only from past)
5. WHERE lag windows extend before the start date, THE Feature_Engineer SHALL pad with the first available value
6. WHEN lagged features are created for validation or test sets, THE Feature_Engineer SHALL use historical training data to fill lag windows without leakage

### Requirement 10: Rolling Statistics Features Engineering

**User Story:** As a data scientist, I want to create rolling averages and standard deviations of weather features, so that models can capture smoothed trends and variability.

#### Acceptance Criteria

1. WHEN weather features are processed, THE Feature_Engineer SHALL create 4-week rolling mean features
2. WHEN weather features are processed, THE Feature_Engineer SHALL create 4-week rolling standard deviation features
3. WHEN rolling statistics are computed, THE Feature_Engineer SHALL use only past data (no future information)
4. WHERE rolling windows extend before the start date, THE Feature_Engineer SHALL compute statistics on available data (minimum 1 week)
5. WHEN rolling features are created, THE Feature_Engineer SHALL validate that values are not NaN after the first 4 weeks

### Requirement 11: Interaction Features Engineering

**User Story:** As a data scientist, I want to create interaction terms between weather features, so that models can capture combined effects (e.g., cold and humid conditions).

#### Acceptance Criteria

1. WHEN temperature and humidity features are available, THE Feature_Engineer SHALL create temperature × humidity interaction feature
2. WHEN temperature and precipitation features are available, THE Feature_Engineer SHALL create temperature × precipitation interaction feature
3. WHEN interaction features are created, THE Feature_Engineer SHALL normalize the product using z-score normalization
4. THE Feature_Engineer SHALL create interaction features after computing lagged and rolling features
5. WHEN interaction features are created, THE Feature_Engineer SHALL validate that values are not NaN or infinite

### Requirement 12: Seasonal Encoding Features Engineering

**User Story:** As a data scientist, I want to encode seasonal patterns using trigonometric functions, so that models can learn cyclic patterns without discontinuities.

#### Acceptance Criteria

1. WHEN temporal features are processed, THE Feature_Engineer SHALL create sin(2π × week_of_year / 52) feature
2. WHEN temporal features are processed, THE Feature_Engineer SHALL create cos(2π × week_of_year / 52) feature
3. THE Feature_Engineer SHALL compute week_of_year using ISO 8601 standard (weeks 1-52 or 1-53)
4. WHEN seasonal encoding is created, THE Feature_Engineer SHALL validate that sin and cos features are within [-1, 1] range
5. THE Feature_Engineer SHALL create seasonal encoding for all 469 weeks in the dataset

### Requirement 13: Dataset_Custom Extension

**User Story:** As a developer, I want to extend the Dataset_Custom class to include external features, so that the data loader can provide multivariate time series to models.

#### Acceptance Criteria

1. THE Dataset_Custom SHALL accept an optional parameter `use_external_features` (boolean, default False)
2. WHERE `use_external_features` is True, THE Dataset_Custom SHALL load external features from a configured path
3. WHEN external features are loaded, THE Dataset_Custom SHALL concatenate them with flu positive_rate data along the feature dimension
4. WHERE `use_external_features` is False, THE Dataset_Custom SHALL return only flu positive_rate data (backward compatible)
5. WHEN external features are concatenated, THE Dataset_Custom SHALL update `self.enc_in` to reflect the total number of features
6. THE Dataset_Custom SHALL validate that external features have the same temporal length as flu data before concatenation
7. WHEN __getitem__ is called, THE Dataset_Custom SHALL return sequences with shape [seq_len, num_features] where num_features includes flu data + external features

### Requirement 14: External Features Configuration

**User Story:** As a developer, I want a configuration system for external features, so that users can easily enable/disable feature types without code changes.

#### Acceptance Criteria

1. THE System SHALL provide a configuration file `external_features_config.yaml` in the project root
2. THE configuration file SHALL specify which feature types are enabled (weather, google_trends, holidays, demographics, vaccination)
3. THE configuration file SHALL specify feature engineering options (lags, rolling_windows, interactions, seasonal_encoding)
4. THE configuration file SHALL specify data source API keys and endpoints for each feature type
5. WHEN the System loads configuration, THE System SHALL validate that required API keys are present for enabled features
6. WHERE a required API key is missing, THE System SHALL raise a configuration error with a descriptive message
7. THE System SHALL log all enabled features and engineering options at initialization

### Requirement 15: Data Quality Validation

**User Story:** As a data scientist, I want automated validation of external features data quality, so that issues are detected before training.

#### Acceptance Criteria

1. WHEN external features are loaded, THE System SHALL check for NaN values and report percentage per feature
2. WHERE NaN percentage exceeds 5% for any feature, THE System SHALL raise a data quality error
3. WHEN external features are loaded, THE System SHALL check for outliers (values beyond mean ± 5 standard deviations)
4. WHERE outliers are detected, THE System SHALL log warnings with specific dates and values
5. WHEN external features are loaded, THE System SHALL validate that all features have exactly 469 weeks of data
6. WHEN external features are loaded, THE System SHALL validate that dates are contiguous (no gaps in weekly sequence)
7. THE System SHALL generate a data quality report saved to `logs/external_features_quality_report.txt`

### Requirement 16: Train-Validation-Test Split Consistency

**User Story:** As a data scientist, I want consistent data splits across flu data and external features, so that there is no data leakage between splits.

#### Acceptance Criteria

1. THE System SHALL use the same train-validation-test split ratios (70%-10%-20%) for flu data and external features
2. WHEN normalization statistics are computed, THE System SHALL use only training split data
3. WHEN lagged or rolling features are computed for validation or test splits, THE System SHALL use only historical data available at that time point
4. THE System SHALL validate that no future information from validation or test splits is used in training data
5. WHEN external features are split, THE System SHALL log the exact week boundaries for train, validation, and test sets

### Requirement 17: Model Input Dimension Update

**User Story:** As a developer, I want model configurations to automatically update input dimensions, so that models can handle multivariate inputs without manual changes.

#### Acceptance Criteria

1. WHEN external features are enabled, THE System SHALL compute total input dimension as 1 (flu data) + number of external features
2. THE System SHALL update the `enc_in` parameter in model configuration to match the total input dimension
3. WHERE a model is loaded from checkpoint, THE System SHALL validate that checkpoint `enc_in` matches current data `enc_in`
4. WHERE `enc_in` mismatch is detected, THE System SHALL raise an error with checkpoint and current dimension values
5. WHEN a new model is initialized, THE System SHALL log the input dimension and feature names being used

### Requirement 18: Backward Compatibility

**User Story:** As a researcher, I want to run existing models without external features, so that baseline comparisons are possible.

#### Acceptance Criteria

1. WHERE `use_external_features` is False, THE System SHALL load and process data identically to the original implementation
2. WHERE `use_external_features` is False, THE System SHALL produce identical results to the original system (given same random seed)
3. THE System SHALL support running experiments with and without external features using the same command-line interface
4. WHEN backward compatibility mode is active, THE System SHALL log "Running in backward compatibility mode (no external features)"

### Requirement 19: Feature Importance Analysis (Ablation Study)

**User Story:** As a researcher, I want to measure the contribution of each feature type, so that I can identify which external features provide the most value.

#### Acceptance Criteria

1. THE System SHALL provide a script `ablation_study.py` for feature importance analysis
2. WHEN ablation study is run, THE System SHALL train models with all possible subsets of feature types (weather only, trends only, weather+trends, etc.)
3. WHEN each model is trained, THE System SHALL record MSE, MAE, and RMSE on the test set
4. THE System SHALL compute improvement percentage for each feature type as (baseline_error - with_feature_error) / baseline_error × 100%
5. WHEN ablation study is complete, THE System SHALL generate a results table showing performance metrics for each feature combination
6. THE System SHALL save ablation study results to `results/ablation_study_results.csv`
7. THE System SHALL generate a bar chart visualizing feature importance and save to `results/ablation_study_chart.png`

### Requirement 20: Performance Comparison with Baseline

**User Story:** As a researcher, I want to compare forecasting performance with and without external features, so that I can validate the expected 15-25% improvement.

#### Acceptance Criteria

1. THE System SHALL train a baseline model using only flu positive_rate data
2. THE System SHALL train an enhanced model using flu data + all enabled external features
3. WHEN both models are trained, THE System SHALL evaluate on the same test set (last 20% of 469 weeks)
4. THE System SHALL compute MSE, MAE, RMSE, and MAPE for both baseline and enhanced models
5. THE System SHALL compute improvement percentage as (baseline_metric - enhanced_metric) / baseline_metric × 100%
6. WHEN comparison is complete, THE System SHALL log improvement percentages for all metrics
7. WHERE improvement is negative (performance degradation), THE System SHALL log a warning message

### Requirement 21: Visualization of External Features

**User Story:** As a data scientist, I want to visualize external features alongside flu data, so that I can understand temporal correlations.

#### Acceptance Criteria

1. THE System SHALL provide a script `visualize_features.py` for generating feature plots
2. WHEN visualization is run, THE System SHALL create a multi-panel plot with flu data + all external features over time
3. WHEN visualization is generated, THE System SHALL use dual y-axes for features with different scales
4. THE System SHALL highlight seasonal patterns using color-coded backgrounds (flu season vs. off-season)
5. WHEN correlation analysis is requested, THE System SHALL compute and display Pearson correlation coefficients between flu data and each external feature
6. THE System SHALL generate separate plots for each region (North China, South China, USA)
7. THE System SHALL save visualizations to `results/features_visualization_{region}.png`

### Requirement 22: API Integration Error Handling

**User Story:** As a developer, I want robust error handling for external API failures, so that data collection is resilient to temporary outages.

#### Acceptance Criteria

1. WHEN an API request fails with HTTP status 5xx, THE Feature_Collector SHALL retry up to 3 times with exponential backoff (2s, 4s, 8s)
2. WHEN an API request fails with HTTP status 429 (rate limit), THE Feature_Collector SHALL wait for the duration specified in Retry-After header before retrying
3. WHEN an API request fails with HTTP status 4xx (client error), THE Feature_Collector SHALL log the error and skip retries
4. WHERE all retry attempts are exhausted, THE Feature_Collector SHALL raise an exception with the last error message
5. WHEN API request succeeds after retry, THE Feature_Collector SHALL log the number of attempts required
6. THE Feature_Collector SHALL implement timeouts (30 seconds per request) to prevent indefinite hangs

### Requirement 23: Caching of External Features

**User Story:** As a developer, I want to cache collected external features, so that repeated experiments do not require re-fetching data from APIs.

#### Acceptance Criteria

1. THE System SHALL create a cache directory `data/external_features_cache/`
2. WHEN external features are collected, THE System SHALL save raw data to cache as `{feature_type}_{region}_{start_date}_{end_date}.csv`
3. WHEN external features are requested, THE System SHALL check cache first before making API calls
4. WHERE cached data exists and is not expired (older than 30 days), THE System SHALL load from cache
5. WHERE cached data is expired or missing, THE System SHALL fetch from API and update cache
6. THE System SHALL provide a command-line option `--force-refresh` to bypass cache and fetch fresh data
7. WHEN cache is used, THE System SHALL log "Loading {feature_type} from cache for {region}"

### Requirement 24: Documentation and Examples

**User Story:** As a new user, I want clear documentation on how to use external features, so that I can quickly integrate them into my workflow.

#### Acceptance Criteria

1. THE System SHALL provide a README section titled "External Features Integration"
2. THE documentation SHALL include step-by-step instructions for obtaining required API keys
3. THE documentation SHALL provide example configuration files for common use cases
4. THE documentation SHALL include a complete example script demonstrating training with external features
5. THE documentation SHALL describe each external feature type and its expected impact on forecasting
6. THE documentation SHALL include troubleshooting section for common API and data quality issues
7. THE documentation SHALL provide performance benchmarks (baseline vs. with external features) for each region

### Requirement 25: Unit Tests for Feature Processing

**User Story:** As a developer, I want comprehensive unit tests for feature processing modules, so that correctness is guaranteed.

#### Acceptance Criteria

1. THE System SHALL provide unit tests for temporal alignment covering daily-to-weekly and monthly-to-weekly conversion
2. THE System SHALL provide unit tests for spatial alignment covering region mapping
3. THE System SHALL provide unit tests for normalization covering z-score computation and application
4. THE System SHALL provide unit tests for lagged features covering no-leakage guarantee
5. THE System SHALL provide unit tests for rolling statistics covering edge cases (beginning of time series)
6. THE System SHALL provide unit tests for interaction features covering NaN and infinite value handling
7. THE System SHALL achieve minimum 80% code coverage for all feature processing modules
8. WHEN tests are run, THE System SHALL validate round-trip properties: parse(format(features)) SHALL equal original features (for serialization)
