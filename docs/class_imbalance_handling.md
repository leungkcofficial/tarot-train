# Class Imbalance Handling for Deep Learning Survival Models

## Overview

This document describes the implementation of class imbalance handling for deep learning survival models in the CKD risk prediction system. The feature addresses the common issue in survival analysis where censored records (endpoint = 0) significantly outnumber event records (endpoint = 1 or 2), leading to biased model training.

## Problem Statement

In survival analysis datasets for CKD risk prediction:

- The majority of records are censored (no event occurred during the observation period)
- Events of interest (e.g., CKD progression, kidney failure) are relatively rare
- This imbalance can cause models to be biased toward predicting non-events
- Models may underestimate risk for patients who are actually at high risk

For example, a typical dataset might have:
- 90% censored records (endpoint = 0)
- 10% event records (endpoint = 1 or 2)

This imbalance is particularly challenging for deep learning models, which may struggle to learn patterns from the minority class.

## Implementation

The solution implements under-sampling of the majority class (censored records) to create a more balanced training dataset. This approach preserves all minority class samples while reducing the number of majority class samples.

### Key Components

1. **`balance_dataframe` Function**: Core implementation in `src/balance_data.py`
2. **Integration with Model Training**: Implementation in `steps/model_train.py`
3. **Configuration Options**: Settings in `src/hyperparameter_config.yml`
4. **MLflow Logging**: Tracking of balancing metrics
5. **Unit Tests**: Verification in `tests/test_balance_data.py`

### Supported Models

The implementation supports both:

- **DeepSurv** (binary outcome): For traditional survival analysis with a single event type
- **DeepHit** (multiple competitive outcomes): For competing risks scenarios with multiple event types

### Supported Under-sampling Methods

The following under-sampling methods are supported:

1. **Random Under-sampling** (`random_under_sampler`): Randomly removes samples from the majority class
2. **NearMiss** (`near_miss`): Selects samples from the majority class based on their distance to minority class samples
3. **Cluster Centroids** (`cluster_centroids`): Replaces clusters of majority samples with their centroids
4. **Tomek Links** (`tomek_links`): Removes majority samples that form Tomek links with minority samples

## Configuration

Class imbalance handling is configured in the `balance` section of `src/hyperparameter_config.yml`:

```yaml
# Class imbalance handling
balance:
  enable: true           # Set false to disable balancing
  method: "near_miss"    # Options: random_under_sampler, near_miss, cluster_centroids, tomek_links
  sampling_strategy: "majority"   # Fixed – do not change
  near_miss_version: 1   # Version of NearMiss algorithm (1, 2, or 3)
```

### Configuration Options

| Option | Description | Default | Allowed Values |
|--------|-------------|---------|----------------|
| `enable` | Enable/disable balancing | `true` | `true`, `false` |
| `method` | Under-sampling method | `"random_under_sampler"` | `"random_under_sampler"`, `"near_miss"`, `"cluster_centroids"`, `"tomek_links"` |
| `sampling_strategy` | Sampling strategy | `"majority"` | Fixed to `"majority"` to ensure no synthetic rows are created |
| `near_miss_version` | Version of NearMiss algorithm | `1` | `1`, `2`, `3` |

## Usage

The balancing functionality is automatically applied during model training if enabled in the configuration. The process works as follows:

1. The training pipeline loads the hyperparameter configuration
2. If `balance.enable` is `true`, the `balance_dataframe` function is called
3. The function under-samples the majority class based on the specified method
4. The balanced dataset is used for cross-validation and model training
5. Balancing metrics are logged to MLflow

### Example Output

When balancing is applied, you'll see output similar to:

```
=== Applying class balancing to training data ===
Model type: deepsurv
Target endpoint: None
Original event distribution: {0.0: 900, 1.0: 100}
Detected majority class: 0.0
Minority class percentage: 10.00%
Applying random_under_sampler with sampling_strategy='majority'
Balanced event distribution: {0.0: 100, 1.0: 100}
Reduced dataset by 80.00% (from 1000 to 200 rows)
Original rows: 1000
Balanced rows: 200
Removed rows: 800 (80.00%)
```

## Special Handling for Different Model Types

### DeepSurv (Binary Outcome)

For DeepSurv models:
- If `target_endpoint` is not specified, all non-zero events are treated as events
- If `target_endpoint` is specified, only that specific event type is considered as an event
- The function creates a binary version of the event column for balancing
- After balancing, the original event values are preserved

### DeepHit (Multiple Competitive Outcomes)

For DeepHit models:
- All event types are preserved
- The majority class (typically 0 for censored) is under-sampled
- The function checks for very small minority classes and issues warnings
- The function checks for uneven minority classes and issues warnings

## Performance Considerations

### Benefits

- **Improved Model Performance**: Models trained on balanced data often show improved discrimination for high-risk patients
- **Better Calibration**: Risk predictions tend to be better calibrated across the risk spectrum
- **Faster Training**: Smaller datasets lead to faster training times

### Potential Drawbacks

- **Information Loss**: Under-sampling discards potentially useful majority class samples
- **Reduced Dataset Size**: The balanced dataset is smaller, which may affect model generalization
- **Potential Overfitting**: With fewer samples, models may be more prone to overfitting

### Warnings

The implementation includes several warnings:
- When the minority class is very small (<5% of the data)
- When minority classes have very different sizes (for DeepHit)
- When under-sampling would result in excessive information loss

## Dependencies

The implementation relies on the `imbalanced-learn` library, which is included in the project's requirements:

```
imbalanced-learn>=0.12.0  # Required for class-imbalance handling
```

## Future Improvements

Potential future enhancements include:

1. **Over-sampling Methods**: Adding support for over-sampling methods like SMOTE
2. **Hybrid Approaches**: Combining under-sampling and over-sampling
3. **Cost-sensitive Learning**: Implementing cost-sensitive learning as an alternative
4. **Adaptive Balancing**: Dynamically adjusting the balancing ratio based on dataset characteristics
5. **Ensemble Approaches**: Training multiple models on different balanced subsets of the data

## References

1. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
2. Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17), 1-5.
3. Kvamme, H., Borgan, Ø., & Scheel, I. (2019). Time-to-event prediction with neural networks and Cox regression. Journal of Machine Learning Research, 20(129), 1-30.