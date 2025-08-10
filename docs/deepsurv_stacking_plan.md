# DeepSurv Model Stacking Plan

## Overview
This document outlines the plan for stacking DeepSurv models to create 12 groups, each producing a CIF array of shape `(2, 5, n_samples)` to match the DeepHit format.

## Current State
- **24 DeepSurv models**: Each produces CIF of shape `(1825, n_samples)`
- **Need**: 12 stacked models, each with shape `(2, 5, n_samples)`

## Key Challenges

### 1. Time Point Mismatch
- **DeepSurv**: 1825 daily time points (0-1825 days)
- **DeepHit**: 5 yearly time points (365, 730, 1095, 1460, 1825 days)
- **Solution**: Extract predictions at the 5 specific time points from DeepSurv outputs

### 2. Event Stacking
- **DeepSurv**: Single event per model
- **Need**: Stack Event 1 and Event 2 predictions together
- **Solution**: Create shape `(2, 5, n_samples)` where:
  - `[0, :, :]` = Event 1 predictions at 5 time points
  - `[1, :, :]` = Event 2 predictions at 5 time points

## Model Groups (from model_grouping_summary.md)

### Group 1: ANN + None + Concordance Index
- Event 1: Model 1
- Event 2: Model 3

### Group 2: ANN + None + Log-likelihood
- Event 1: Model 2
- Event 2: Model 4

### Group 3: ANN + NearMiss version 1 + Concordance Index
- Event 1: Model 5
- Event 2: Model 7

### Group 4: ANN + NearMiss version 1 + Log-likelihood
- Event 1: Model 6
- Event 2: Model 8

### Group 5: ANN + KNN + Concordance Index
- Event 1: Model 9
- Event 2: Model 11

### Group 6: ANN + KNN + Log-likelihood
- Event 1: Model 10
- Event 2: Model 12

### Group 7: LSTM + None + Concordance Index
- Event 1: Model 13
- Event 2: Model 15

### Group 8: LSTM + None + Log-likelihood
- Event 1: Model 14
- Event 2: Model 16

### Group 9: LSTM + NearMiss version 3 + Concordance Index
- Event 1: Model 17
- Event 2: Model 19

### Group 10: LSTM + NearMiss version 3 + Log-likelihood
- Event 1: Model 18
- Event 2: Model 20

### Group 11: LSTM + KNN + Concordance Index
- Event 1: Model 21
- Event 2: Model 23

### Group 12: LSTM + KNN + Log-likelihood
- Event 1: Model 22
- Event 2: Model 24

## Implementation Steps

### Step 1: Define Time Point Mapping
```python
# DeepHit time points (in days)
deephit_time_points = [365, 730, 1095, 1460, 1825]

# For DeepSurv predictions (1825 time points, 0-indexed)
# Map to closest indices
time_indices = [364, 729, 1094, 1459, 1824]  # 0-indexed
```

### Step 2: Create Stacking Function
```python
def stack_deepsurv_predictions(event1_cif, event2_cif, time_indices):
    """
    Stack two DeepSurv predictions into DeepHit format
    
    Args:
        event1_cif: shape (1825, n_samples)
        event2_cif: shape (1825, n_samples)
        time_indices: list of 5 indices to extract
    
    Returns:
        stacked_cif: shape (2, 5, n_samples)
    """
    n_samples = event1_cif.shape[1]
    stacked_cif = np.zeros((2, 5, n_samples))
    
    # Extract predictions at specific time points
    stacked_cif[0] = event1_cif[time_indices]  # Event 1
    stacked_cif[1] = event2_cif[time_indices]  # Event 2
    
    return stacked_cif
```

### Step 3: Process Each Group
For each of the 12 groups:
1. Load Event 1 predictions
2. Load Event 2 predictions
3. Stack using the function above
4. Save as new H5 file with naming convention: `stacked_group{N}_predictions_{dataset}.h5`

### Step 4: Validation
After stacking, verify:
1. Shape is correct: `(2, 5, n_samples)`
2. Values at extracted time points match original
3. Event competition is maintained (sum ≤ 1)
4. Monotonicity is preserved

## Output Structure

### File Naming Convention
```
results/final_deploy/stacked_predictions/
├── temporal_stacked_group1_20250808_HHMMSS.h5
├── spatial_stacked_group1_20250808_HHMMSS.h5
├── temporal_stacked_group2_20250808_HHMMSS.h5
├── spatial_stacked_group2_20250808_HHMMSS.h5
...
├── temporal_stacked_group12_20250808_HHMMSS.h5
└── spatial_stacked_group12_20250808_HHMMSS.h5
```

### H5 File Structure
```
/predictions  # Dataset of shape (2, 5, n_samples)
/metadata     # Attributes including:
  - group_number
  - event1_model
  - event2_model
  - time_points
  - stacking_timestamp
```

## Summary JSON
Create a summary file documenting:
- All 12 stacked groups
- Source models for each group
- File paths for stacked predictions
- Validation results

## Integration with Ensemble Pipeline

After stacking, we'll have:
- **12 stacked DeepSurv groups**: Shape `(2, 5, n_samples)`
- **12 DeepHit models**: Shape `(2, 5, n_samples)`
- **Total**: 24 models with consistent shape for ensemble combination

This allows the ensemble pipeline to:
1. Load all 24 predictions with same shape
2. Apply ensemble methods (averaging, weighted combination, etc.)
3. Evaluate ensemble performance

## Next Steps

1. Implement the stacking logic in a ZenML step
2. Run stacking for all 12 groups
3. Validate stacked predictions
4. Update ensemble pipeline to use stacked predictions
5. Run final ensemble evaluation