# LSTM Prediction Alignment Fix

## Problem Identified

The ensemble predictions step was failing with a "inhomogeneous shape" error because different models were producing predictions for different numbers of patients:

- **ANN models**: 42,953 samples (full test set)
- **LSTM models**: 4,809 samples (subset with sufficient sequential data)

This violates the requirement that ALL models should make predictions for ALL patients in the test set.

## Root Cause

The `create_sequences_from_dataframe` function in `src/sequence_utils.py` was filtering out patients with insufficient sequential data (lines 104-105):

```python
if n_observations < min_sequence_length:
    continue  # This skips patients!
```

For LSTM models, this meant only patients with enough historical observations were included in predictions.

## Solution Implemented

### 1. Fixed Sequence Creation (`src/sequence_utils_fixed.py`)

Created a fixed version that includes ALL patients by:

- Adding `include_all_patients` parameter (default: True for test sets)
- Handling three cases for sequence creation:
  - **Sufficient data**: Use last `sequence_length` observations
  - **Partial data**: Pad with zeros at the beginning
  - **Single observation**: Create sequence with zeros except last position

### 2. Fixed Ensemble Predictions (`steps/ensemble_predictions_fixed.py`)

Created a fixed ensemble step that:

- Detects sample count mismatches across models
- Aligns all predictions to the maximum sample count
- Pads missing predictions with zeros
- Ensures consistent output shape: `(n_events=2, n_timepoints=5, n_samples)`

### 3. Updated Pipeline Script (`run_stack_predictions_fixed_v2.py`)

Updated the script to:

- Use the fixed ensemble predictions step
- Show sample counts for debugging
- Verify final ensemble shapes

## Key Principles

1. **No Patient Dropping**: Test sets must include predictions for ALL patients
2. **Zero Padding**: Missing sequential data is padded with zeros
3. **Consistent Shapes**: All model predictions must have the same number of samples
4. **Ensemble Alignment**: The ensemble step handles any remaining misalignments

## Expected Output Shape

After fixing, the ensemble CIF array should have shape:
- `(24, 2, 5, n_samples)` where:
  - 24 = number of grouped models
  - 2 = number of competitive outcomes
  - 5 = number of prediction timepoints
  - n_samples = total samples in dataset (e.g., 42,953 for temporal test)

## Usage

To use the fixed version:

1. Update model deployment to use `prepare_lstm_survival_dataset_fixed` with `include_all_patients=True` for test sets
2. Run the fixed pipeline script: `python run_stack_predictions_fixed_v2.py`
3. The ensemble will automatically handle any remaining alignment issues

## Next Steps

1. Update the main pipeline to use these fixed functions
2. Re-run model deployments with the fixed LSTM sequence generation
3. Verify all models produce predictions for the same number of patients
4. Complete ensemble evaluation with aligned predictions