import numpy as np
import pandas as pd
import h5py
from src.evaluation_metrics_fixed import concordance_index_censored

# Load one of the stacked predictions
stacked_path = "results/final_deploy/stacked_predictions/group_1_temporal_stacked.h5"
with h5py.File(stacked_path, 'r') as f:
    temporal_pred = f['predictions'][:]
    print(f"Temporal prediction shape: {temporal_pred.shape}")

# Load ground truth
temporal_test = pd.read_csv("results/final_deploy/temporal_test_labels.csv")
print(f"\nTemporal test shape: {temporal_test.shape}")
print(f"Event distribution:\n{temporal_test['event'].value_counts().sort_index()}")

# Test C-index calculation for Event 1
event1_mask = temporal_test['event'] != 2
times_event1 = temporal_test.loc[event1_mask, 'time'].values
events_event1 = temporal_test.loc[event1_mask, 'event'].values

# Get predictions at last time point for event 1
pred_event1 = temporal_pred[0, -1, event1_mask]

print(f"\nEvent 1 analysis:")
print(f"Number of samples (excluding event 2): {len(times_event1)}")
print(f"Number of actual Event 1 occurrences: {np.sum(events_event1 == 1)}")
print(f"Prediction range: [{pred_event1.min():.4f}, {pred_event1.max():.4f}]")

# Calculate C-index with positive values (higher CIF = higher risk)
c_index_pos = concordance_index_censored(
    events_event1 == 1,
    times_event1,
    pred_event1
)[0]

# Calculate C-index with negative values (for comparison)
c_index_neg = concordance_index_censored(
    events_event1 == 1,
    times_event1,
    -pred_event1
)[0]

# Also try with 1 - predictions (inverse probability)
c_index_inv = concordance_index_censored(
    events_event1 == 1,
    times_event1,
    1 - pred_event1
)[0]

print(f"\nC-index results:")
print(f"With positive predictions (higher CIF = higher risk): {c_index_pos:.4f}")
print(f"With negative predictions: {c_index_neg:.4f}")
print(f"With 1 - predictions (survival probability): {c_index_inv:.4f}")

# Check if there's a correlation between predictions and actual events
event1_occurred = events_event1 == 1
print(f"\nMean prediction for Event 1 occurred: {pred_event1[event1_occurred].mean():.4f}")
print(f"Mean prediction for Event 1 NOT occurred: {pred_event1[~event1_occurred].mean():.4f}")

# Check the actual ensemble calculation (average of 2 models)
print("\n" + "="*50)
print("Testing ensemble of first 2 models:")

# Load both individual predictions
all_predictions = []
for i in [1, 2]:  # Models 1 and 2
    pred_path = f"results/final_deploy/individual_predictions/Ensemble_model{i}_DeepSurv_ANN_Event1_None_CI_temporal_predictions.h5"
    with h5py.File(pred_path, 'r') as f:
        pred = f['predictions'][:]
        all_predictions.append(pred)
        print(f"Model {i} shape: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")

# Average the predictions
ensemble_pred = np.mean(all_predictions, axis=0)
print(f"\nEnsemble shape: {ensemble_pred.shape}, range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")

# Calculate C-index for ensemble at last time point
ensemble_pred_last = ensemble_pred[-1, :]  # Last time point
c_index_ensemble = concordance_index_censored(
    temporal_test['event'] == 1,
    temporal_test['time'].values,
    ensemble_pred_last
)[0]

print(f"Ensemble C-index (all samples, event 1): {c_index_ensemble:.4f}")