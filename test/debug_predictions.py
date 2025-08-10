import h5py
import numpy as np
import pandas as pd

# Load one of the stacked predictions
stacked_path = "results/final_deploy/stacked_predictions/group_1_temporal_stacked.h5"
with h5py.File(stacked_path, 'r') as f:
    temporal_pred = f['predictions'][:]
    print(f"Temporal stacked prediction shape: {temporal_pred.shape}")
    print(f"Temporal predictions range: [{temporal_pred.min():.4f}, {temporal_pred.max():.4f}]")
    
    # Check a few samples
    print("\nFirst 5 temporal predictions (Event 1, all time points):")
    print(temporal_pred[0, :, :5])
    print("\nFirst 5 temporal predictions (Event 2, all time points):")
    print(temporal_pred[1, :, :5])

# Also load spatial predictions
spatial_path = "results/final_deploy/stacked_predictions/group_1_spatial_stacked.h5"
with h5py.File(spatial_path, 'r') as f:
    spatial_pred = f['predictions'][:]
    print(f"\nSpatial stacked prediction shape: {spatial_pred.shape}")
    print(f"Spatial predictions range: [{spatial_pred.min():.4f}, {spatial_pred.max():.4f}]")

# Load ground truth to compare
temporal_test = pd.read_csv("results/final_deploy/temporal_test_labels.csv")
spatial_test = pd.read_csv("results/final_deploy/spatial_test_labels.csv")

print(f"\nTemporal test events distribution:")
print(temporal_test['event'].value_counts().sort_index())
print(f"\nSpatial test events distribution:")
print(spatial_test['event'].value_counts().sort_index())

# Check if predictions are monotonically increasing (as CIF should be)
print("\nChecking monotonicity of predictions...")
temporal_mono = np.all(np.diff(temporal_pred, axis=1) >= 0)
spatial_mono = np.all(np.diff(spatial_pred, axis=1) >= 0)
print(f"Temporal predictions monotonic: {temporal_mono}")
print(f"Spatial predictions monotonic: {spatial_mono}")

# Check the actual C-index calculation
from src.evaluation_metrics_fixed import concordance_index_censored

# For Event 1 in temporal data
event1_mask = temporal_test['event'] != 2
times_event1 = temporal_test.loc[event1_mask, 'time'].values
events_event1 = temporal_test.loc[event1_mask, 'event'].values

# Get predictions at last time point for event 1
pred_event1 = temporal_pred[0, -1, event1_mask]

# Calculate C-index with positive values
c_index_pos = concordance_index_censored(
    events_event1 == 1,
    times_event1,
    pred_event1  # Positive values
)[0]

# Calculate C-index with negative values
c_index_neg = concordance_index_censored(
    events_event1 == 1,
    times_event1,
    -pred_event1  # Negative values
)[0]

print(f"\nC-index with positive predictions: {c_index_pos:.4f}")
print(f"C-index with negative predictions: {c_index_neg:.4f}")
print(f"Expected range: [0.5, 1.0] where 0.5 is random and 1.0 is perfect")