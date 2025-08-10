import h5py
import pandas as pd
import numpy as np

# Load the evaluation results
results_path = "results/final_deploy/ensemble_evaluation/ensemble_evaluation_results.h5"

with h5py.File(results_path, 'r') as f:
    print("Available datasets:")
    for key in f.keys():
        print(f"  {key}")
    
    # Load the metrics
    temporal_cidx_event1 = f['temporal_cidx_event1'][:]
    spatial_cidx_event1 = f['spatial_cidx_event1'][:]
    temporal_cidx_event2 = f['temporal_cidx_event2'][:]
    spatial_cidx_event2 = f['spatial_cidx_event2'][:]
    temporal_ibs = f['temporal_ibs'][:]
    spatial_ibs = f['spatial_ibs'][:]
    
    print(f"\nTemporal C-index Event 1: min={temporal_cidx_event1.min():.4f}, max={temporal_cidx_event1.max():.4f}, mean={temporal_cidx_event1.mean():.4f}")
    print(f"Spatial C-index Event 1: min={spatial_cidx_event1.min():.4f}, max={spatial_cidx_event1.max():.4f}, mean={spatial_cidx_event1.mean():.4f}")
    print(f"Temporal C-index Event 2: min={temporal_cidx_event2.min():.4f}, max={temporal_cidx_event2.max():.4f}, mean={temporal_cidx_event2.mean():.4f}")
    print(f"Spatial C-index Event 2: min={spatial_cidx_event2.min():.4f}, max={spatial_cidx_event2.max():.4f}, mean={spatial_cidx_event2.mean():.4f}")
    
    print(f"\nTemporal IBS: min={temporal_ibs.min():.4f}, max={temporal_ibs.max():.4f}, mean={temporal_ibs.mean():.4f}")
    print(f"Spatial IBS: min={spatial_ibs.min():.4f}, max={spatial_ibs.max():.4f}, mean={spatial_ibs.mean():.4f}")

# Also check the CSV
csv_path = "results/final_deploy/ensemble_evaluation/ensemble_evaluation_results.csv"
df = pd.read_csv(csv_path)
print(f"\nDataFrame shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for NaN values
print("\nNaN counts:")
print(df.isna().sum())

# Look at the distribution of C-index values
print("\nC-index distribution (should be between 0.5 and 1.0):")
print(df[['temporal_cidx_event1', 'spatial_cidx_event1', 'temporal_cidx_event2', 'spatial_cidx_event2']].describe())