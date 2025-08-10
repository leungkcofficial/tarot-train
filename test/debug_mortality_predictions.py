#!/usr/bin/env python3
"""
Debug script to investigate why DeepHit mortality predictions are showing 100% risk.
"""

import numpy as np
import pandas as pd
import h5py
import sys
import os

# Add src to path
sys.path.append('src')
from util import load_predictions_from_hdf5

def debug_mortality_predictions():
    """Debug the mortality prediction values to understand why they're 100%."""
    
    print("=== DeepHit Mortality Prediction Debug ===\n")
    
    # Use the most recent spatial test prediction file based on the calibration plot timestamp
    prediction_file = '/mnt/dump/yard/projects/tarot2/results/test_predictions/spatial_test_predictions_20250721_234135.h5'
    
    if not os.path.exists(prediction_file):
        print(f"Prediction file not found: {prediction_file}")
        # Try to find any recent spatial test file
        prediction_files = []
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('_predictions.h5') and 'spatial_test' in file:
                    prediction_files.append(os.path.join(root, file))
        
        if not prediction_files:
            print("No prediction files found!")
            return
        
        # Use the most recent file
        prediction_file = max(prediction_files, key=os.path.getmtime)
    print(f"Using prediction file: {prediction_file}")
    
    # Load predictions directly from HDF5
    print("\n1. Loading raw predictions from HDF5...")
    try:
        with h5py.File(prediction_file, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")
            
            if 'predictions' in f:
                predictions_array = f['predictions'][:]
                print(f"Raw predictions shape: {predictions_array.shape}")
                print(f"Raw predictions dtype: {predictions_array.dtype}")
                
                if len(predictions_array.shape) == 3:
                    # Competing risks format: (2, 5, n_samples)
                    num_causes, num_time_points, num_samples = predictions_array.shape
                    print(f"Competing risks format detected:")
                    print(f"  - Causes: {num_causes}")
                    print(f"  - Time points: {num_time_points}")
                    print(f"  - Samples: {num_samples}")
                    
                    # Extract CIF for each cause
                    cif_cause1 = predictions_array[0]  # Event 1 (RRT/eGFR<15)
                    cif_cause2 = predictions_array[1]  # Event 2 (Mortality)
                    
                    print(f"\n2. Analyzing CIF values...")
                    print(f"Cause 1 (RRT/eGFR<15) CIF:")
                    print(f"  - Shape: {cif_cause1.shape}")
                    print(f"  - Min: {cif_cause1.min():.6f}")
                    print(f"  - Max: {cif_cause1.max():.6f}")
                    print(f"  - Mean: {cif_cause1.mean():.6f}")
                    print(f"  - Std: {cif_cause1.std():.6f}")
                    
                    print(f"\nCause 2 (Mortality) CIF:")
                    print(f"  - Shape: {cif_cause2.shape}")
                    print(f"  - Min: {cif_cause2.min():.6f}")
                    print(f"  - Max: {cif_cause2.max():.6f}")
                    print(f"  - Mean: {cif_cause2.mean():.6f}")
                    print(f"  - Std: {cif_cause2.std():.6f}")
                    
                    # Check if mortality predictions are saturated
                    mortality_near_100 = (cif_cause2 > 0.99).sum()
                    mortality_exactly_1 = (cif_cause2 == 1.0).sum()
                    total_predictions = cif_cause2.size
                    
                    print(f"\n3. Mortality saturation analysis:")
                    print(f"  - Predictions > 99%: {mortality_near_100}/{total_predictions} ({100*mortality_near_100/total_predictions:.1f}%)")
                    print(f"  - Predictions exactly 100%: {mortality_exactly_1}/{total_predictions} ({100*mortality_exactly_1/total_predictions:.1f}%)")
                    
                    # Check time progression
                    print(f"\n4. Time progression analysis:")
                    for t in range(num_time_points):
                        time_point_mortality = cif_cause2[t]
                        print(f"  Time {t+1}: Min={time_point_mortality.min():.6f}, Max={time_point_mortality.max():.6f}, Mean={time_point_mortality.mean():.6f}")
                    
                    # Check if CIF values are monotonically increasing over time
                    print(f"\n5. Monotonicity check (CIF should increase over time):")
                    for sample_idx in range(min(10, num_samples)):  # Check first 10 samples
                        sample_mortality = cif_cause2[:, sample_idx]
                        is_monotonic = np.all(np.diff(sample_mortality) >= 0)
                        print(f"  Sample {sample_idx}: {sample_mortality} - Monotonic: {is_monotonic}")
                    
                    # Check for competing risks constraint (CIF1 + CIF2 <= 1)
                    print(f"\n6. Competing risks constraint check (CIF1 + CIF2 <= 1):")
                    total_cif = cif_cause1 + cif_cause2
                    violations = (total_cif > 1.0).sum()
                    max_total = total_cif.max()
                    print(f"  - Violations (CIF1 + CIF2 > 1): {violations}/{total_predictions} ({100*violations/total_predictions:.1f}%)")
                    print(f"  - Maximum total CIF: {max_total:.6f}")
                    
                    if violations > 0:
                        print(f"  - This indicates a problem with the DeepHit model predictions!")
                    
                else:
                    print(f"Unexpected prediction shape: {predictions_array.shape}")
                    
            if 'time_grid' in f:
                time_grid = f['time_grid'][:]
                print(f"Time grid: {time_grid}")
                
    except Exception as e:
        print(f"Error loading predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Also load using the utility function to see how it's processed
    print(f"\n7. Loading via utility function...")
    try:
        predictions_df = load_predictions_from_hdf5(prediction_file)
        print(f"Processed predictions shape: {predictions_df.shape}")
        print(f"Processed predictions columns: {list(predictions_df.columns)[:10]}...")  # First 10 columns
        print(f"Processed predictions index: {list(predictions_df.index)[:10]}...")  # First 10 rows
        
        # Look for mortality-related rows
        mortality_rows = [idx for idx in predictions_df.index if 'cause_2' in str(idx)]
        if mortality_rows:
            print(f"Found {len(mortality_rows)} mortality prediction rows")
            mortality_data = predictions_df.loc[mortality_rows]
            print(f"Mortality predictions range: [{mortality_data.min().min():.6f}, {mortality_data.max().max():.6f}]")
        
    except Exception as e:
        print(f"Error with utility function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mortality_predictions()