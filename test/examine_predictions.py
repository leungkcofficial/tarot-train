#!/usr/bin/env python3
"""
Examine actual DeepHit prediction files to understand the shape and structure.
"""

import h5py
import pandas as pd
import numpy as np

def examine_prediction_file():
    """Examine the most recent DeepHit prediction file."""
    
    # Use the most recent prediction file
    prediction_file = "results/test_predictions/temporal_test_predictions_20250721_132629.h5"
    metadata_file = "results/test_predictions/temporal_test_metadata_20250721_132629.csv"
    
    print("=== DeepHit Prediction File Analysis ===\n")
    
    # Read metadata
    try:
        metadata = pd.read_csv(metadata_file)
        print(f"1. Metadata shape: {metadata.shape}")
        print(f"   Columns: {metadata.columns.tolist()}")
        print(f"   Sample metadata:")
        print(metadata.head())
        print()
    except Exception as e:
        print(f"Error reading metadata: {e}")
    
    # Read predictions
    try:
        with h5py.File(prediction_file, 'r') as f:
            print(f"2. HDF5 file structure:")
            def print_structure(name, obj):
                print(f"   {name}: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"      Shape: {obj.shape}")
                if hasattr(obj, 'dtype'):
                    print(f"      Dtype: {obj.dtype}")
            
            f.visititems(print_structure)
            print()
            
            # Get the main prediction dataset
            if 'predictions' in f:
                predictions = f['predictions'][:]
                print(f"3. Predictions array shape: {predictions.shape}")
                print(f"   Dtype: {predictions.dtype}")
                
                # Analyze the shape
                if len(predictions.shape) == 2:
                    rows, cols = predictions.shape
                    print(f"   - Rows: {rows}")
                    print(f"   - Columns: {cols}")
                    
                    # Check if this could be competing risks format
                    # For DeepHit with 2 causes and 5 time points, we expect:
                    # Either (10, n_samples) or (5, n_samples) depending on format
                    
                    if rows == 10:  # 2 causes × 5 time points
                        print(f"\n4. COMPETING RISKS FORMAT DETECTED!")
                        print(f"   - Likely format: 2 causes × 5 time points = 10 rows")
                        print(f"   - Cause 1 (Event 1): rows 0-4")
                        print(f"   - Cause 2 (Event 2): rows 5-9")
                        
                        # Show sample values
                        print(f"\n5. Sample predictions for first 3 patients:")
                        for patient in range(min(3, cols)):
                            print(f"   Patient {patient+1}:")
                            print(f"     Cause 1: {predictions[0:5, patient]}")
                            print(f"     Cause 2: {predictions[5:10, patient]}")
                            
                    elif rows == 5:  # Single cause format
                        print(f"\n4. SINGLE CAUSE FORMAT")
                        print(f"   - 5 time points, single survival function")
                        print(f"   - Sample predictions for first 3 patients:")
                        for patient in range(min(3, cols)):
                            print(f"     Patient {patient+1}: {predictions[:, patient]}")
                    
                    else:
                        print(f"\n4. UNKNOWN FORMAT")
                        print(f"   - {rows} rows doesn't match expected patterns")
                
                print(f"\n6. Time grid analysis:")
                # Try to infer time grid from the data structure
                time_grid = [365, 730, 1095, 1460, 1825]  # Expected time grid
                print(f"   Expected time grid: {time_grid}")
                print(f"   Number of time points: {len(time_grid)}")
                
            else:
                print("No 'predictions' dataset found in HDF5 file")
                
    except Exception as e:
        print(f"Error reading predictions: {e}")

if __name__ == "__main__":
    examine_prediction_file()