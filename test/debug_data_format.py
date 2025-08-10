#!/usr/bin/env python3
"""
Debug script to understand the actual data format after LabTransDiscreteTime transformation
"""

import pandas as pd
import numpy as np
import h5py

def examine_metadata():
    """Examine the metadata files to understand duration and event formats"""
    
    # Load metadata files
    temporal_metadata = "results/test_predictions/temporal_test_metadata_20250721_234135.csv"
    spatial_metadata = "results/test_predictions/spatial_test_metadata_20250721_234135.csv"
    
    print("=== TEMPORAL TEST METADATA ===")
    try:
        temporal_df = pd.read_csv(temporal_metadata)
        print(f"Shape: {temporal_df.shape}")
        print(f"Columns: {temporal_df.columns.tolist()}")
        
        if 'duration' in temporal_df.columns:
            durations = temporal_df['duration'].values
            print(f"Duration values (first 20): {durations[:20]}")
            print(f"Duration unique values: {np.unique(durations)}")
            print(f"Duration min: {durations.min()}, max: {durations.max()}")
            
        if 'event' in temporal_df.columns:
            events = temporal_df['event'].values
            print(f"Event values (first 20): {events[:20]}")
            print(f"Event unique values: {np.unique(events)}")
            print(f"Event value counts:")
            for val in np.unique(events):
                count = np.sum(events == val)
                print(f"  Event {val}: {count} samples ({count/len(events)*100:.1f}%)")
                
    except Exception as e:
        print(f"Error loading temporal metadata: {e}")
    
    print("\n=== SPATIAL TEST METADATA ===")
    try:
        spatial_df = pd.read_csv(spatial_metadata)
        print(f"Shape: {spatial_df.shape}")
        print(f"Columns: {spatial_df.columns.tolist()}")
        
        if 'duration' in spatial_df.columns:
            durations = spatial_df['duration'].values
            print(f"Duration values (first 20): {durations[:20]}")
            print(f"Duration unique values: {np.unique(durations)}")
            print(f"Duration min: {durations.min()}, max: {durations.max()}")
            
        if 'event' in spatial_df.columns:
            events = spatial_df['event'].values
            print(f"Event values (first 20): {events[:20]}")
            print(f"Event unique values: {np.unique(events)}")
            print(f"Event value counts:")
            for val in np.unique(events):
                count = np.sum(events == val)
                print(f"  Event {val}: {count} samples ({count/len(events)*100:.1f}%)")
                
    except Exception as e:
        print(f"Error loading spatial metadata: {e}")

def examine_predictions():
    """Examine the prediction files to understand the format"""
    
    temporal_predictions = "results/test_predictions/temporal_test_predictions_20250721_234135.h5"
    spatial_predictions = "results/test_predictions/spatial_test_predictions_20250721_234135.h5"
    
    print("\n=== TEMPORAL TEST PREDICTIONS ===")
    try:
        with h5py.File(temporal_predictions, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")
            if 'predictions' in f:
                preds = f['predictions'][:]
                print(f"Predictions shape: {preds.shape}")
                print(f"Predictions dtype: {preds.dtype}")
                print(f"Predictions sample (first patient, all time points): {preds[0, :, 0] if len(preds.shape) == 3 else preds[0, :]}")
                
    except Exception as e:
        print(f"Error loading temporal predictions: {e}")
    
    print("\n=== SPATIAL TEST PREDICTIONS ===")
    try:
        with h5py.File(spatial_predictions, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")
            if 'predictions' in f:
                preds = f['predictions'][:]
                print(f"Predictions shape: {preds.shape}")
                print(f"Predictions dtype: {preds.dtype}")
                print(f"Predictions sample (first patient, all time points): {preds[0, :, 0] if len(preds.shape) == 3 else preds[0, :]}")
                
    except Exception as e:
        print(f"Error loading spatial predictions: {e}")

def understand_labtrans():
    """Understand LabTransDiscreteTime transformation"""
    
    print("\n=== UNDERSTANDING LABTRANSDISCRETETIME ===")
    print("Based on PyCox documentation:")
    print("- LabTransDiscreteTime transforms continuous time to discrete intervals")
    print("- Duration becomes interval index: 0, 1, 2, 3, 4 for time intervals")
    print("- Time intervals correspond to: [365, 730, 1095, 1460, 1825] days")
    print("- Duration 0 = [0, 365) days")
    print("- Duration 1 = [365, 730) days") 
    print("- Duration 2 = [730, 1095) days")
    print("- Duration 3 = [1095, 1460) days")
    print("- Duration 4 = [1460, 1825) days")
    print("- Events remain: 0=censored, 1=dialysis/RRT, 2=mortality")

if __name__ == "__main__":
    examine_metadata()
    examine_predictions()
    understand_labtrans()