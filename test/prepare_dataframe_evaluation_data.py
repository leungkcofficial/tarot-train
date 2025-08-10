"""
Script to prepare data for DataFrame-based evaluation by loading individual predictions
and saving them in the format expected by the evaluation scripts.
"""

import numpy as np
import h5py
import pickle
import os
import pandas as pd
from datetime import datetime


def load_latest_predictions():
    """Load the latest predictions for all 36 models."""
    
    predictions_dir = "results/final_deploy/individual_predictions"
    
    # Dictionary to store all predictions
    temporal_predictions = {}
    spatial_predictions = {}
    
    # Load predictions for models 1-36
    print("Loading individual model predictions...")
    
    # Find the latest prediction file for each model
    for model_id in range(1, 37):
        print(f"Loading model {model_id}...")
        
        # Find temporal prediction files for this model
        temporal_files = [f for f in os.listdir(predictions_dir) 
                         if f.startswith(f"temporal_predictions_model{model_id}_") 
                         and f.endswith(".h5")]
        
        # Find spatial prediction files for this model
        spatial_files = [f for f in os.listdir(predictions_dir) 
                        if f.startswith(f"spatial_predictions_model{model_id}_") 
                        and f.endswith(".h5")]
        
        if temporal_files:
            # Sort by timestamp and get the latest
            temporal_files.sort()
            latest_temporal = temporal_files[-1]
            
            # Load temporal predictions
            with h5py.File(os.path.join(predictions_dir, latest_temporal), 'r') as f:
                temporal_predictions[f"model_{model_id}"] = f['predictions'][:]
                print(f"  Loaded temporal: {latest_temporal}, shape: {f['predictions'][:].shape}")
        else:
            print(f"  WARNING: No temporal predictions found for model {model_id}")
        
        if spatial_files:
            # Sort by timestamp and get the latest
            spatial_files.sort()
            latest_spatial = spatial_files[-1]
            
            # Load spatial predictions
            with h5py.File(os.path.join(predictions_dir, latest_spatial), 'r') as f:
                spatial_predictions[f"model_{model_id}"] = f['predictions'][:]
                print(f"  Loaded spatial: {latest_spatial}, shape: {f['predictions'][:].shape}")
        else:
            print(f"  WARNING: No spatial predictions found for model {model_id}")
    
    return temporal_predictions, spatial_predictions


def load_ground_truth_labels():
    """Load ground truth labels from CSV files."""
    
    print("\nLoading ground truth labels...")
    
    # Load temporal labels
    temporal_labels_df = pd.read_csv('results/final_deploy/temporal_test_labels.csv')
    temporal_labels = {
        'event_indicators': temporal_labels_df['event'].values,
        'event_times': temporal_labels_df['time'].values
    }
    print(f"Loaded temporal labels: {len(temporal_labels_df)} samples")
    
    # Load spatial labels
    spatial_labels_df = pd.read_csv('results/final_deploy/spatial_test_labels.csv')
    spatial_labels = {
        'event_indicators': spatial_labels_df['event'].values,
        'event_times': spatial_labels_df['time'].values
    }
    print(f"Loaded spatial labels: {len(spatial_labels_df)} samples")
    
    return temporal_labels, spatial_labels


def save_prepared_data():
    """Load all data and save in pickle format for evaluation."""
    
    print("Preparing data for DataFrame-based evaluation...")
    print("="*60)
    
    # Load predictions
    temporal_predictions, spatial_predictions = load_latest_predictions()
    
    # Load labels
    temporal_labels, spatial_labels = load_ground_truth_labels()
    
    # Verify we have all 36 models
    print(f"\nVerifying data completeness:")
    print(f"Temporal predictions: {len(temporal_predictions)} models")
    print(f"Spatial predictions: {len(spatial_predictions)} models")
    
    missing_temporal = [i for i in range(1, 37) if f"model_{i}" not in temporal_predictions]
    missing_spatial = [i for i in range(1, 37) if f"model_{i}" not in spatial_predictions]
    
    if missing_temporal:
        print(f"WARNING: Missing temporal predictions for models: {missing_temporal}")
    if missing_spatial:
        print(f"WARNING: Missing spatial predictions for models: {missing_spatial}")
    
    # Save data in pickle format
    print("\nSaving prepared data...")
    
    with open('results/final_deploy/temporal_predictions.pkl', 'wb') as f:
        pickle.dump(temporal_predictions, f)
    print("Saved: results/final_deploy/temporal_predictions.pkl")
    
    with open('results/final_deploy/spatial_predictions.pkl', 'wb') as f:
        pickle.dump(spatial_predictions, f)
    print("Saved: results/final_deploy/spatial_predictions.pkl")
    
    with open('results/final_deploy/temporal_labels.pkl', 'wb') as f:
        pickle.dump(temporal_labels, f)
    print("Saved: results/final_deploy/temporal_labels.pkl")
    
    with open('results/final_deploy/spatial_labels.pkl', 'wb') as f:
        pickle.dump(spatial_labels, f)
    print("Saved: results/final_deploy/spatial_labels.pkl")
    
    print("\n✓ Data preparation complete!")
    print("You can now run the evaluation scripts:")
    print("  - python test_dataframe_approach_simple.py")
    print("  - python run_dataframe_evaluation.py")


if __name__ == "__main__":
    save_prepared_data()