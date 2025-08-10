"""
Script to fill in missing metrics in the evaluation results CSV.
Loads the stacked CIF predictions and calculates metrics for unfilled rows.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta


def load_stacked_predictions():
    """Load the pre-stacked CIF predictions."""
    print("Loading stacked predictions...")
    
    # Load from H5 files
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    
    print(f"Temporal stacked shape: {temporal_stacked.shape}")
    print(f"Spatial stacked shape: {spatial_stacked.shape}")
    
    return temporal_stacked, spatial_stacked


def load_labels():
    """Load ground truth labels."""
    print("Loading ground truth labels...")
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    return temporal_labels, spatial_labels


def evaluate_combination(model_indices, temporal_stacked, spatial_stacked, 
                        temporal_labels, spatial_labels):
    """
    Evaluate a single combination by slicing and averaging predictions.
    
    Args:
        model_indices: List of model indices to include in ensemble
        temporal_stacked: Pre-stacked temporal predictions (24, 2, 5, n_samples)
        spatial_stacked: Pre-stacked spatial predictions (24, 2, 5, n_samples)
        temporal_labels: Ground truth for temporal test set
        spatial_labels: Ground truth for spatial test set
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Slice predictions for this combination
    temporal_combo = temporal_stacked[model_indices, :, :, :]  # (n_models, 2, 5, n_samples)
    spatial_combo = spatial_stacked[model_indices, :, :, :]    # (n_models, 2, 5, n_samples)
    
    # Average across models
    temporal_ensemble = np.mean(temporal_combo, axis=0)  # (2, 5, n_samples)
    spatial_ensemble = np.mean(spatial_combo, axis=0)    # (2, 5, n_samples)
    
    # Time points for evaluation (in days)
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Calculate metrics
    temporal_metrics = calculate_all_metrics(
        temporal_labels['event_times'],      # times
        temporal_labels['event_indicators'],  # events
        temporal_ensemble,                    # predictions
        time_points                          # time_points
    )
    
    spatial_metrics = calculate_all_metrics(
        spatial_labels['event_times'],       # times
        spatial_labels['event_indicators'],   # events
        spatial_ensemble,                     # predictions
        time_points                          # time_points
    )
    
    return {
        'temporal_ibs': temporal_metrics['ibs'],
        'temporal_ibs_event1': temporal_metrics['ibs_event1'],
        'temporal_ibs_event2': temporal_metrics['ibs_event2'],
        'temporal_cidx_event1': temporal_metrics['cidx_event1'],
        'temporal_cidx_event2': temporal_metrics['cidx_event2'],
        'temporal_nll': temporal_metrics['nll'],
        'spatial_ibs': spatial_metrics['ibs'],
        'spatial_ibs_event1': spatial_metrics['ibs_event1'],
        'spatial_ibs_event2': spatial_metrics['ibs_event2'],
        'spatial_cidx_event1': spatial_metrics['cidx_event1'],
        'spatial_cidx_event2': spatial_metrics['cidx_event2'],
        'spatial_nll': spatial_metrics['nll']
    }


def fill_missing_metrics():
    """Fill in missing metrics in the evaluation results CSV."""
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return
    
    # Load the CSV
    print(f"\nLoading results from {results_file}...")
    df = pd.read_csv(results_file)
    
    # Convert model_indices from string to list
    df['model_indices'] = df['model_indices'].apply(eval)
    
    # Count missing metrics
    missing_mask = df['temporal_ibs'].isna()
    n_missing = missing_mask.sum()
    n_total = len(df)
    
    print(f"\nTotal rows: {n_total:,}")
    print(f"Rows with missing metrics: {n_missing:,}")
    print(f"Rows already filled: {n_total - n_missing:,}")
    
    if n_missing == 0:
        print("\nAll metrics are already filled!")
        return
    
    # Load stacked predictions and labels
    temporal_stacked, spatial_stacked = load_stacked_predictions()
    temporal_labels, spatial_labels = load_labels()
    
    # Process missing rows
    print(f"\nProcessing {n_missing:,} rows with missing metrics...")
    
    start_time = time.time()
    processed = 0
    
    # Get indices of rows with missing metrics
    missing_indices = df[missing_mask].index.tolist()
    
    for idx in missing_indices:
        # Get model indices for this combination
        model_indices = df.loc[idx, 'model_indices']
        
        try:
            # Calculate metrics
            metrics = evaluate_combination(
                model_indices,
                temporal_stacked,
                spatial_stacked,
                temporal_labels,
                spatial_labels
            )
            
            # Update DataFrame
            for key, value in metrics.items():
                df.loc[idx, key] = value
            
            processed += 1
            
            # Progress update every 1000 rows
            if processed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = n_missing - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"Progress: {processed:,}/{n_missing:,} ({processed/n_missing*100:.1f}%) | "
                      f"Rate: {rate:.1f} rows/sec | ETA: {eta_str}")
                
                # Save intermediate results
                df.to_csv(results_file, index=False)
                
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            continue
    
    # Final save
    df.to_csv(results_file, index=False)
    
    # Calculate final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"{'='*60}")
    print(f"Processed {processed:,} rows in {str(timedelta(seconds=int(total_time)))}")
    print(f"Average rate: {processed/total_time:.1f} rows/second")
    
    # Calculate averages and find best combinations
    print("\nCalculating best combinations...")
    
    df['temporal_cidx_avg'] = (df['temporal_cidx_event1'] + df['temporal_cidx_event2']) / 2
    df['spatial_cidx_avg'] = (df['spatial_cidx_event1'] + df['spatial_cidx_event2']) / 2
    df['overall_cidx_avg'] = (df['temporal_cidx_avg'] + df['spatial_cidx_avg']) / 2
    
    # Save updated file
    df.to_csv(results_file, index=False)
    
    # Sort by overall C-index and save top results
    df_sorted = df.sort_values('overall_cidx_avg', ascending=False)
    top_results_file = 'results/ensemble_checkpoints/top_100_results.csv'
    df_sorted.head(100).to_csv(top_results_file, index=False)
    
    print(f"\nSaved updated results to: {results_file}")
    print(f"Saved top 100 results to: {top_results_file}")
    
    # Show top 10
    print("\nTop 10 combinations by overall C-index:")
    top_10 = df_sorted[['combination_id', 'n_models', 'model_indices', 
                        'overall_cidx_avg', 'temporal_cidx_avg', 'spatial_cidx_avg']].head(10)
    print(top_10.to_string(index=False))


if __name__ == "__main__":
    fill_missing_metrics()