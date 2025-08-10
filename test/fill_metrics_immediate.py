"""
Fill missing metrics with immediate row-by-row saving and clear progress logs.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
import sys


def process_and_save_row(idx, row, temporal_stacked, spatial_stacked, 
                        temporal_labels, spatial_labels, results_file):
    """Process a single row and immediately save to CSV."""
    
    # Get model indices
    model_indices = eval(row['model_indices']) if isinstance(row['model_indices'], str) else row['model_indices']
    
    # Slice predictions for this combination
    temporal_combo = temporal_stacked[model_indices, :, :, :]
    spatial_combo = spatial_stacked[model_indices, :, :, :]
    
    # Average across models
    temporal_ensemble = np.mean(temporal_combo, axis=0)
    spatial_ensemble = np.mean(spatial_combo, axis=0)
    
    # Time points for evaluation
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Calculate metrics
    temporal_metrics = calculate_all_metrics(
        temporal_labels['event_times'],
        temporal_labels['event_indicators'],
        temporal_ensemble,
        time_points
    )
    
    spatial_metrics = calculate_all_metrics(
        spatial_labels['event_times'],
        spatial_labels['event_indicators'],
        spatial_ensemble,
        time_points
    )
    
    # Create result dictionary
    result = {
        'combination_id': row['combination_id'],
        'n_models': row['n_models'],
        'model_indices': row['model_indices'],
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
    
    # Calculate averages
    temporal_cidx_avg = (result['temporal_cidx_event1'] + result['temporal_cidx_event2']) / 2
    spatial_cidx_avg = (result['spatial_cidx_event1'] + result['spatial_cidx_event2']) / 2
    overall_cidx_avg = (temporal_cidx_avg + spatial_cidx_avg) / 2
    
    result['temporal_cidx_avg'] = temporal_cidx_avg
    result['spatial_cidx_avg'] = spatial_cidx_avg
    result['overall_cidx_avg'] = overall_cidx_avg
    
    return result


def main():
    """Main function to fill metrics with immediate saving."""
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_filled.csv'
    checkpoint_file = 'results/ensemble_checkpoints/fill_progress.txt'
    
    print("="*80)
    print("ENSEMBLE EVALUATION - FILLING MISSING METRICS")
    print("="*80)
    print(f"Input file: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    # Load the CSV
    print("\nLoading evaluation results CSV...")
    df = pd.read_csv(results_file)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    
    # Check for existing progress
    start_row = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_row = int(f.read().strip())
        print(f"Resuming from row {start_row:,}")
    
    # Load stacked predictions
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    print(f"✓ Temporal stacked loaded: {temporal_stacked.shape}")
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    print(f"✓ Spatial stacked loaded: {spatial_stacked.shape}")
    
    # Load labels
    print("\nLoading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    print(f"✓ Temporal labels loaded: {len(temporal_labels['event_times'])} samples")
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    print(f"✓ Spatial labels loaded: {len(spatial_labels['event_times'])} samples")
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    # Initialize output file if starting fresh
    if start_row == 0:
        # Write header
        with open(output_file, 'w') as f:
            f.write(','.join(['combination_id', 'n_models', 'model_indices',
                            'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                            'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                            'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                            'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                            'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    # Process rows
    start_time = time.time()
    last_log_time = start_time
    
    for idx in range(start_row, total_rows):
        row = df.iloc[idx]
        
        try:
            # Process row
            result = process_and_save_row(
                idx, row, temporal_stacked, spatial_stacked,
                temporal_labels, spatial_labels, results_file
            )
            
            # Write result immediately
            with open(output_file, 'a') as f:
                f.write(f"{result['combination_id']},{result['n_models']},\"{result['model_indices']}\","
                       f"{result['temporal_ibs']},{result['temporal_ibs_event1']},{result['temporal_ibs_event2']},"
                       f"{result['temporal_cidx_event1']},{result['temporal_cidx_event2']},{result['temporal_nll']},"
                       f"{result['spatial_ibs']},{result['spatial_ibs_event1']},{result['spatial_ibs_event2']},"
                       f"{result['spatial_cidx_event1']},{result['spatial_cidx_event2']},{result['spatial_nll']},"
                       f"{result['temporal_cidx_avg']},{result['spatial_cidx_avg']},{result['overall_cidx_avg']}\n")
            
            # Update checkpoint
            with open(checkpoint_file, 'w') as f:
                f.write(str(idx + 1))
            
            # Log progress every 100 rows or every 10 seconds
            current_time = time.time()
            if (idx + 1) % 100 == 0 or (current_time - last_log_time) > 10:
                elapsed = current_time - start_time
                processed = idx - start_row + 1
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total_rows - idx - 1
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Row {idx+1:,}/{total_rows:,} ({(idx+1)/total_rows*100:.2f}%) | "
                      f"Rate: {rate:.1f} rows/sec | "
                      f"ETA: {eta_str} | "
                      f"C-idx: {result['overall_cidx_avg']:.4f}")
                
                last_log_time = current_time
                
        except Exception as e:
            print(f"\n❌ ERROR at row {idx}: {e}")
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {total_rows - start_row:,}")
    print(f"Average rate: {(total_rows - start_row)/total_time:.1f} rows/second")
    print(f"Output saved to: {output_file}")
    
    # Find and display top results
    print("\nLoading results to find best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    print("-"*80)
    top_10 = results_df_sorted.head(10)
    for i, row in top_10.iterrows():
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Indices: {row['model_indices']}")
    
    # Save top 100
    top_100_file = 'results/ensemble_checkpoints/top_100_results_final.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()