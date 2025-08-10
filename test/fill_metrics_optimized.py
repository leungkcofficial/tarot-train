"""
Highly optimized version using vectorized operations and numba JIT compilation.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import time
from datetime import datetime, timedelta
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')


@jit(nopython=True)
def fast_concordance_index(event_times, event_indicators, predictions, event_of_interest):
    """Fast C-index calculation using numba."""
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        if event_indicators[i] != event_of_interest:
            continue
            
        for j in range(n):
            if i == j:
                continue
                
            if event_times[i] < event_times[j]:
                if predictions[i] > predictions[j]:
                    concordant += 1
                elif predictions[i] < predictions[j]:
                    discordant += 1
                else:
                    tied_risk += 1
            elif event_times[i] == event_times[j] and event_indicators[j] == event_of_interest:
                tied_risk += 1
    
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied_risk) / total


@jit(nopython=True)
def fast_brier_score(event_times, event_indicators, predictions, time_point, event_of_interest):
    """Fast Brier score calculation using numba."""
    n = len(event_times)
    score = 0.0
    count = 0
    
    for i in range(n):
        # Only consider samples at risk at time_point
        if event_times[i] >= time_point:
            observed = 0.0
            if event_times[i] <= time_point and event_indicators[i] == event_of_interest:
                observed = 1.0
            
            score += (predictions[i] - observed) ** 2
            count += 1
    
    if count == 0:
        return 0.0
    
    return score / count


def calculate_metrics_fast(event_times, event_indicators, cif_array, time_points):
    """Calculate metrics using optimized functions."""
    n_events, n_times, n_samples = cif_array.shape
    
    # Use middle time point for C-index (more stable)
    mid_idx = n_times // 2
    
    # Calculate C-index for each event
    cidx_event1 = fast_concordance_index(
        event_times, event_indicators, 
        cif_array[0, mid_idx, :], 1
    )
    cidx_event2 = fast_concordance_index(
        event_times, event_indicators, 
        cif_array[1, mid_idx, :], 2
    )
    
    # Calculate IBS (simplified - just use middle time point)
    ibs_event1 = fast_brier_score(
        event_times, event_indicators,
        cif_array[0, mid_idx, :], time_points[mid_idx], 1
    )
    ibs_event2 = fast_brier_score(
        event_times, event_indicators,
        cif_array[1, mid_idx, :], time_points[mid_idx], 2
    )
    
    ibs = (ibs_event1 + ibs_event2) / 2
    
    # Simplified NLL (just use mean of log probabilities)
    eps = 1e-7
    nll = -np.mean(np.log(np.clip(cif_array, eps, 1-eps)))
    
    return {
        'ibs': ibs,
        'ibs_event1': ibs_event1,
        'ibs_event2': ibs_event2,
        'cidx_event1': cidx_event1,
        'cidx_event2': cidx_event2,
        'nll': nll
    }


def process_batch_vectorized(batch_indices, temporal_stacked, spatial_stacked,
                           temporal_labels, spatial_labels, time_points):
    """Process a batch of combinations using vectorized operations."""
    results = []
    
    for indices in batch_indices:
        # Get ensemble predictions by averaging
        temporal_ensemble = np.mean(temporal_stacked[indices, :, :, :], axis=0)
        spatial_ensemble = np.mean(spatial_stacked[indices, :, :, :], axis=0)
        
        # Calculate metrics
        temporal_metrics = calculate_metrics_fast(
            temporal_labels['event_times'],
            temporal_labels['event_indicators'],
            temporal_ensemble,
            time_points
        )
        
        spatial_metrics = calculate_metrics_fast(
            spatial_labels['event_times'],
            spatial_labels['event_indicators'],
            spatial_ensemble,
            time_points
        )
        
        results.append({
            'temporal_metrics': temporal_metrics,
            'spatial_metrics': spatial_metrics
        })
    
    return results


def main():
    """Main function with optimized processing."""
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_optimized.csv'
    
    print("="*80)
    print("OPTIMIZED ENSEMBLE EVALUATION")
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
    
    # Check existing progress
    start_row = 0
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        start_row = len(existing_df)
        print(f"Found {start_row:,} existing rows, resuming...")
    
    # Load stacked predictions
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    
    print(f"✓ Loaded stacked predictions")
    
    # Load labels
    print("Loading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    print("✓ Loaded labels")
    
    # Time points
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Initialize output file
    if start_row == 0:
        with open(output_file, 'w') as f:
            f.write(','.join(['combination_id', 'n_models', 'model_indices',
                            'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                            'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                            'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                            'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                            'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    print("\n" + "="*80)
    print("STARTING OPTIMIZED EVALUATION")
    print("="*80)
    
    start_time = time.time()
    batch_size = 1000
    
    # Process in batches
    with open(output_file, 'a') as f:
        for batch_start in range(start_row, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_time_start = time.time()
            
            # Prepare batch
            batch_indices = []
            batch_rows = []
            
            for idx in range(batch_start, batch_end):
                row = df.iloc[idx]
                indices = eval(row['model_indices']) if isinstance(row['model_indices'], str) else row['model_indices']
                batch_indices.append(indices)
                batch_rows.append(row)
            
            # Process batch
            results = process_batch_vectorized(
                batch_indices, temporal_stacked, spatial_stacked,
                temporal_labels, spatial_labels, time_points
            )
            
            # Write results
            for i, (row, result) in enumerate(zip(batch_rows, results)):
                t_metrics = result['temporal_metrics']
                s_metrics = result['spatial_metrics']
                
                # Calculate averages
                t_cidx_avg = (t_metrics['cidx_event1'] + t_metrics['cidx_event2']) / 2
                s_cidx_avg = (s_metrics['cidx_event1'] + s_metrics['cidx_event2']) / 2
                overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
                
                # Write row
                f.write(f"{row['combination_id']},{row['n_models']},\"{row['model_indices']}\","
                       f"{t_metrics['ibs']},{t_metrics['ibs_event1']},{t_metrics['ibs_event2']},"
                       f"{t_metrics['cidx_event1']},{t_metrics['cidx_event2']},{t_metrics['nll']},"
                       f"{s_metrics['ibs']},{s_metrics['ibs_event1']},{s_metrics['ibs_event2']},"
                       f"{s_metrics['cidx_event1']},{s_metrics['cidx_event2']},{s_metrics['nll']},"
                       f"{t_cidx_avg},{s_cidx_avg},{overall_cidx_avg}\n")
            
            # Progress update
            batch_time = time.time() - batch_time_start
            elapsed = time.time() - start_time
            processed = batch_end - start_row
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_rows - batch_end
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Batch {batch_start//batch_size + 1}: {batch_end:,}/{total_rows:,} "
                  f"({batch_end/total_rows*100:.2f}%) | "
                  f"Batch time: {batch_time:.1f}s | "
                  f"Rate: {rate:.1f} rows/sec | "
                  f"ETA: {eta_str}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {total_rows - start_row:,}")
    print(f"Average rate: {(total_rows - start_row)/total_time:.1f} rows/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    print("-"*80)
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f}")
    
    # Save top 100
    top_100_file = 'results/ensemble_checkpoints/top_100_optimized.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")


if __name__ == "__main__":
    main()