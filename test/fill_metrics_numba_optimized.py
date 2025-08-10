"""
Optimized ensemble evaluation using fixed Numba JIT acceleration.
Combines best practices for maximum performance.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import time
from datetime import datetime, timedelta
import numba
from numba import jit, prange
from src.evaluation_metrics import calculate_all_metrics


@jit(nopython=True)
def numba_concordance_index(event_times, event_indicators, predictions, event_of_interest):
    """Optimized Numba C-index implementation."""
    n = len(event_times)
    
    # Pre-filter indices
    event_mask = event_indicators == event_of_interest
    event_indices = np.where(event_mask)[0]
    n_events = len(event_indices)
    
    if n_events == 0:
        return 0.5
    
    concordant = 0
    discordant = 0
    tied = 0
    
    # Process in chunks for better cache locality
    chunk_size = 100
    
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        
        for idx in range(chunk_start, chunk_end):
            i = event_indices[idx]
            time_i = event_times[i]
            pred_i = predictions[i]
            
            for j in range(n):
                if time_i < event_times[j]:
                    pred_diff = pred_i - predictions[j]
                    if pred_diff > 0:
                        concordant += 1
                    elif pred_diff < 0:
                        discordant += 1
                    else:
                        tied += 1
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


@jit(nopython=True)
def numba_brier_score(event_times, event_indicators, predictions, time_point, event_of_interest):
    """Numba-accelerated Brier score calculation."""
    # Samples at risk
    at_risk = event_times >= time_point
    
    # Observed events
    observed = ((event_times <= time_point) & (event_indicators == event_of_interest))
    
    if np.sum(at_risk) == 0:
        return 0.0
    
    # Calculate score for at-risk samples
    score = 0.0
    count = 0
    
    for i in range(len(event_times)):
        if at_risk[i]:
            # Convert boolean to float (1.0 or 0.0)
            obs_value = 1.0 if observed[i] else 0.0
            diff = predictions[i] - obs_value
            score += diff * diff
            count += 1
    
    return score / count if count > 0 else 0.0


def calculate_metrics_numba(event_times, event_indicators, cif_array, time_points):
    """Calculate metrics using Numba-accelerated functions."""
    n_events, n_times, n_samples = cif_array.shape
    mid_idx = n_times // 2
    
    # Calculate C-index
    cidx_event1 = numba_concordance_index(
        event_times, event_indicators,
        cif_array[0, mid_idx, :], 1
    )
    cidx_event2 = numba_concordance_index(
        event_times, event_indicators,
        cif_array[1, mid_idx, :], 2
    )
    
    # Calculate IBS
    ibs_event1 = numba_brier_score(
        event_times, event_indicators,
        cif_array[0, mid_idx, :], time_points[mid_idx], 1
    )
    ibs_event2 = numba_brier_score(
        event_times, event_indicators,
        cif_array[1, mid_idx, :], time_points[mid_idx], 2
    )
    
    ibs = (ibs_event1 + ibs_event2) / 2
    
    # NLL (simplified)
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


def process_batch(batch_df, temporal_stacked, spatial_stacked,
                 temporal_labels, spatial_labels, time_points):
    """Process a batch of combinations using Numba acceleration."""
    results = []
    
    for idx, row in batch_df.iterrows():
        # Get model indices
        model_indices = eval(row['model_indices']) if isinstance(row['model_indices'], str) else row['model_indices']
        
        # Create ensemble
        temporal_ensemble = np.mean(temporal_stacked[model_indices], axis=0)
        spatial_ensemble = np.mean(spatial_stacked[model_indices], axis=0)
        
        # Calculate metrics with Numba
        temporal_metrics = calculate_metrics_numba(
            temporal_labels['event_times'],
            temporal_labels['event_indicators'],
            temporal_ensemble,
            time_points
        )
        
        spatial_metrics = calculate_metrics_numba(
            spatial_labels['event_times'],
            spatial_labels['event_indicators'],
            spatial_ensemble,
            time_points
        )
        
        # Calculate averages
        t_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
        s_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
        overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
        
        results.append({
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
            'spatial_nll': spatial_metrics['nll'],
            'temporal_cidx_avg': t_cidx_avg,
            'spatial_cidx_avg': s_cidx_avg,
            'overall_cidx_avg': overall_cidx_avg
        })
    
    return results


def main():
    """Main function with Numba optimization."""
    
    # Set optimal number of threads
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    print(f"Using {numba.config.NUMBA_NUM_THREADS} Numba threads")
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_numba.csv'
    batch_size = 1000
    
    print("\n" + "="*80)
    print("NUMBA-OPTIMIZED ENSEMBLE EVALUATION")
    print("="*80)
    print(f"Input file: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size:,}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    # Load data
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    
    print("Loading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Warm up Numba
    print("\nWarming up Numba JIT...")
    test_ensemble = np.mean(temporal_stacked[:2], axis=0)
    _ = calculate_metrics_numba(
        temporal_labels['event_times'][:100],
        temporal_labels['event_indicators'][:100],
        test_ensemble[:, :, :100],
        time_points
    )
    
    # Get total rows
    print("\nCounting total rows...")
    total_rows = sum(1 for line in open(results_file)) - 1
    print(f"Total rows: {total_rows:,}")
    
    # Check existing progress
    start_row = 0
    if os.path.exists(output_file):
        start_row = sum(1 for line in open(output_file)) - 1
        print(f"Found existing output with {start_row:,} rows, resuming...")
    
    # Initialize output
    if start_row == 0:
        columns = ['combination_id', 'n_models', 'model_indices',
                  'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                  'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                  'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                  'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                  'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("STARTING NUMBA-OPTIMIZED EVALUATION")
    print("="*80)
    
    start_time = time.time()
    processed = 0
    
    # Process in batches
    for chunk_idx, chunk in enumerate(pd.read_csv(results_file, chunksize=batch_size)):
        # Skip processed chunks
        if chunk.index[-1] < start_row:
            continue
        
        # Filter to unprocessed rows
        chunk_to_process = chunk[chunk.index >= start_row]
        
        if len(chunk_to_process) == 0:
            continue
        
        # Debug: print when starting a new batch
        if processed == 0:
            print(f"\nProcessing first batch (chunk {chunk_idx})...", flush=True)
        
        # Process batch
        batch_results = process_batch(
            chunk_to_process, temporal_stacked, spatial_stacked,
            temporal_labels, spatial_labels, time_points
        )</search>
</search_and_replace>
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(batch_results)
        results_df.to_csv(output_file, mode='a', header=False, index=False)
        
        processed += len(batch_results)
        
        # Progress update
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total_rows - start_row - processed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Processed {processed + start_row:,}/{total_rows:,} "
              f"({(processed + start_row)/total_rows*100:.2f}%) | "
              f"Rate: {rate:.1f} rows/sec | "
              f"ETA: {eta_str}", flush=True)
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {processed:,}")
    print(f"Average rate: {processed/total_time:.1f} rows/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    print("-" * 80)
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Temporal: {row['temporal_cidx_avg']:.4f} | "
              f"Spatial: {row['spatial_cidx_avg']:.4f}")
    
    # Save top combinations
    top_100 = results_df_sorted.head(100)
    top_100.to_csv('results/ensemble_checkpoints/top_100_combinations_numba.csv', index=False)
    print(f"\nSaved top 100 combinations to: results/ensemble_checkpoints/top_100_combinations_numba.csv")


if __name__ == "__main__":
    main()