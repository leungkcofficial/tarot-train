"""
Highly optimized GPU-accelerated ensemble evaluation using CuPy.
Minimizes CPU-GPU transfers and processes large batches on GPU.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import time
from datetime import datetime, timedelta
import cupy as cp
from cupyx.scipy import ndimage


def gpu_concordance_index_batch(event_times_gpu, event_indicators_gpu, predictions_batch_gpu, event_of_interest):
    """
    GPU-accelerated C-index calculation for a batch of predictions.
    
    Args:
        event_times_gpu: (n_samples,) array
        event_indicators_gpu: (n_samples,) array
        predictions_batch_gpu: (batch_size, n_samples) array
        event_of_interest: int
    
    Returns:
        (batch_size,) array of C-index values
    """
    batch_size, n_samples = predictions_batch_gpu.shape
    
    # Create masks
    event_mask = (event_indicators_gpu == event_of_interest)
    
    # Expand dimensions for broadcasting
    times_i = event_times_gpu[None, :, None]  # (1, n, 1)
    times_j = event_times_gpu[None, None, :]  # (1, 1, n)
    
    # Time comparisons
    time_diff = times_i - times_j  # (1, n, n)
    valid_pairs = (time_diff < 0) & event_mask[None, :, None]  # (1, n, n)
    
    # Batch predictions comparison
    preds_i = predictions_batch_gpu[:, :, None]  # (batch, n, 1)
    preds_j = predictions_batch_gpu[:, None, :]  # (batch, 1, n)
    pred_diff = preds_i - preds_j  # (batch, n, n)
    
    # Count concordant, discordant, tied for each batch
    concordant = cp.sum((pred_diff > 0) & valid_pairs, axis=(1, 2))
    discordant = cp.sum((pred_diff < 0) & valid_pairs, axis=(1, 2))
    tied = cp.sum((pred_diff == 0) & valid_pairs, axis=(1, 2))
    
    total = concordant + discordant + tied
    
    # Avoid division by zero
    c_index = cp.where(total > 0, 
                      (concordant + 0.5 * tied) / total,
                      0.5)
    
    return c_index


def gpu_brier_score_batch(event_times_gpu, event_indicators_gpu, predictions_batch_gpu, 
                         time_point, event_of_interest):
    """
    GPU-accelerated Brier score calculation for a batch of predictions.
    
    Args:
        event_times_gpu: (n_samples,) array
        event_indicators_gpu: (n_samples,) array  
        predictions_batch_gpu: (batch_size, n_samples) array
        time_point: float
        event_of_interest: int
        
    Returns:
        (batch_size,) array of Brier scores
    """
    # Samples at risk
    at_risk = event_times_gpu >= time_point
    
    # Observed events
    observed = ((event_times_gpu <= time_point) & 
                (event_indicators_gpu == event_of_interest)).astype(cp.float32)
    
    if cp.sum(at_risk) == 0:
        return cp.zeros(predictions_batch_gpu.shape[0])
    
    # Calculate squared differences for at-risk samples
    diff_squared = (predictions_batch_gpu[:, at_risk] - observed[at_risk]) ** 2
    brier_scores = cp.mean(diff_squared, axis=1)
    
    return brier_scores


def calculate_metrics_gpu_batch(event_times_gpu, event_indicators_gpu, cif_batch_gpu, time_points_gpu):
    """
    Calculate all metrics for a batch of CIF arrays on GPU.
    
    Args:
        event_times_gpu: (n_samples,) array
        event_indicators_gpu: (n_samples,) array
        cif_batch_gpu: (batch_size, n_events, n_times, n_samples) array
        time_points_gpu: (n_times,) array
        
    Returns:
        Dictionary of metric arrays, each of shape (batch_size,)
    """
    batch_size, n_events, n_times, n_samples = cif_batch_gpu.shape
    mid_idx = n_times // 2
    
    # Extract predictions at mid time point
    pred_event1_mid = cif_batch_gpu[:, 0, mid_idx, :]  # (batch, n_samples)
    pred_event2_mid = cif_batch_gpu[:, 1, mid_idx, :]  # (batch, n_samples)
    
    # Calculate C-indices
    cidx_event1 = gpu_concordance_index_batch(
        event_times_gpu, event_indicators_gpu, pred_event1_mid, 1
    )
    cidx_event2 = gpu_concordance_index_batch(
        event_times_gpu, event_indicators_gpu, pred_event2_mid, 2
    )
    
    # Calculate Brier scores
    ibs_event1 = gpu_brier_score_batch(
        event_times_gpu, event_indicators_gpu, pred_event1_mid,
        time_points_gpu[mid_idx], 1
    )
    ibs_event2 = gpu_brier_score_batch(
        event_times_gpu, event_indicators_gpu, pred_event2_mid,
        time_points_gpu[mid_idx], 2
    )
    
    ibs = (ibs_event1 + ibs_event2) / 2
    
    # NLL (simplified) - average across all predictions
    eps = 1e-7
    log_probs = cp.log(cp.clip(cif_batch_gpu, eps, 1-eps))
    nll = -cp.mean(log_probs, axis=(1, 2, 3))
    
    return {
        'cidx_event1': cidx_event1,
        'cidx_event2': cidx_event2,
        'ibs_event1': ibs_event1,
        'ibs_event2': ibs_event2,
        'ibs': ibs,
        'nll': nll
    }


def process_mega_batch_gpu(combinations_df, start_idx, end_idx,
                          temporal_stacked_gpu, spatial_stacked_gpu,
                          temporal_times_gpu, temporal_indicators_gpu,
                          spatial_times_gpu, spatial_indicators_gpu,
                          time_points_gpu):
    """
    Process a large batch of combinations entirely on GPU.
    """
    batch_df = combinations_df.iloc[start_idx:end_idx]
    batch_size = len(batch_df)
    
    # Pre-allocate GPU arrays for ensemble CIFs
    n_events, n_times, n_samples_temporal = temporal_stacked_gpu.shape[1:]
    n_samples_spatial = spatial_stacked_gpu.shape[3]
    
    temporal_ensembles = cp.zeros((batch_size, n_events, n_times, n_samples_temporal), dtype=cp.float32)
    spatial_ensembles = cp.zeros((batch_size, n_events, n_times, n_samples_spatial), dtype=cp.float32)
    
    # Process each combination
    for i, (idx, row) in enumerate(batch_df.iterrows()):
        # Parse model indices
        model_indices = eval(row['model_indices']) if isinstance(row['model_indices'], str) else row['model_indices']
        model_indices_gpu = cp.array(model_indices)
        
        # Create ensemble by averaging selected models
        temporal_ensembles[i] = cp.mean(temporal_stacked_gpu[model_indices_gpu], axis=0)
        spatial_ensembles[i] = cp.mean(spatial_stacked_gpu[model_indices_gpu], axis=0)
    
    # Calculate all metrics in batch
    temporal_metrics = calculate_metrics_gpu_batch(
        temporal_times_gpu, temporal_indicators_gpu,
        temporal_ensembles, time_points_gpu
    )
    
    spatial_metrics = calculate_metrics_gpu_batch(
        spatial_times_gpu, spatial_indicators_gpu,
        spatial_ensembles, time_points_gpu
    )
    
    # Calculate averages
    t_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
    s_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
    overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
    
    # Transfer results back to CPU
    results = {
        'temporal_cidx_event1': cp.asnumpy(temporal_metrics['cidx_event1']),
        'temporal_cidx_event2': cp.asnumpy(temporal_metrics['cidx_event2']),
        'temporal_ibs_event1': cp.asnumpy(temporal_metrics['ibs_event1']),
        'temporal_ibs_event2': cp.asnumpy(temporal_metrics['ibs_event2']),
        'temporal_ibs': cp.asnumpy(temporal_metrics['ibs']),
        'temporal_nll': cp.asnumpy(temporal_metrics['nll']),
        'spatial_cidx_event1': cp.asnumpy(spatial_metrics['cidx_event1']),
        'spatial_cidx_event2': cp.asnumpy(spatial_metrics['cidx_event2']),
        'spatial_ibs_event1': cp.asnumpy(spatial_metrics['ibs_event1']),
        'spatial_ibs_event2': cp.asnumpy(spatial_metrics['ibs_event2']),
        'spatial_ibs': cp.asnumpy(spatial_metrics['ibs']),
        'spatial_nll': cp.asnumpy(spatial_metrics['nll']),
        'temporal_cidx_avg': cp.asnumpy(t_cidx_avg),
        'spatial_cidx_avg': cp.asnumpy(s_cidx_avg),
        'overall_cidx_avg': cp.asnumpy(overall_cidx_avg)
    }
    
    return results


def main():
    """Main function with optimized GPU acceleration."""
    
    # GPU info
    print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"GPU Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
    
    # Configuration
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_gpu_optimized.csv'
    mega_batch_size = 10000  # Process 10K combinations at once on GPU
    
    print("\n" + "="*80)
    print("OPTIMIZED GPU-ACCELERATED ENSEMBLE EVALUATION")
    print("="*80)
    print(f"Input file: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Mega batch size: {mega_batch_size:,}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    # Load data and transfer to GPU once
    print("\nLoading and transferring data to GPU...")
    
    # Load stacked predictions
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked_gpu = cp.asarray(f['stacked_cif'][:], dtype=cp.float32)
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked_gpu = cp.asarray(f['stacked_cif'][:], dtype=cp.float32)
    
    print(f"Temporal stacked shape: {temporal_stacked_gpu.shape}")
    print(f"Spatial stacked shape: {spatial_stacked_gpu.shape}")
    
    # Load labels and transfer to GPU
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    temporal_times_gpu = cp.asarray(temporal_labels['event_times'])
    temporal_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    spatial_times_gpu = cp.asarray(spatial_labels['event_times'])
    spatial_indicators_gpu = cp.asarray(spatial_labels['event_indicators'])
    
    time_points_gpu = cp.asarray([365, 730, 1095, 1460, 1825])
    
    # Load combinations DataFrame
    print("\nLoading combinations...")
    combinations_df = pd.read_csv(results_file)
    total_rows = len(combinations_df)
    print(f"Total combinations: {total_rows:,}")
    
    # Check existing progress
    start_row = 0
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        start_row = len(existing_df)
        print(f"Found existing output with {start_row:,} rows, resuming...")
        del existing_df
    
    # Initialize output file
    if start_row == 0:
        columns = ['combination_id', 'n_models', 'model_indices',
                  'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                  'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                  'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                  'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                  'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("STARTING OPTIMIZED GPU EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    # Process in mega batches
    for batch_start in range(start_row, total_rows, mega_batch_size):
        batch_end = min(batch_start + mega_batch_size, total_rows)
        batch_size = batch_end - batch_start
        
        # Process mega batch on GPU
        batch_results = process_mega_batch_gpu(
            combinations_df, batch_start, batch_end,
            temporal_stacked_gpu, spatial_stacked_gpu,
            temporal_times_gpu, temporal_indicators_gpu,
            spatial_times_gpu, spatial_indicators_gpu,
            time_points_gpu
        )
        
        # Create results DataFrame
        batch_df = combinations_df.iloc[batch_start:batch_end].copy()
        
        # Add metric columns
        for metric_name, metric_values in batch_results.items():
            batch_df[metric_name] = metric_values
        
        # Append to output file
        batch_df.to_csv(output_file, mode='a', header=False, index=False)
        
        # Progress update
        processed = batch_end
        elapsed = time.time() - start_time
        rate = (processed - start_row) / elapsed if elapsed > 0 else 0
        remaining = total_rows - processed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # GPU memory usage
        gpu_mem_used = cp.cuda.runtime.memGetInfo()[0] / 1e9
        gpu_mem_total = cp.cuda.runtime.memGetInfo()[1] / 1e9
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Processed {processed:,}/{total_rows:,} "
              f"({processed/total_rows*100:.2f}%) | "
              f"Rate: {rate:.1f} rows/sec | "
              f"GPU Mem: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB | "
              f"ETA: {eta_str}")
    
    # Summary
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
    print("-" * 80)
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Temporal: {row['temporal_cidx_avg']:.4f} | "
              f"Spatial: {row['spatial_cidx_avg']:.4f}")
        
    # Save top combinations
    top_100 = results_df_sorted.head(100)
    top_100.to_csv('results/ensemble_checkpoints/top_100_combinations_gpu.csv', index=False)
    print(f"\nSaved top 100 combinations to: results/ensemble_checkpoints/top_100_combinations_gpu.csv")


if __name__ == "__main__":
    main()