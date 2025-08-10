"""
GPU-accelerated ensemble evaluation using CuPy.
Processes metrics calculations on GPU for massive speedup.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import time
from datetime import datetime, timedelta

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available!")
except ImportError:
    print("CuPy not installed. Install with: pip install cupy-cuda11x")
    GPU_AVAILABLE = False
    import numpy as cp  # Fallback to CPU


def gpu_concordance_index(event_times_gpu, event_indicators_gpu, predictions_gpu, event_of_interest):
    """GPU-accelerated C-index calculation."""
    n = len(event_times_gpu)
    
    # Create masks for event of interest
    event_mask = (event_indicators_gpu == event_of_interest)
    
    # Vectorized comparison
    time_diff = event_times_gpu[:, None] - event_times_gpu[None, :]
    pred_diff = predictions_gpu[:, None] - predictions_gpu[None, :]
    
    # Count concordant pairs
    concordant_mask = (time_diff < 0) & event_mask[:, None]
    concordant = cp.sum((pred_diff > 0) & concordant_mask)
    discordant = cp.sum((pred_diff < 0) & concordant_mask)
    tied = cp.sum((pred_diff == 0) & concordant_mask)
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return float((concordant + 0.5 * tied) / total)


def gpu_brier_score(event_times_gpu, event_indicators_gpu, predictions_gpu, time_point, event_of_interest):
    """GPU-accelerated Brier score calculation."""
    # Samples at risk
    at_risk = event_times_gpu >= time_point
    
    # Observed events
    observed = ((event_times_gpu <= time_point) & (event_indicators_gpu == event_of_interest)).astype(cp.float32)
    
    # Calculate score
    if cp.sum(at_risk) == 0:
        return 0.0
    
    score = cp.mean((predictions_gpu[at_risk] - observed[at_risk]) ** 2)
    return float(score)


def calculate_metrics_gpu(event_times, event_indicators, cif_array, time_points):
    """Calculate metrics using GPU acceleration."""
    # Transfer to GPU
    event_times_gpu = cp.asarray(event_times)
    event_indicators_gpu = cp.asarray(event_indicators)
    cif_gpu = cp.asarray(cif_array)
    
    n_events, n_times, n_samples = cif_array.shape
    mid_idx = n_times // 2
    
    # Calculate C-index
    cidx_event1 = gpu_concordance_index(
        event_times_gpu, event_indicators_gpu,
        cif_gpu[0, mid_idx, :], 1
    )
    cidx_event2 = gpu_concordance_index(
        event_times_gpu, event_indicators_gpu,
        cif_gpu[1, mid_idx, :], 2
    )
    
    # Calculate IBS
    ibs_event1 = gpu_brier_score(
        event_times_gpu, event_indicators_gpu,
        cif_gpu[0, mid_idx, :], time_points[mid_idx], 1
    )
    ibs_event2 = gpu_brier_score(
        event_times_gpu, event_indicators_gpu,
        cif_gpu[1, mid_idx, :], time_points[mid_idx], 2
    )
    
    ibs = (ibs_event1 + ibs_event2) / 2
    
    # NLL (simplified)
    eps = 1e-7
    nll = -float(cp.mean(cp.log(cp.clip(cif_gpu, eps, 1-eps))))
    
    return {
        'ibs': ibs,
        'ibs_event1': ibs_event1,
        'ibs_event2': ibs_event2,
        'cidx_event1': cidx_event1,
        'cidx_event2': cidx_event2,
        'nll': nll
    }


def process_batch_gpu(batch_data, temporal_stacked, spatial_stacked,
                     temporal_labels, spatial_labels, time_points):
    """Process a batch of combinations on GPU."""
    results = []
    
    # Transfer stacked arrays to GPU once
    temporal_gpu = cp.asarray(temporal_stacked)
    spatial_gpu = cp.asarray(spatial_stacked)
    
    for idx, combination_id, n_models, model_indices_str in batch_data:
        try:
            # Parse indices
            model_indices = eval(model_indices_str) if isinstance(model_indices_str, str) else model_indices_str
            
            # Slice and average on GPU
            temporal_combo = temporal_gpu[model_indices, :, :, :]
            spatial_combo = spatial_gpu[model_indices, :, :, :]
            
            temporal_ensemble = cp.mean(temporal_combo, axis=0)
            spatial_ensemble = cp.mean(spatial_combo, axis=0)
            
            # Transfer back to CPU for metrics (or keep on GPU if metrics support it)
            temporal_ensemble_cpu = cp.asnumpy(temporal_ensemble)
            spatial_ensemble_cpu = cp.asnumpy(spatial_ensemble)
            
            # Calculate metrics
            temporal_metrics = calculate_metrics_gpu(
                temporal_labels['event_times'],
                temporal_labels['event_indicators'],
                temporal_ensemble_cpu,
                time_points
            )
            
            spatial_metrics = calculate_metrics_gpu(
                spatial_labels['event_times'],
                spatial_labels['event_indicators'],
                spatial_ensemble_cpu,
                time_points
            )
            
            # Calculate averages
            t_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
            s_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
            overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
            
            results.append({
                'combination_id': combination_id,
                'n_models': n_models,
                'model_indices': model_indices_str,
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
            
        except Exception as e:
            print(f"Error processing combination {combination_id}: {e}")
            continue
    
    return results


def main():
    """Main function with GPU acceleration."""
    
    if not GPU_AVAILABLE:
        print("GPU not available! Please install CuPy for GPU acceleration.")
        print("Install with: pip install cupy-cuda11x")
        return
    
    # Check GPU
    print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"GPU Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_gpu.csv'
    batch_size = 1000  # Process 1000 at a time on GPU
    
    print("\n" + "="*80)
    print("GPU-ACCELERATED ENSEMBLE EVALUATION")
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
        with open(output_file, 'w') as f:
            f.write(','.join(['combination_id', 'n_models', 'model_indices',
                            'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                            'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                            'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                            'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                            'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    print("\n" + "="*80)
    print("STARTING GPU EVALUATION")
    print("="*80)
    
    start_time = time.time()
    processed = 0
    
    # Process in batches
    for chunk in pd.read_csv(results_file, chunksize=batch_size):
        # Skip processed chunks
        if chunk.index[-1] < start_row:
            continue
        
        # Prepare batch data
        batch_data = []
        for idx, row in chunk.iterrows():
            if idx >= start_row:
                batch_data.append((idx, row['combination_id'], row['n_models'], row['model_indices']))
        
        if not batch_data:
            continue
        
        # Process batch on GPU
        batch_results = process_batch_gpu(
            batch_data, temporal_stacked, spatial_stacked,
            temporal_labels, spatial_labels, time_points
        )
        
        # Write results
        with open(output_file, 'a') as f:
            for result in batch_results:
                f.write(f"{result['combination_id']},{result['n_models']},\"{result['model_indices']}\","
                       f"{result['temporal_ibs']},{result['temporal_ibs_event1']},{result['temporal_ibs_event2']},"
                       f"{result['temporal_cidx_event1']},{result['temporal_cidx_event2']},{result['temporal_nll']},"
                       f"{result['spatial_ibs']},{result['spatial_ibs_event1']},{result['spatial_ibs_event2']},"
                       f"{result['spatial_cidx_event1']},{result['spatial_cidx_event2']},{result['spatial_nll']},"
                       f"{result['temporal_cidx_avg']},{result['spatial_cidx_avg']},{result['overall_cidx_avg']}\n")
        
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
              f"ETA: {eta_str}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {processed:,}")
    print(f"Average rate: {processed/total_time:.1f} rows/second")
    
    # Find best
    print("\nFinding best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f}")


if __name__ == "__main__":
    main()