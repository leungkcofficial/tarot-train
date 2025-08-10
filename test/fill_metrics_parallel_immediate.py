"""
Parallel version that fills missing metrics with immediate row-by-row saving.
Uses multiprocessing to speed up evaluation while maintaining immediate saves.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, Manager, Lock
from functools import partial
import sys
from threading import Thread
import queue


def process_single_row(args):
    """Process a single row and return results."""
    idx, row, temporal_stacked, spatial_stacked, temporal_labels, spatial_labels = args
    
    try:
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
        
        # Calculate averages
        temporal_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
        spatial_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
        overall_cidx_avg = (temporal_cidx_avg + spatial_cidx_avg) / 2
        
        # Return result
        return {
            'idx': idx,
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
            'temporal_cidx_avg': temporal_cidx_avg,
            'spatial_cidx_avg': spatial_cidx_avg,
            'overall_cidx_avg': overall_cidx_avg
        }
    except Exception as e:
        return {'idx': idx, 'error': str(e)}


def result_writer(result_queue, output_file, total_rows, start_time):
    """Thread function to write results as they come in."""
    processed = 0
    last_log_time = time.time()
    
    with open(output_file, 'a') as f:
        while True:
            try:
                result = result_queue.get(timeout=1)
                
                if result == 'DONE':
                    break
                
                if 'error' in result:
                    print(f"\n❌ ERROR at row {result['idx']}: {result['error']}")
                    continue
                
                # Write result immediately
                f.write(f"{result['combination_id']},{result['n_models']},\"{result['model_indices']}\","
                       f"{result['temporal_ibs']},{result['temporal_ibs_event1']},{result['temporal_ibs_event2']},"
                       f"{result['temporal_cidx_event1']},{result['temporal_cidx_event2']},{result['temporal_nll']},"
                       f"{result['spatial_ibs']},{result['spatial_ibs_event1']},{result['spatial_ibs_event2']},"
                       f"{result['spatial_cidx_event1']},{result['spatial_cidx_event2']},{result['spatial_nll']},"
                       f"{result['temporal_cidx_avg']},{result['spatial_cidx_avg']},{result['overall_cidx_avg']}\n")
                f.flush()  # Force write to disk
                
                processed += 1
                
                # Log progress
                current_time = time.time()
                if processed % 100 == 0 or (current_time - last_log_time) > 10:
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = total_rows - processed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Processed {processed:,}/{total_rows:,} ({processed/total_rows*100:.2f}%) | "
                          f"Rate: {rate:.1f} rows/sec | "
                          f"ETA: {eta_str} | "
                          f"Last C-idx: {result['overall_cidx_avg']:.4f}")
                    
                    last_log_time = current_time
                    
            except queue.Empty:
                continue


def main():
    """Main function to fill metrics with parallel processing and immediate saving."""
    
    n_workers = cpu_count()
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_filled.csv'
    checkpoint_file = 'results/ensemble_checkpoints/fill_progress.txt'
    
    print("="*80)
    print("PARALLEL ENSEMBLE EVALUATION - FILLING MISSING METRICS")
    print("="*80)
    print(f"CPU cores available: {n_workers}")
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
    if os.path.exists(checkpoint_file) and os.path.exists(output_file):
        # Count existing rows in output file
        with open(output_file, 'r') as f:
            start_row = sum(1 for line in f) - 1  # Subtract header
        print(f"Found existing output with {start_row:,} rows, resuming...")
    
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
    print(f"STARTING PARALLEL EVALUATION WITH {n_workers} WORKERS")
    print("="*80)
    
    # Initialize output file if starting fresh
    if start_row == 0:
        with open(output_file, 'w') as f:
            f.write(','.join(['combination_id', 'n_models', 'model_indices',
                            'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                            'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                            'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                            'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                            'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    # Prepare data for parallel processing
    rows_to_process = []
    for idx in range(start_row, total_rows):
        row = df.iloc[idx]
        rows_to_process.append((idx, row, temporal_stacked, spatial_stacked, 
                               temporal_labels, spatial_labels))
    
    # Create result queue and start writer thread
    manager = Manager()
    result_queue = manager.Queue()
    
    start_time = time.time()
    
    # Start writer thread
    writer_thread = Thread(target=result_writer, 
                          args=(result_queue, output_file, len(rows_to_process), start_time))
    writer_thread.start()
    
    # Process in parallel
    print(f"\nProcessing {len(rows_to_process):,} rows in parallel...")
    
    with Pool(n_workers) as pool:
        # Use imap_unordered for better performance
        for result in pool.imap_unordered(process_single_row, rows_to_process, chunksize=10):
            if result:
                result_queue.put(result)
    
    # Signal writer thread to stop
    result_queue.put('DONE')
    writer_thread.join()
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {len(rows_to_process):,}")
    print(f"Average rate: {len(rows_to_process)/total_time:.1f} rows/second")
    print(f"Output saved to: {output_file}")
    
    # Find and display top results
    print("\nLoading results to find best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    print("-"*80)
    top_10 = results_df_sorted.head(10)
    for i, (_, row) in enumerate(top_10.iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Temporal: {row['temporal_cidx_avg']:.4f} | "
              f"Spatial: {row['spatial_cidx_avg']:.4f}")
    
    # Save top 100
    top_100_file = 'results/ensemble_checkpoints/top_100_results_final.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()