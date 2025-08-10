"""
Chunked processing to avoid loading entire 16.7M row CSV into memory.
Processes the CSV in manageable chunks.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue


# Global variables for shared data
temporal_stacked_global = None
spatial_stacked_global = None
temporal_labels_global = None
spatial_labels_global = None
time_points_global = None

# Thread-safe queue and lock
result_queue = Queue()
write_lock = threading.Lock()
progress_lock = threading.Lock()
total_processed = 0


def init_shared_data():
    """Initialize global shared data."""
    global temporal_stacked_global, spatial_stacked_global
    global temporal_labels_global, spatial_labels_global, time_points_global
    
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked_global = f['stacked_cif'][:]
    print(f"✓ Temporal stacked loaded: {temporal_stacked_global.shape}")
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked_global = f['stacked_cif'][:]
    print(f"✓ Spatial stacked loaded: {spatial_stacked_global.shape}")
    
    print("\nLoading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels_global = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels_global = pickle.load(f)
    
    time_points_global = np.array([365, 730, 1095, 1460, 1825])
    
    print("✓ Shared data initialized")


def evaluate_row(row):
    """Evaluate a single row."""
    try:
        # Parse model indices
        model_indices = eval(row['model_indices']) if isinstance(row['model_indices'], str) else row['model_indices']
        
        # Slice predictions
        temporal_combo = temporal_stacked_global[model_indices, :, :, :]
        spatial_combo = spatial_stacked_global[model_indices, :, :, :]
        
        # Average across models
        temporal_ensemble = np.mean(temporal_combo, axis=0)
        spatial_ensemble = np.mean(spatial_combo, axis=0)
        
        # Calculate metrics
        temporal_metrics = calculate_all_metrics(
            temporal_labels_global['event_times'],
            temporal_labels_global['event_indicators'],
            temporal_ensemble,
            time_points_global
        )
        
        spatial_metrics = calculate_all_metrics(
            spatial_labels_global['event_times'],
            spatial_labels_global['event_indicators'],
            spatial_ensemble,
            time_points_global
        )
        
        # Calculate averages
        t_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
        s_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
        overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
        
        return {
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
        }
        
    except Exception as e:
        print(f"Error processing combination {row['combination_id']}: {e}")
        return None


def process_chunk(chunk, output_file, start_time, total_rows):
    """Process a chunk of rows using threads."""
    global total_processed
    
    n_threads = min(os.cpu_count(), 24)
    chunk_results = []
    
    # Process chunk with threads
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(evaluate_row, row) for _, row in chunk.iterrows()]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                chunk_results.append(result)
    
    # Write results
    if chunk_results:
        with write_lock:
            with open(output_file, 'a') as f:
                for result in chunk_results:
                    f.write(f"{result['combination_id']},{result['n_models']},\"{result['model_indices']}\","
                           f"{result['temporal_ibs']},{result['temporal_ibs_event1']},{result['temporal_ibs_event2']},"
                           f"{result['temporal_cidx_event1']},{result['temporal_cidx_event2']},{result['temporal_nll']},"
                           f"{result['spatial_ibs']},{result['spatial_ibs_event1']},{result['spatial_ibs_event2']},"
                           f"{result['spatial_cidx_event1']},{result['spatial_cidx_event2']},{result['spatial_nll']},"
                           f"{result['temporal_cidx_avg']},{result['spatial_cidx_avg']},{result['overall_cidx_avg']}\n")
    
    # Update progress
    with progress_lock:
        total_processed += len(chunk_results)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = total_rows - total_processed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Processed {total_processed:,}/{total_rows:,} ({total_processed/total_rows*100:.2f}%) | "
              f"Rate: {rate:.1f} rows/sec | "
              f"ETA: {eta_str}")


def main():
    """Main function with chunked processing."""
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_chunked.csv'
    chunk_size = 10000  # Process 10k rows at a time
    
    print("="*80)
    print("CHUNKED ENSEMBLE EVALUATION")
    print("="*80)
    print(f"Input file: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    # Initialize shared data
    init_shared_data()
    
    # Get total rows without loading entire file
    print("\nCounting total rows...")
    total_rows = sum(1 for line in open(results_file)) - 1  # Subtract header
    print(f"Total rows: {total_rows:,}")
    
    # Check for existing progress
    start_row = 0
    if os.path.exists(output_file):
        start_row = sum(1 for line in open(output_file)) - 1  # Subtract header
        print(f"Found existing output with {start_row:,} rows, resuming...")
        global total_processed
        total_processed = start_row
    
    # Initialize output file if needed
    if start_row == 0:
        with open(output_file, 'w') as f:
            f.write(','.join(['combination_id', 'n_models', 'model_indices',
                            'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                            'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                            'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                            'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                            'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    print("\n" + "="*80)
    print("STARTING CHUNKED EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    # Process CSV in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(results_file, chunksize=chunk_size)):
        # Skip already processed chunks
        chunk_start = chunk_num * chunk_size
        chunk_end = chunk_start + len(chunk)
        
        if chunk_end <= start_row:
            continue
        
        # Skip already processed rows within chunk
        if chunk_start < start_row:
            skip_rows = start_row - chunk_start
            chunk = chunk.iloc[skip_rows:]
        
        if len(chunk) == 0:
            continue
        
        print(f"\nProcessing chunk {chunk_num + 1} (rows {chunk_start:,} to {chunk_end:,})...")
        process_chunk(chunk, output_file, start_time, total_rows)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {total_processed:,}")
    print(f"Average rate: {total_processed/total_time:.1f} rows/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 10 BEST COMBINATIONS:")
    print("-"*80)
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Indices: {row['model_indices']}")
    
    # Save top 100
    top_100_file = 'results/ensemble_checkpoints/top_100_chunked.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")


if __name__ == "__main__":
    main()