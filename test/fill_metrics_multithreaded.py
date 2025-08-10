"""
Multithreaded evaluation using shared memory.
Threads share the same memory space, avoiding the copying overhead of multiprocessing.
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
import gc


# Global variables for shared data
temporal_stacked_global = None
spatial_stacked_global = None
temporal_labels_global = None
spatial_labels_global = None
time_points_global = None

# Thread-safe queue for results
result_queue = Queue()
write_lock = threading.Lock()


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


def evaluate_combination_thread(row_data):
    """Evaluate a single combination in a thread."""
    idx, combination_id, n_models, model_indices_str = row_data
    
    try:
        # Parse model indices
        model_indices = eval(model_indices_str) if isinstance(model_indices_str, str) else model_indices_str
        
        # Slice predictions (using global shared arrays)
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
            'idx': idx,
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
        }
        
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None


def result_writer(output_file, total_rows):
    """Thread function to write results as they come in."""
    processed = 0
    start_time = time.time()
    last_log_time = start_time
    
    with open(output_file, 'a') as f:
        while processed < total_rows:
            try:
                result = result_queue.get(timeout=1)
                
                if result is None:
                    continue
                
                # Write result
                f.write(f"{result['combination_id']},{result['n_models']},\"{result['model_indices']}\","
                       f"{result['temporal_ibs']},{result['temporal_ibs_event1']},{result['temporal_ibs_event2']},"
                       f"{result['temporal_cidx_event1']},{result['temporal_cidx_event2']},{result['temporal_nll']},"
                       f"{result['spatial_ibs']},{result['spatial_ibs_event1']},{result['spatial_ibs_event2']},"
                       f"{result['spatial_cidx_event1']},{result['spatial_cidx_event2']},{result['spatial_nll']},"
                       f"{result['temporal_cidx_avg']},{result['spatial_cidx_avg']},{result['overall_cidx_avg']}\n")
                f.flush()
                
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
                    
            except:
                continue


def main():
    """Main function with multithreading."""
    
    n_threads = min(os.cpu_count(), 24)  # Use up to 24 threads
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_multithreaded.csv'
    
    print("="*80)
    print("MULTITHREADED ENSEMBLE EVALUATION WITH SHARED MEMORY")
    print("="*80)
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"Using {n_threads} threads")
    print(f"Input file: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    # Initialize shared data
    init_shared_data()
    
    # Load the CSV
    print("\nLoading evaluation results CSV...")
    df = pd.read_csv(results_file)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    
    # Check for existing progress
    start_row = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            start_row = sum(1 for line in f) - 1  # Subtract header
        print(f"Found existing output with {start_row:,} rows, resuming...")
    
    # Prepare rows to process
    rows_to_process = []
    for idx in range(start_row, total_rows):
        row = df.iloc[idx]
        rows_to_process.append((idx, row['combination_id'], row['n_models'], row['model_indices']))
    
    print(f"\nRows to process: {len(rows_to_process):,}")
    
    if len(rows_to_process) == 0:
        print("All rows already processed!")
        return
    
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
    print(f"STARTING EVALUATION WITH {n_threads} THREADS")
    print("="*80)
    
    # Start writer thread
    writer_thread = threading.Thread(
        target=result_writer,
        args=(output_file, len(rows_to_process))
    )
    writer_thread.start()
    
    # Process with thread pool
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit all tasks
        futures = []
        for row_data in rows_to_process:
            future = executor.submit(evaluate_combination_thread, row_data)
            futures.append(future)
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                result_queue.put(result)
    
    # Wait for writer to finish
    writer_thread.join()
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total rows processed: {len(rows_to_process):,}")
    print(f"Average rate: {len(rows_to_process)/total_time:.1f} rows/second")
    
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
    top_100_file = 'results/ensemble_checkpoints/top_100_multithreaded.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")


if __name__ == "__main__":
    main()