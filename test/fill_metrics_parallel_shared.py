"""
Parallel evaluation using shared memory for large arrays.
This avoids the overhead of copying large arrays to each worker.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, shared_memory
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc


# Global variables for shared memory
shm_temporal = None
shm_spatial = None
temporal_shape = None
spatial_shape = None
temporal_labels_global = None
spatial_labels_global = None


def init_worker(temporal_name, spatial_name, t_shape, s_shape, t_labels, s_labels):
    """Initialize worker with shared memory arrays."""
    global shm_temporal, shm_spatial, temporal_shape, spatial_shape
    global temporal_labels_global, spatial_labels_global
    
    shm_temporal = shared_memory.SharedMemory(name=temporal_name)
    shm_spatial = shared_memory.SharedMemory(name=spatial_name)
    temporal_shape = t_shape
    spatial_shape = s_shape
    temporal_labels_global = t_labels
    spatial_labels_global = s_labels


def process_row_shared(row_data):
    """Process a single row using shared memory arrays."""
    idx, combination_id, n_models, model_indices_str = row_data
    
    try:
        # Parse model indices
        model_indices = eval(model_indices_str) if isinstance(model_indices_str, str) else model_indices_str
        
        # Reconstruct arrays from shared memory
        temporal_stacked = np.ndarray(temporal_shape, dtype=np.float32, buffer=shm_temporal.buf)
        spatial_stacked = np.ndarray(spatial_shape, dtype=np.float32, buffer=shm_spatial.buf)
        
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
            temporal_labels_global['event_times'],
            temporal_labels_global['event_indicators'],
            temporal_ensemble,
            time_points
        )
        
        spatial_metrics = calculate_all_metrics(
            spatial_labels_global['event_times'],
            spatial_labels_global['event_indicators'],
            spatial_ensemble,
            time_points
        )
        
        # Calculate averages
        temporal_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
        spatial_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
        overall_cidx_avg = (temporal_cidx_avg + spatial_cidx_avg) / 2
        
        return (idx, combination_id, n_models, model_indices_str,
                temporal_metrics['ibs'], temporal_metrics['ibs_event1'], temporal_metrics['ibs_event2'],
                temporal_metrics['cidx_event1'], temporal_metrics['cidx_event2'], temporal_metrics['nll'],
                spatial_metrics['ibs'], spatial_metrics['ibs_event1'], spatial_metrics['ibs_event2'],
                spatial_metrics['cidx_event1'], spatial_metrics['cidx_event2'], spatial_metrics['nll'],
                temporal_cidx_avg, spatial_cidx_avg, overall_cidx_avg)
                
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None


def main():
    """Main function with shared memory for parallel processing."""
    
    n_workers = min(cpu_count(), 12)  # Limit workers to avoid memory issues
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    output_file = 'results/ensemble_checkpoints/evaluation_results_filled.csv'
    
    print("="*80)
    print("PARALLEL ENSEMBLE EVALUATION WITH SHARED MEMORY")
    print("="*80)
    print(f"CPU cores available: {cpu_count()}")
    print(f"Using {n_workers} workers")
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
    processed_combinations = set()
    
    if os.path.exists(output_file):
        print("Found existing output file, loading processed combinations...")
        existing_df = pd.read_csv(output_file)
        processed_combinations = set(existing_df['combination_id'].values)
        start_row = len(processed_combinations)
        print(f"Already processed {start_row:,} combinations")
    
    # Load stacked predictions
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:].astype(np.float32)
    print(f"✓ Temporal stacked loaded: {temporal_stacked.shape}")
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:].astype(np.float32)
    print(f"✓ Spatial stacked loaded: {spatial_stacked.shape}")
    
    # Load labels
    print("\nLoading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    # Create shared memory
    print("\nCreating shared memory arrays...")
    shm_temporal = shared_memory.SharedMemory(create=True, size=temporal_stacked.nbytes)
    shm_spatial = shared_memory.SharedMemory(create=True, size=spatial_stacked.nbytes)
    
    # Copy data to shared memory
    temporal_shared = np.ndarray(temporal_stacked.shape, dtype=np.float32, buffer=shm_temporal.buf)
    spatial_shared = np.ndarray(spatial_stacked.shape, dtype=np.float32, buffer=shm_spatial.buf)
    temporal_shared[:] = temporal_stacked[:]
    spatial_shared[:] = spatial_stacked[:]
    
    # Free original arrays
    del temporal_stacked
    del spatial_stacked
    gc.collect()
    
    print("✓ Shared memory created")
    
    # Prepare rows to process
    rows_to_process = []
    for idx, row in df.iterrows():
        if row['combination_id'] not in processed_combinations:
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
    print(f"STARTING EVALUATION")
    print("="*80)
    
    start_time = time.time()
    processed = 0
    last_log_time = start_time
    
    # Process in batches
    batch_size = 1000
    
    with Pool(n_workers, initializer=init_worker, 
              initargs=(shm_temporal.name, shm_spatial.name, 
                       temporal_shared.shape, spatial_shared.shape,
                       temporal_labels, spatial_labels)) as pool:
        
        for batch_start in range(0, len(rows_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(rows_to_process))
            batch = rows_to_process[batch_start:batch_end]
            
            # Process batch
            results = pool.map(process_row_shared, batch)
            
            # Write results
            with open(output_file, 'a') as f:
                for result in results:
                    if result is not None:
                        idx, comb_id, n_models, model_indices = result[:4]
                        metrics = result[4:]
                        
                        f.write(f"{comb_id},{n_models},\"{model_indices}\",")
                        f.write(','.join(map(str, metrics)) + '\n')
                        
                        processed += 1
            
            # Log progress
            current_time = time.time()
            if current_time - last_log_time > 5:  # Log every 5 seconds
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(rows_to_process) - processed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Processed {processed:,}/{len(rows_to_process):,} "
                      f"({processed/len(rows_to_process)*100:.2f}%) | "
                      f"Rate: {rate:.1f} rows/sec | "
                      f"ETA: {eta_str}")
                
                last_log_time = current_time
    
    # Clean up shared memory
    shm_temporal.close()
    shm_temporal.unlink()
    shm_spatial.close()
    shm_spatial.unlink()
    
    # Final summary
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
    print("-"*80)
    for i, (_, row) in enumerate(results_df_sorted.head(10).iterrows()):
        print(f"Rank {i+1}: Combination {row['combination_id']} | "
              f"Models: {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Indices: {row['model_indices']}")
    
    # Save top 100
    top_100_file = 'results/ensemble_checkpoints/top_100_results_final.csv'
    results_df_sorted.head(100).to_csv(top_100_file, index=False)
    print(f"\nTop 100 results saved to: {top_100_file}")


if __name__ == "__main__":
    main()