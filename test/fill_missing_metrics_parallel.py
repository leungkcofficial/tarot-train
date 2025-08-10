"""
Parallel version to fill in missing metrics in the evaluation results CSV.
Uses multiprocessing to speed up the evaluation of 16.7 million combinations.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
import gc


def evaluate_batch(batch_indices, df, temporal_stacked, spatial_stacked, 
                  temporal_labels, spatial_labels):
    """Evaluate a batch of combinations and return results."""
    results = []
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    for idx in batch_indices:
        try:
            # Get model indices for this combination
            model_indices = df.loc[idx, 'model_indices']
            if isinstance(model_indices, str):
                model_indices = eval(model_indices)
            
            # Slice predictions for this combination
            temporal_combo = temporal_stacked[model_indices, :, :, :]
            spatial_combo = spatial_stacked[model_indices, :, :, :]
            
            # Average across models
            temporal_ensemble = np.mean(temporal_combo, axis=0)
            spatial_ensemble = np.mean(spatial_combo, axis=0)
            
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
            
            result = {
                'idx': idx,
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
            results.append(result)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return results


def fill_missing_metrics_parallel(n_workers=None, batch_size=10000):
    """Fill in missing metrics using parallel processing."""
    
    if n_workers is None:
        n_workers = cpu_count()
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    checkpoint_file = 'results/ensemble_checkpoints/fill_checkpoint.pkl'
    
    print(f"Starting parallel metric filling with {n_workers} workers")
    print(f"Batch size: {batch_size:,}")
    print("="*60)
    
    # Load the CSV
    print(f"\nLoading results from {results_file}...")
    df = pd.read_csv(results_file)
    
    # Count missing metrics
    missing_mask = df['temporal_ibs'].isna()
    n_missing = missing_mask.sum()
    n_total = len(df)
    
    print(f"\nTotal rows: {n_total:,}")
    print(f"Rows with missing metrics: {n_missing:,}")
    
    if n_missing == 0:
        print("\nAll metrics are already filled!")
        return
    
    # Check for checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_idx = checkpoint_data['last_processed_idx'] + 1
                print(f"Resuming from index {start_idx:,}")
        except:
            start_idx = 0
    
    # Load stacked predictions
    print("\nLoading stacked predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    
    print(f"Temporal stacked shape: {temporal_stacked.shape}")
    print(f"Spatial stacked shape: {spatial_stacked.shape}")
    
    # Load labels
    print("Loading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    # Get indices of rows with missing metrics
    missing_indices = df[missing_mask].index.tolist()
    
    # Filter based on checkpoint
    if start_idx > 0:
        missing_indices = [idx for idx in missing_indices if idx >= start_idx]
    
    print(f"\nProcessing {len(missing_indices):,} rows...")
    
    start_time = time.time()
    processed = 0
    
    # Process in batches
    for batch_start in range(0, len(missing_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(missing_indices))
        batch_indices = missing_indices[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start//batch_size + 1}: indices {batch_indices[0]} to {batch_indices[-1]}")
        batch_time_start = time.time()
        
        # Split batch for parallel processing
        chunk_size = max(1, len(batch_indices) // n_workers)
        chunks = [batch_indices[i:i + chunk_size] for i in range(0, len(batch_indices), chunk_size)]
        
        # Create partial function with fixed arguments
        evaluate_func = partial(
            evaluate_batch,
            df=df,
            temporal_stacked=temporal_stacked,
            spatial_stacked=spatial_stacked,
            temporal_labels=temporal_labels,
            spatial_labels=spatial_labels
        )
        
        # Process chunks in parallel
        with Pool(n_workers) as pool:
            chunk_results = pool.map(evaluate_func, chunks)
        
        # Update DataFrame with results
        for results in chunk_results:
            for result in results:
                idx = result['idx']
                for key, value in result.items():
                    if key != 'idx':
                        df.loc[idx, key] = value
        
        # Save batch results
        df.to_csv(results_file, index=False)
        
        # Update checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'last_processed_idx': batch_indices[-1],
                'timestamp': datetime.now()
            }, f)
        
        # Progress update
        processed += len(batch_indices)
        batch_time = time.time() - batch_time_start
        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = len(missing_indices) - processed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        print(f"Batch completed in {batch_time:.1f} seconds")
        print(f"Overall progress: {processed:,}/{len(missing_indices):,} ({processed/len(missing_indices)*100:.1f}%)")
        print(f"Overall rate: {rate:.1f} rows/sec")
        print(f"ETA: {eta_str}")
        
        # Force garbage collection
        gc.collect()
    
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
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nRemoved checkpoint file: {checkpoint_file}")


if __name__ == "__main__":
    # Check available CPU cores
    n_cores = cpu_count()
    print(f"Available CPU cores: {n_cores}")
    
    # Run with all available cores
    fill_missing_metrics_parallel(n_workers=n_cores, batch_size=10000)