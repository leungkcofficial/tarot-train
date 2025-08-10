"""
Parallel version of the DataFrame-based ensemble evaluation.
Uses multiprocessing to speed up the evaluation significantly.
"""

import numpy as np
import pandas as pd
import pickle
import os
from itertools import combinations
import time
from datetime import datetime, timedelta
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
from steps.evaluate_ensemble_combinations_dataframe import stack_deepsurv_models, evaluate_combination
import sys


def evaluate_batch(batch_data, temporal_stacked, spatial_stacked, temporal_labels, spatial_labels):
    """Evaluate a batch of combinations."""
    results = []
    for idx, model_indices in batch_data:
        try:
            metrics = evaluate_combination(
                model_indices,
                temporal_stacked,
                spatial_stacked,
                temporal_labels,
                spatial_labels
            )
            results.append((idx, metrics))
        except Exception as e:
            print(f"Error evaluating combination {idx}: {e}")
            results.append((idx, None))
    return results


def run_parallel_evaluation(
    checkpoint_dir="results/ensemble_checkpoints",
    batch_size=1000,
    n_workers=None
):
    """Run the ensemble evaluation using parallel processing."""
    
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Starting parallel ensemble evaluation at {datetime.now()}")
    print(f"Using {n_workers} workers")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Batch size: {batch_size:,}")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "evaluation_checkpoint.pkl")
    results_file = os.path.join(checkpoint_dir, "evaluation_results.csv")
    
    # Load predictions and labels
    print("\nLoading predictions and labels...")
    
    with open('results/final_deploy/temporal_predictions.pkl', 'rb') as f:
        temporal_predictions = pickle.load(f)
    
    with open('results/final_deploy/spatial_predictions.pkl', 'rb') as f:
        spatial_predictions = pickle.load(f)
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    print(f"Loaded {len(temporal_predictions)} temporal predictions")
    print(f"Loaded {len(spatial_predictions)} spatial predictions")
    
    # Check if we're resuming from a checkpoint
    start_row = 0
    if os.path.exists(checkpoint_file):
        print(f"\nFound checkpoint file, attempting to resume...")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_row = checkpoint_data['last_completed_row'] + 1
                print(f"Resuming from row {start_row:,}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from beginning")
            start_row = 0
    
    # Pre-stack all models
    print("\nPre-stacking all models...")
    temporal_stacked = stack_deepsurv_models(temporal_predictions)
    spatial_stacked = stack_deepsurv_models(spatial_predictions)
    print(f"Stacked shapes - Temporal: {temporal_stacked.shape}, Spatial: {spatial_stacked.shape}")
    
    # Generate all combinations and create DataFrame
    if start_row == 0:
        print("\nGenerating all combinations...")
        all_combinations = []
        combo_idx = 0
        
        # Generate combinations of all possible lengths (2 to 24)
        for length in range(2, 25):
            for combo in combinations(range(24), length):
                all_combinations.append({
                    'combination_id': combo_idx,
                    'n_models': len(combo),
                    'model_indices': list(combo),
                    'temporal_ibs': np.nan,
                    'temporal_ibs_event1': np.nan,
                    'temporal_ibs_event2': np.nan,
                    'temporal_cidx_event1': np.nan,
                    'temporal_cidx_event2': np.nan,
                    'temporal_nll': np.nan,
                    'spatial_ibs': np.nan,
                    'spatial_ibs_event1': np.nan,
                    'spatial_ibs_event2': np.nan,
                    'spatial_cidx_event1': np.nan,
                    'spatial_cidx_event2': np.nan,
                    'spatial_nll': np.nan
                })
                combo_idx += 1
                
                # Print progress every 100k combinations
                if combo_idx % 100000 == 0:
                    print(f"Generated {combo_idx:,} combinations...")
        
        print(f"Total combinations generated: {len(all_combinations):,}")
        
        # Create DataFrame
        df = pd.DataFrame(all_combinations)
        
        # Save initial DataFrame
        df.to_csv(results_file, index=False)
        print(f"Saved initial DataFrame to {results_file}")
    else:
        # Load existing DataFrame
        print(f"\nLoading existing DataFrame from {results_file}")
        df = pd.read_csv(results_file)
        # Convert model_indices from string to list
        df['model_indices'] = df['model_indices'].apply(eval)
    
    total_combinations = len(df)
    print(f"\nTotal combinations to evaluate: {total_combinations:,}")
    
    # Process in parallel batches
    start_time = time.time()
    
    # Create partial function with fixed arguments
    evaluate_func = partial(
        evaluate_batch,
        temporal_stacked=temporal_stacked,
        spatial_stacked=spatial_stacked,
        temporal_labels=temporal_labels,
        spatial_labels=spatial_labels
    )
    
    with Pool(n_workers) as pool:
        for batch_start in range(start_row, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)
            batch_time_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_start:,} to {batch_end:,}")
            print(f"{'='*60}")
            
            # Prepare batch data
            batch_data = []
            for idx in range(batch_start, batch_end):
                batch_data.append((idx, df.loc[idx, 'model_indices']))
            
            # Split batch into chunks for parallel processing
            chunk_size = max(1, len(batch_data) // n_workers)
            chunks = [batch_data[i:i + chunk_size] for i in range(0, len(batch_data), chunk_size)]
            
            # Process chunks in parallel
            print(f"Processing {len(chunks)} chunks in parallel...")
            results = pool.map(evaluate_func, chunks)
            
            # Flatten results and update DataFrame
            for chunk_results in results:
                for idx, metrics in chunk_results:
                    if metrics is not None:
                        for key, value in metrics.items():
                            df.loc[idx, key] = value
            
            # Save batch results
            df.to_csv(results_file, index=False)
            
            # Update checkpoint
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'last_completed_row': batch_end - 1,
                    'timestamp': datetime.now()
                }, f)
            
            batch_time = time.time() - batch_time_start
            elapsed = time.time() - start_time
            rate = (batch_end - start_row) / elapsed
            eta_seconds = (total_combinations - batch_end) / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            print(f"\nBatch completed in {batch_time:.1f} seconds")
            print(f"Overall progress: {batch_end:,}/{total_combinations:,} ({batch_end/total_combinations*100:.2f}%)")
            print(f"Overall rate: {rate:.1f} combos/sec")
            print(f"ETA: {eta_str}")
            
            # Force garbage collection
            gc.collect()
    
    # Final save
    df.to_csv(results_file, index=False)
    
    # Calculate final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETED!")
    print(f"{'='*60}")
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total combinations evaluated: {total_combinations:,}")
    print(f"Average rate: {(total_combinations - start_row)/total_time:.1f} combinations/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    
    # Best by temporal C-index (average of both events)
    df['temporal_cidx_avg'] = (df['temporal_cidx_event1'] + df['temporal_cidx_event2']) / 2
    df['spatial_cidx_avg'] = (df['spatial_cidx_event1'] + df['spatial_cidx_event2']) / 2
    df['overall_cidx_avg'] = (df['temporal_cidx_avg'] + df['spatial_cidx_avg']) / 2
    
    # Sort by overall C-index
    df_sorted = df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTop 10 combinations by overall C-index:")
    top_10 = df_sorted[['combination_id', 'n_models', 'model_indices', 'overall_cidx_avg', 
                        'temporal_cidx_avg', 'spatial_cidx_avg']].head(10)
    
    for idx, row in top_10.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  Combination ID: {row['combination_id']}")
        print(f"  Number of models: {row['n_models']}")
        print(f"  Model indices: {row['model_indices']}")
        print(f"  Overall C-index: {row['overall_cidx_avg']:.4f}")
        print(f"  Temporal C-index: {row['temporal_cidx_avg']:.4f}")
        print(f"  Spatial C-index: {row['spatial_cidx_avg']:.4f}")
    
    # Save top results separately
    top_results_file = os.path.join(checkpoint_dir, "top_100_results.csv")
    df_sorted.head(100).to_csv(top_results_file, index=False)
    print(f"\nSaved top 100 results to {top_results_file}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nRemoved checkpoint file: {checkpoint_file}")
    
    print(f"\nAll results saved to: {results_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    # Check available CPU cores
    n_cores = cpu_count()
    print(f"Available CPU cores: {n_cores}")
    
    # Run with all available cores
    run_parallel_evaluation(
        checkpoint_dir="results/ensemble_checkpoints",
        batch_size=1000,
        n_workers=n_cores
    )