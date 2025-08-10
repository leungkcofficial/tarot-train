"""
Optimized step to evaluate ALL ensemble combinations using parallel processing with efficient memory usage.
"""

import numpy as np
import pandas as pd
import h5py
import os
import pickle
from typing import Dict, Annotated, Tuple, List, Iterator
from itertools import combinations
from zenml import step
from src.evaluation_metrics import calculate_all_metrics
from multiprocessing import Pool, cpu_count
import time
import sys
from datetime import datetime, timedelta
import math


def generate_combinations_batch(start_idx: int, batch_size: int, n_models: int = 24) -> List[Tuple[int, ...]]:
    """Generate a batch of combinations starting from start_idx."""
    combinations_batch = []
    current_idx = 0
    
    for length in range(2, n_models + 1):
        # Calculate number of combinations for this length
        n_combos_this_length = math.comb(n_models, length)
        
        # Check if start_idx falls within this length
        if current_idx + n_combos_this_length > start_idx:
            # Generate combinations for this length
            combo_generator = combinations(range(n_models), length)
            
            # Skip to start position within this length
            skip_count = max(0, start_idx - current_idx)
            for _ in range(skip_count):
                next(combo_generator)
            
            # Collect combinations for this batch
            for combo in combo_generator:
                if len(combinations_batch) >= batch_size:
                    return combinations_batch
                combinations_batch.append(combo)
        
        current_idx += n_combos_this_length
    
    return combinations_batch


def evaluate_single_combination(args: Tuple) -> Dict:
    """
    Evaluate a single ensemble combination.
    
    Args:
        args: Tuple containing (combo_idx, combo, temporal_cif_all, spatial_cif_all,
                               y_temporal_test, y_spatial_test, TIME_POINTS)
    
    Returns:
        Dictionary with evaluation results
    """
    combo_idx, combo, temporal_cif_all, spatial_cif_all, y_temporal_test, y_spatial_test, TIME_POINTS = args
    
    # Average the selected models
    temporal_ensemble = np.mean(temporal_cif_all[list(combo), :, :, :], axis=0)
    spatial_ensemble = np.mean(spatial_cif_all[list(combo), :, :, :], axis=0)
    
    # Evaluate temporal
    temporal_metrics = calculate_all_metrics(
        times=y_temporal_test['time'].values,
        events=y_temporal_test['event'].values,
        predictions=temporal_ensemble,
        time_points=TIME_POINTS
    )
    
    # Evaluate spatial
    spatial_metrics = calculate_all_metrics(
        times=y_spatial_test['time'].values,
        events=y_spatial_test['event'].values,
        predictions=spatial_ensemble,
        time_points=TIME_POINTS
    )
    
    # Store results - including event-specific IBS
    result = {
        'combination_id': combo_idx,
        'n_models': len(combo),
        'model_indices': list(combo),
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
    
    return result


def save_checkpoint(results: List[Dict], checkpoint_path: str, last_combo_idx: int):
    """Save intermediate results to checkpoint file."""
    checkpoint_data = {
        'results': results,
        'last_combo_idx': last_combo_idx,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict], int]:
    """Load results from checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"Resuming from checkpoint: {checkpoint_data['timestamp']}")
        print(f"Already evaluated: {len(checkpoint_data['results'])} combinations")
        return checkpoint_data['results'], checkpoint_data['last_combo_idx']
    return [], -1


@step
def evaluate_ensemble_combinations_parallel_efficient(
    temporal_cif_all: np.ndarray,
    spatial_cif_all: np.ndarray,
    y_temporal_test: pd.DataFrame,
    y_spatial_test: pd.DataFrame,
    model_info: Dict,
    batch_size: int = 10000,
    n_jobs: int = -1,
    checkpoint_interval: int = 50000
) -> Annotated[Dict, "evaluation_results"]:
    """
    Evaluate ALL ensemble combinations using parallel processing with efficient memory usage.
    
    Args:
        temporal_cif_all: All temporal predictions (24, 2, 5, n_samples)
        spatial_cif_all: All spatial predictions (24, 2, 5, n_samples)
        y_temporal_test: Temporal test labels
        y_spatial_test: Spatial test labels
        model_info: Model information dictionary
        batch_size: Number of combinations to process in each batch
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        checkpoint_interval: Save checkpoint every N combinations
        
    Returns:
        Dictionary containing evaluation results
    """
    print("\nEvaluating ALL ensemble combinations with efficient memory usage...")
    
    # Time points for evaluation
    TIME_POINTS = np.array([365, 730, 1095, 1460, 1825])
    
    # Determine number of CPUs to use
    if n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = min(n_jobs, cpu_count())
    
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Output directory
    output_dir = "results/final_deploy/ensemble_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "evaluation_checkpoint.pkl")
    
    # Load checkpoint if exists
    results, last_combo_idx = load_checkpoint(checkpoint_path)
    
    # Calculate total combinations
    total_combinations = sum(math.comb(24, k) for k in range(2, 25))
    print(f"Total combinations to evaluate: {total_combinations:,}")
    
    # Skip already evaluated combinations
    start_idx = last_combo_idx + 1 if last_combo_idx >= 0 else 0
    remaining_combinations = total_combinations - start_idx
    
    if start_idx > 0:
        print(f"Skipping first {start_idx:,} combinations (already evaluated)")
        print(f"Remaining combinations: {remaining_combinations:,}")
    
    # Estimate time
    estimated_rate = n_jobs * 15  # Conservative estimate: 15 combinations/second/core
    estimated_seconds = remaining_combinations / estimated_rate
    estimated_time = timedelta(seconds=estimated_seconds)
    print(f"\nEstimated time: {estimated_time} (at {estimated_rate} combinations/second)")
    
    # Process in batches
    start_time = time.time()
    batch_start_time = start_time
    current_idx = start_idx
    
    print(f"\nStarting evaluation from combination {current_idx:,}")
    print("=" * 80)
    sys.stdout.flush()
    
    while current_idx < total_combinations:
        # Generate next batch of combinations
        batch_combos = generate_combinations_batch(current_idx, batch_size)
        if not batch_combos:
            break
        
        # Prepare arguments for this batch
        eval_args = []
        for i, combo in enumerate(batch_combos):
            eval_args.append((
                current_idx + i,
                combo,
                temporal_cif_all,
                spatial_cif_all,
                y_temporal_test,
                y_spatial_test,
                TIME_POINTS
            ))
        
        # Evaluate batch in parallel
        batch_num = (current_idx - start_idx) // batch_size + 1
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing batch {batch_num} "
              f"(combinations {current_idx:,} to {current_idx + len(batch_combos) - 1:,})...")
        sys.stdout.flush()
        
        with Pool(processes=n_jobs) as pool:
            batch_results = pool.map(evaluate_single_combination, eval_args)
        
        results.extend(batch_results)
        current_idx += len(batch_combos)
        
        # Progress update
        completed = current_idx
        elapsed = time.time() - start_time
        rate = (completed - start_idx) / elapsed if elapsed > 0 else 0
        remaining = total_combinations - completed
        eta = remaining / rate if rate > 0 else 0
        
        batch_time = time.time() - batch_start_time
        batch_rate = len(batch_combos) / batch_time
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_num} completed")
        print(f"Progress: {completed:,}/{total_combinations:,} ({100*completed/total_combinations:.2f}%)")
        print(f"  - Batch rate: {batch_rate:.1f} combinations/second")
        print(f"  - Overall rate: {rate:.1f} combinations/second")
        print(f"  - Time elapsed: {timedelta(seconds=int(elapsed))}")
        print(f"  - ETA: {timedelta(seconds=int(eta))}")
        print(f"  - Estimated completion: {(datetime.now() + timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        sys.stdout.flush()
        
        batch_start_time = time.time()
        
        # Save checkpoint
        if completed % checkpoint_interval == 0 or completed == total_combinations:
            print(f"\n[CHECKPOINT] Saving at {completed:,} combinations...")
            save_checkpoint(results, checkpoint_path, completed - 1)
            print(f"[CHECKPOINT] Saved successfully")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {timedelta(seconds=int(total_time))}")
    print(f"Average rate: {remaining_combinations/total_time:.1f} combinations/second")
    
    # Convert to DataFrame
    print("\nCreating results DataFrame...")
    results_df = pd.DataFrame(results)
    
    # Save final results
    print("Saving final results...")
    
    # Save to H5 in chunks to handle large data
    output_path = os.path.join(output_dir, "ensemble_evaluation_results_full.h5")
    chunk_size = 100000
    
    with h5py.File(output_path, 'w') as f:
        # Save results in chunks
        for col in results_df.columns:
            if col != 'model_indices':
                f.create_dataset(col, data=results_df[col].values, chunks=True, compression='gzip')
        
        # Save model indices separately
        print("Saving model indices...")
        indices_group = f.create_group('model_indices')
        for i in range(0, len(results_df), chunk_size):
            chunk_end = min(i + chunk_size, len(results_df))
            for j in range(i, chunk_end):
                indices_group.create_dataset(f'combo_{j}', data=np.array(results_df.iloc[j]['model_indices']))
    
    # Save summary statistics
    print("Computing summary statistics...")
    summary_stats = {
        'total_combinations': total_combinations,
        'evaluation_time_seconds': total_time,
        'average_rate': remaining_combinations / total_time,
        'n_jobs_used': n_jobs,
        'best_temporal_ibs': float(results_df['temporal_ibs'].min()),
        'best_temporal_ibs_event1': float(results_df['temporal_ibs_event1'].min()),
        'best_temporal_ibs_event2': float(results_df['temporal_ibs_event2'].min()),
        'best_spatial_ibs': float(results_df['spatial_ibs'].min()),
        'best_spatial_ibs_event1': float(results_df['spatial_ibs_event1'].min()),
        'best_spatial_ibs_event2': float(results_df['spatial_ibs_event2'].min()),
        'best_temporal_cidx_event1': float(results_df['temporal_cidx_event1'].max()),
        'best_spatial_cidx_event1': float(results_df['spatial_cidx_event1'].max()),
        'best_temporal_cidx_event2': float(results_df['temporal_cidx_event2'].max()),
        'best_spatial_cidx_event2': float(results_df['spatial_cidx_event2'].max()),
    }
    
    # Find best combinations for each metric
    print("\nBest combinations by metric:")
    best_combos = {}
    
    for metric, direction in [
        ('temporal_ibs', 'min'),
        ('temporal_ibs_event1', 'min'),
        ('temporal_ibs_event2', 'min'),
        ('spatial_ibs', 'min'),
        ('spatial_ibs_event1', 'min'),
        ('spatial_ibs_event2', 'min'),
        ('temporal_cidx_event1', 'max'),
        ('spatial_cidx_event1', 'max'),
        ('temporal_cidx_event2', 'max'),
        ('spatial_cidx_event2', 'max')
    ]:
        if direction == 'min':
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        
        best_combo = results_df.iloc[best_idx]
        best_combos[metric] = {
            'combination_id': int(best_combo['combination_id']),
            'n_models': int(best_combo['n_models']),
            'model_indices': best_combo['model_indices'],
            'value': float(best_combo[metric])
        }
        
        print(f"  Best {metric}: Combination {best_combo['combination_id']} "
              f"({best_combo['n_models']} models) = {best_combo[metric]:.4f}")
    
    summary_stats['best_combinations'] = best_combos
    
    # Save summary
    import json
    with open(os.path.join(output_dir, "evaluation_summary_full.json"), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save top 1000 combinations for each metric to CSV
    print("\nSaving top combinations to CSV...")
    for metric in ['temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                   'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                   'temporal_cidx_event1', 'spatial_cidx_event1', 
                   'temporal_cidx_event2', 'spatial_cidx_event2']:
        if 'ibs' in metric:
            top_df = results_df.nsmallest(1000, metric)
        else:
            top_df = results_df.nlargest(1000, metric)
        
        top_df.to_csv(os.path.join(output_dir, f"top_1000_{metric}.csv"), index=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed")
    
    # Create evaluation results dictionary
    evaluation_results = {
        'results_df': results_df,
        'output_path': output_path,
        'n_combinations_evaluated': len(results_df),
        'summary_stats': summary_stats,
        'best_combinations': best_combos
    }
    
    return evaluation_results