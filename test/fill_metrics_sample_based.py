"""
Sample-based evaluation to get results in reasonable time.
Evaluates a representative sample instead of all 16.7M combinations.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime, timedelta
import random
from collections import defaultdict


def evaluate_combination(model_indices, temporal_stacked, spatial_stacked,
                        temporal_labels, spatial_labels):
    """Evaluate a single combination."""
    # Slice predictions
    temporal_combo = temporal_stacked[model_indices, :, :, :]
    spatial_combo = spatial_stacked[model_indices, :, :, :]
    
    # Average across models
    temporal_ensemble = np.mean(temporal_combo, axis=0)
    spatial_ensemble = np.mean(spatial_combo, axis=0)
    
    # Time points
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
    
    return temporal_metrics, spatial_metrics


def generate_stratified_sample(total_models=24, samples_per_size=None):
    """Generate a stratified sample of combinations."""
    if samples_per_size is None:
        # Default sampling strategy
        samples_per_size = {
            2: 5000,    # 5k samples for 2-model combinations
            3: 5000,    # 5k samples for 3-model combinations
            4: 5000,    # etc.
            5: 5000,
            6: 4000,
            7: 4000,
            8: 3000,
            9: 3000,
            10: 2000,
            11: 2000,
            12: 2000,
            13: 1500,
            14: 1500,
            15: 1000,
            16: 1000,
            17: 1000,
            18: 1000,
            19: 1000,
            20: 1000,
            21: 1000,
            22: 1000,
            23: 1000,
            24: 1000
        }
    
    sampled_combinations = []
    
    for n_models, n_samples in samples_per_size.items():
        # Generate all possible combinations of this size
        from itertools import combinations
        all_combos = list(combinations(range(total_models), n_models))
        
        # Sample from them
        if len(all_combos) <= n_samples:
            # If we have fewer combinations than requested, use all
            sampled = all_combos
        else:
            # Random sample
            sampled = random.sample(all_combos, n_samples)
        
        sampled_combinations.extend(sampled)
        print(f"Sampled {len(sampled)} combinations of size {n_models}")
    
    return sampled_combinations


def main():
    """Main function for sample-based evaluation."""
    
    output_file = 'results/ensemble_checkpoints/evaluation_results_sampled.csv'
    
    print("="*80)
    print("SAMPLE-BASED ENSEMBLE EVALUATION")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
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
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    print("✓ Labels loaded")
    
    # Generate stratified sample
    print("\nGenerating stratified sample of combinations...")
    sampled_combinations = generate_stratified_sample()
    total_samples = len(sampled_combinations)
    print(f"\nTotal combinations to evaluate: {total_samples:,}")
    print(f"This is {total_samples/16777191*100:.2f}% of all possible combinations")
    
    # Initialize output file
    with open(output_file, 'w') as f:
        f.write(','.join(['combination_id', 'n_models', 'model_indices',
                        'temporal_ibs', 'temporal_ibs_event1', 'temporal_ibs_event2',
                        'temporal_cidx_event1', 'temporal_cidx_event2', 'temporal_nll',
                        'spatial_ibs', 'spatial_ibs_event1', 'spatial_ibs_event2',
                        'spatial_cidx_event1', 'spatial_cidx_event2', 'spatial_nll',
                        'temporal_cidx_avg', 'spatial_cidx_avg', 'overall_cidx_avg']) + '\n')
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    start_time = time.time()
    last_log_time = start_time
    
    # Process combinations
    for idx, model_indices in enumerate(sampled_combinations):
        try:
            # Evaluate
            temporal_metrics, spatial_metrics = evaluate_combination(
                list(model_indices),
                temporal_stacked,
                spatial_stacked,
                temporal_labels,
                spatial_labels
            )
            
            # Calculate averages
            t_cidx_avg = (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2
            s_cidx_avg = (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2
            overall_cidx_avg = (t_cidx_avg + s_cidx_avg) / 2
            
            # Write result
            with open(output_file, 'a') as f:
                f.write(f"{idx},{len(model_indices)},\"{list(model_indices)}\","
                       f"{temporal_metrics['ibs']},{temporal_metrics['ibs_event1']},{temporal_metrics['ibs_event2']},"
                       f"{temporal_metrics['cidx_event1']},{temporal_metrics['cidx_event2']},{temporal_metrics['nll']},"
                       f"{spatial_metrics['ibs']},{spatial_metrics['ibs_event1']},{spatial_metrics['ibs_event2']},"
                       f"{spatial_metrics['cidx_event1']},{spatial_metrics['cidx_event2']},{spatial_metrics['nll']},"
                       f"{t_cidx_avg},{s_cidx_avg},{overall_cidx_avg}\n")
            
            # Progress update
            current_time = time.time()
            if (idx + 1) % 100 == 0 or (current_time - last_log_time) > 10:
                elapsed = current_time - start_time
                rate = (idx + 1) / elapsed
                remaining = total_samples - idx - 1
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Progress: {idx+1:,}/{total_samples:,} ({(idx+1)/total_samples*100:.1f}%) | "
                      f"Rate: {rate:.1f} combos/sec | "
                      f"ETA: {eta_str} | "
                      f"Last C-idx: {overall_cidx_avg:.4f}")
                
                last_log_time = current_time
                
        except Exception as e:
            print(f"\nError at combination {idx}: {e}")
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Total combinations evaluated: {total_samples:,}")
    print(f"Average rate: {total_samples/total_time:.1f} combinations/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    results_df = pd.read_csv(output_file)
    results_df_sorted = results_df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTOP 20 BEST COMBINATIONS:")
    print("-"*80)
    for i, (_, row) in enumerate(results_df_sorted.head(20).iterrows()):
        print(f"Rank {i+1}: Models {row['n_models']} | "
              f"Overall C-idx: {row['overall_cidx_avg']:.4f} | "
              f"Temporal: {row['temporal_cidx_avg']:.4f} | "
              f"Spatial: {row['spatial_cidx_avg']:.4f} | "
              f"Indices: {row['model_indices']}")
    
    # Save top results
    top_results_file = 'results/ensemble_checkpoints/top_1000_sampled.csv'
    results_df_sorted.head(1000).to_csv(top_results_file, index=False)
    print(f"\nTop 1000 results saved to: {top_results_file}")
    
    # Analysis by number of models
    print("\nBest performance by number of models:")
    print("-"*50)
    for n_models in range(2, 25):
        subset = results_df[results_df['n_models'] == n_models]
        if len(subset) > 0:
            best = subset.nlargest(1, 'overall_cidx_avg').iloc[0]
            print(f"{n_models} models: C-idx = {best['overall_cidx_avg']:.4f} | Indices: {best['model_indices']}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()