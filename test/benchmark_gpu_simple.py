"""
Simple benchmark to test GPU acceleration potential for ensemble evaluation.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import time
import cupy as cp
from datetime import timedelta


def cpu_concordance_index(event_times, event_indicators, predictions, event_of_interest):
    """CPU version of C-index calculation."""
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(n):
        if event_indicators[i] == event_of_interest:
            for j in range(n):
                if event_times[i] < event_times[j]:
                    if predictions[i] > predictions[j]:
                        concordant += 1
                    elif predictions[i] < predictions[j]:
                        discordant += 1
                    else:
                        tied += 1
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


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


def main():
    """Main benchmark function."""
    
    print("="*80)
    print("GPU ACCELERATION BENCHMARK FOR ENSEMBLE EVALUATION")
    print("="*80)
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
    
    # Load sample data
    print("\nLoading data...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    print(f"Stacked shape: {temporal_stacked.shape}")
    print(f"Number of samples: {len(temporal_labels['event_times'])}")
    
    # Test single C-index calculation
    print("\n" + "-"*80)
    print("SINGLE C-INDEX CALCULATION")
    print("-"*80)
    
    # Create test ensemble
    test_ensemble = np.mean(temporal_stacked[:5], axis=0)  # Average first 5 models
    test_predictions = test_ensemble[0, 2, :]  # Event 1, time point 2
    
    # CPU timing
    print("\nCPU:")
    start = time.time()
    cpu_cidx = cpu_concordance_index(
        temporal_labels['event_times'],
        temporal_labels['event_indicators'],
        test_predictions,
        1
    )
    cpu_time = time.time() - start
    print(f"C-index: {cpu_cidx:.4f}")
    print(f"Time: {cpu_time*1000:.2f} ms")
    
    # GPU timing
    print("\nGPU (with transfer):")
    start = time.time()
    
    # Transfer to GPU
    event_times_gpu = cp.asarray(temporal_labels['event_times'])
    event_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
    predictions_gpu = cp.asarray(test_predictions)
    
    # Calculate
    gpu_cidx = gpu_concordance_index(event_times_gpu, event_indicators_gpu, predictions_gpu, 1)
    
    gpu_time = time.time() - start
    print(f"C-index: {gpu_cidx:.4f}")
    print(f"Time: {gpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    # Test batch processing
    print("\n" + "-"*80)
    print("BATCH PROCESSING (100 combinations)")
    print("-"*80)
    
    # Generate 100 random combinations
    n_combinations = 100
    combinations = []
    for i in range(n_combinations):
        n_models = np.random.randint(5, 15)
        indices = np.random.choice(24, n_models, replace=False).tolist()
        combinations.append(indices)
    
    # CPU timing
    print("\nCPU:")
    start = time.time()
    for indices in combinations:
        ensemble = np.mean(temporal_stacked[indices], axis=0)
        predictions = ensemble[0, 2, :]
        cidx = cpu_concordance_index(
            temporal_labels['event_times'],
            temporal_labels['event_indicators'],
            predictions,
            1
        )
    cpu_batch_time = time.time() - start
    print(f"Time: {cpu_batch_time:.2f} seconds")
    print(f"Per combination: {cpu_batch_time/n_combinations*1000:.2f} ms")
    
    # GPU timing (with pre-loaded data)
    print("\nGPU (data pre-loaded):")
    
    # Pre-load to GPU
    temporal_stacked_gpu = cp.asarray(temporal_stacked)
    event_times_gpu = cp.asarray(temporal_labels['event_times'])
    event_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
    
    start = time.time()
    for indices in combinations:
        indices_gpu = cp.array(indices)
        ensemble_gpu = cp.mean(temporal_stacked_gpu[indices_gpu], axis=0)
        predictions_gpu = ensemble_gpu[0, 2, :]
        cidx = gpu_concordance_index(event_times_gpu, event_indicators_gpu, predictions_gpu, 1)
    gpu_batch_time = time.time() - start
    print(f"Time: {gpu_batch_time:.2f} seconds")
    print(f"Per combination: {gpu_batch_time/n_combinations*1000:.2f} ms")
    print(f"Speedup: {cpu_batch_time/gpu_batch_time:.1f}x")
    
    # Estimate full evaluation time
    print("\n" + "-"*80)
    print("ESTIMATED TIME FOR FULL EVALUATION")
    print("-"*80)
    
    total_combinations = 16_777_215
    
    # Assume we need to calculate 6 metrics per combination (2 events × 3 metrics)
    # and process both temporal and spatial
    metrics_per_combination = 12
    
    # Conservative estimate: use the batch timing
    seconds_per_combination_cpu = cpu_batch_time / n_combinations
    seconds_per_combination_gpu = gpu_batch_time / n_combinations
    
    cpu_total_seconds = total_combinations * seconds_per_combination_cpu * 2  # ×2 for full metrics
    gpu_total_seconds = total_combinations * seconds_per_combination_gpu * 2
    
    print(f"\nTotal combinations: {total_combinations:,}")
    print(f"\nCPU estimate:")
    print(f"  - {cpu_total_seconds:,.0f} seconds")
    print(f"  - {cpu_total_seconds/3600:,.1f} hours")
    print(f"  - {cpu_total_seconds/86400:,.1f} days")
    
    print(f"\nGPU estimate:")
    print(f"  - {gpu_total_seconds:,.0f} seconds")
    print(f"  - {gpu_total_seconds/3600:,.1f} hours")
    print(f"  - {gpu_total_seconds/86400:,.1f} days")
    
    print(f"\nPotential speedup: {cpu_total_seconds/gpu_total_seconds:.1f}x")
    
    # Memory estimate
    print("\n" + "-"*80)
    print("MEMORY REQUIREMENTS")
    print("-"*80)
    
    batch_size = 10000
    n_events, n_times, n_samples = temporal_stacked.shape[1:]
    
    # Memory for batch of ensembles
    ensemble_memory = batch_size * n_events * n_times * n_samples * 4 / 1e9  # float32
    
    print(f"Memory per {batch_size:,} combinations: {ensemble_memory:.2f} GB")
    print(f"GPU memory available: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
    
    if ensemble_memory < cp.cuda.runtime.memGetInfo()[1] / 1e9:
        print("✓ Batch size is feasible for GPU memory")
    else:
        suggested_batch = int(batch_size * (cp.cuda.runtime.memGetInfo()[1] / 1e9) / ensemble_memory * 0.8)
        print(f"✗ Reduce batch size to ~{suggested_batch:,}")


if __name__ == "__main__":
    main()