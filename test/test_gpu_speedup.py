"""
Test script to benchmark GPU acceleration vs CPU for ensemble evaluation.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import time
import cupy as cp
from fill_metrics_immediate import calculate_metrics as cpu_calculate_metrics
from fill_metrics_gpu_optimized import calculate_metrics_gpu_batch, gpu_concordance_index_batch


def benchmark_single_combination():
    """Benchmark single combination evaluation: CPU vs GPU."""
    print("="*80)
    print("BENCHMARKING SINGLE COMBINATION")
    print("="*80)
    
    # Load test data
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Test ensemble (average of first 5 models)
    test_indices = [0, 1, 2, 3, 4]
    ensemble_cif = np.mean(temporal_stacked[test_indices], axis=0)
    
    # CPU timing
    print("\nCPU Evaluation:")
    start = time.time()
    cpu_metrics = cpu_calculate_metrics(
        temporal_labels['event_times'],
        temporal_labels['event_indicators'],
        ensemble_cif,
        time_points
    )
    cpu_time = time.time() - start
    print(f"Time: {cpu_time:.4f} seconds")
    print(f"C-index Event 1: {cpu_metrics['cidx_event1']:.4f}")
    print(f"C-index Event 2: {cpu_metrics['cidx_event2']:.4f}")
    
    # GPU timing (including transfer time)
    print("\nGPU Evaluation (with transfer):")
    start = time.time()
    
    # Transfer to GPU
    event_times_gpu = cp.asarray(temporal_labels['event_times'])
    event_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
    ensemble_cif_gpu = cp.asarray(ensemble_cif[np.newaxis, :, :, :])
    time_points_gpu = cp.asarray(time_points)
    
    # Calculate on GPU
    gpu_metrics = calculate_metrics_gpu_batch(
        event_times_gpu, event_indicators_gpu,
        ensemble_cif_gpu, time_points_gpu
    )
    
    # Transfer back
    gpu_metrics_cpu = {k: float(v[0]) for k, v in gpu_metrics.items()}
    
    gpu_time_with_transfer = time.time() - start
    print(f"Time: {gpu_time_with_transfer:.4f} seconds")
    print(f"C-index Event 1: {gpu_metrics_cpu['cidx_event1']:.4f}")
    print(f"C-index Event 2: {gpu_metrics_cpu['cidx_event2']:.4f}")
    
    print(f"\nSpeedup: {cpu_time/gpu_time_with_transfer:.2f}x")


def benchmark_batch_processing():
    """Benchmark batch processing: CPU vs GPU."""
    print("\n" + "="*80)
    print("BENCHMARKING BATCH PROCESSING")
    print("="*80)
    
    # Load data
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random combinations
        combinations = []
        for i in range(batch_size):
            n_models = np.random.randint(5, 15)
            indices = np.random.choice(24, n_models, replace=False).tolist()
            combinations.append(indices)
        
        # CPU timing
        start = time.time()
        cpu_results = []
        for indices in combinations:
            ensemble = np.mean(temporal_stacked[indices], axis=0)
            metrics = cpu_calculate_metrics(
                temporal_labels['event_times'],
                temporal_labels['event_indicators'],
                ensemble,
                time_points
            )
            cpu_results.append(metrics)
        cpu_time = time.time() - start
        
        # GPU timing (with pre-loaded data)
        # Pre-load to GPU
        temporal_stacked_gpu = cp.asarray(temporal_stacked)
        event_times_gpu = cp.asarray(temporal_labels['event_times'])
        event_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
        time_points_gpu = cp.asarray(time_points)
        
        start = time.time()
        
        # Create batch of ensembles
        n_events, n_times, n_samples = temporal_stacked.shape[1:]
        ensemble_batch = cp.zeros((batch_size, n_events, n_times, n_samples), dtype=cp.float32)
        
        for i, indices in enumerate(combinations):
            indices_gpu = cp.array(indices)
            ensemble_batch[i] = cp.mean(temporal_stacked_gpu[indices_gpu], axis=0)
        
        # Calculate metrics in batch
        gpu_metrics = calculate_metrics_gpu_batch(
            event_times_gpu, event_indicators_gpu,
            ensemble_batch, time_points_gpu
        )
        
        gpu_time = time.time() - start
        
        print(f"CPU time: {cpu_time:.4f} seconds ({cpu_time/batch_size:.4f} per combination)")
        print(f"GPU time: {gpu_time:.4f} seconds ({gpu_time/batch_size:.4f} per combination)")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")


def estimate_full_evaluation_time():
    """Estimate time for full 16.7M evaluation with GPU."""
    print("\n" + "="*80)
    print("ESTIMATING FULL EVALUATION TIME")
    print("="*80)
    
    # Benchmark 1000 combinations
    print("\nBenchmarking 1000 combinations...")
    
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked_gpu = cp.asarray(f['stacked_cif'][:])
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked_gpu = cp.asarray(f['stacked_cif'][:])
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    temporal_times_gpu = cp.asarray(temporal_labels['event_times'])
    temporal_indicators_gpu = cp.asarray(temporal_labels['event_indicators'])
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    spatial_times_gpu = cp.asarray(spatial_labels['event_times'])
    spatial_indicators_gpu = cp.asarray(spatial_labels['event_indicators'])
    
    time_points_gpu = cp.asarray([365, 730, 1095, 1460, 1825])
    
    # Generate 1000 random combinations
    test_size = 1000
    combinations = []
    for i in range(test_size):
        n_models = np.random.randint(1, 25)
        indices = np.random.choice(24, n_models, replace=False).tolist()
        combinations.append(indices)
    
    # Time the evaluation
    start = time.time()
    
    n_events, n_times, n_samples_t = temporal_stacked_gpu.shape[1:]
    n_samples_s = spatial_stacked_gpu.shape[3]
    
    temporal_batch = cp.zeros((test_size, n_events, n_times, n_samples_t), dtype=cp.float32)
    spatial_batch = cp.zeros((test_size, n_events, n_times, n_samples_s), dtype=cp.float32)
    
    for i, indices in enumerate(combinations):
        indices_gpu = cp.array(indices)
        temporal_batch[i] = cp.mean(temporal_stacked_gpu[indices_gpu], axis=0)
        spatial_batch[i] = cp.mean(spatial_stacked_gpu[indices_gpu], axis=0)
    
    # Calculate metrics
    temporal_metrics = calculate_metrics_gpu_batch(
        temporal_times_gpu, temporal_indicators_gpu,
        temporal_batch, time_points_gpu
    )
    
    spatial_metrics = calculate_metrics_gpu_batch(
        spatial_times_gpu, spatial_indicators_gpu,
        spatial_batch, time_points_gpu
    )
    
    elapsed = time.time() - start
    rate = test_size / elapsed
    
    print(f"Time for {test_size} combinations: {elapsed:.2f} seconds")
    print(f"Rate: {rate:.1f} combinations/second")
    
    # Estimate for full evaluation
    total_combinations = 16777215
    estimated_seconds = total_combinations / rate
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\nEstimated time for {total_combinations:,} combinations:")
    print(f"  - {estimated_seconds:,.0f} seconds")
    print(f"  - {estimated_hours:,.1f} hours")
    print(f"  - {estimated_days:,.1f} days")
    
    print(f"\nCompared to CPU estimate of 4,488 days:")
    print(f"GPU Speedup: {4488 / estimated_days:.0f}x faster!")


if __name__ == "__main__":
    print("GPU ACCELERATION BENCHMARK")
    print("="*80)
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
    
    benchmark_single_combination()
    benchmark_batch_processing()
    estimate_full_evaluation_time()