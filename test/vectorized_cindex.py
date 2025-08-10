"""
Highly optimized vectorized concordance index implementation.
Uses advanced NumPy/CuPy techniques to minimize loops and maximize parallelism.
"""

import numpy as np
import cupy as cp
import time
from numba import jit, prange
import pickle
import h5py


@jit(nopython=True, parallel=True)
def numba_concordance_index(event_times, event_indicators, predictions, event_of_interest):
    """Numba JIT-compiled C-index for CPU with parallel execution."""
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in prange(n):
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


def vectorized_concordance_index_numpy(event_times, event_indicators, predictions, event_of_interest):
    """Fully vectorized C-index using NumPy broadcasting with memory optimization."""
    # Filter to event of interest
    mask = event_indicators == event_of_interest
    event_times_filtered = event_times[mask]
    predictions_filtered = predictions[mask]
    
    n_events = len(event_times_filtered)
    n_all = len(event_times)
    
    if n_events == 0:
        return 0.5
    
    # Process in chunks to avoid memory explosion
    chunk_size = min(1000, n_events)  # Adjust based on available memory
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(0, n_events, chunk_size):
        end_i = min(i + chunk_size, n_events)
        
        # Vectorized comparisons for this chunk
        times_chunk = event_times_filtered[i:end_i, np.newaxis]
        preds_chunk = predictions_filtered[i:end_i, np.newaxis]
        
        # Compare with all samples
        time_diff = times_chunk - event_times[np.newaxis, :]
        pred_diff = preds_chunk - predictions[np.newaxis, :]
        
        # Valid pairs where event happened before
        valid_mask = time_diff < 0
        
        # Count outcomes
        concordant += np.sum((pred_diff > 0) & valid_mask)
        discordant += np.sum((pred_diff < 0) & valid_mask)
        tied += np.sum((pred_diff == 0) & valid_mask)
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


def vectorized_concordance_index_gpu(event_times_gpu, event_indicators_gpu, predictions_gpu, event_of_interest):
    """GPU-optimized vectorized C-index with memory-efficient chunking."""
    # Filter to event of interest
    mask = event_indicators_gpu == event_of_interest
    event_times_filtered = event_times_gpu[mask]
    predictions_filtered = predictions_gpu[mask]
    
    n_events = len(event_times_filtered)
    n_all = len(event_times_gpu)
    
    if n_events == 0:
        return 0.5
    
    # Determine optimal chunk size based on GPU memory
    free_mem = cp.cuda.runtime.memGetInfo()[0]
    # Each comparison needs 2 * chunk_size * n_all * 4 bytes (float32)
    max_chunk = int(free_mem / (2 * n_all * 4 * 10))  # Use 10% of free memory
    chunk_size = min(max_chunk, n_events, 2000)
    
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(0, n_events, chunk_size):
        end_i = min(i + chunk_size, n_events)
        
        # Vectorized comparisons for this chunk
        times_chunk = event_times_filtered[i:end_i, None]
        preds_chunk = predictions_filtered[i:end_i, None]
        
        # Compare with all samples
        time_diff = times_chunk - event_times_gpu[None, :]
        pred_diff = preds_chunk - predictions_gpu[None, :]
        
        # Valid pairs where event happened before
        valid_mask = time_diff < 0
        
        # Count outcomes
        concordant += int(cp.sum((pred_diff > 0) & valid_mask))
        discordant += int(cp.sum((pred_diff < 0) & valid_mask))
        tied += int(cp.sum((pred_diff == 0) & valid_mask))
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


def benchmark_vectorized_approaches():
    """Benchmark different vectorized implementations."""
    print("="*80)
    print("VECTORIZED C-INDEX BENCHMARK")
    print("="*80)
    
    # Load test data
    print("Loading data...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    # Create test ensemble
    test_ensemble = np.mean(temporal_stacked[:5], axis=0)
    test_predictions = test_ensemble[0, 2, :]
    
    event_times = temporal_labels['event_times']
    event_indicators = temporal_labels['event_indicators']
    
    print(f"\nData shape: {len(event_times)} samples")
    print(f"Event 1 count: {np.sum(event_indicators == 1)}")
    print(f"Event 2 count: {np.sum(event_indicators == 2)}")
    
    # Warm up Numba
    print("\nWarming up Numba JIT...")
    _ = numba_concordance_index(event_times[:100], event_indicators[:100], test_predictions[:100], 1)
    
    # Benchmark implementations
    print("\n" + "-"*80)
    print("SINGLE C-INDEX CALCULATION")
    print("-"*80)
    
    # 1. Numba JIT (parallel CPU)
    print("\n1. Numba JIT (parallel CPU):")
    start = time.time()
    cidx_numba = numba_concordance_index(event_times, event_indicators, test_predictions, 1)
    numba_time = time.time() - start
    print(f"   C-index: {cidx_numba:.4f}")
    print(f"   Time: {numba_time*1000:.2f} ms")
    
    # 2. Vectorized NumPy
    print("\n2. Vectorized NumPy (chunked):")
    start = time.time()
    cidx_numpy = vectorized_concordance_index_numpy(event_times, event_indicators, test_predictions, 1)
    numpy_time = time.time() - start
    print(f"   C-index: {cidx_numpy:.4f}")
    print(f"   Time: {numpy_time*1000:.2f} ms")
    print(f"   Speedup vs Numba: {numba_time/numpy_time:.2f}x")
    
    # 3. Vectorized GPU
    print("\n3. Vectorized GPU (chunked):")
    # Transfer to GPU
    event_times_gpu = cp.asarray(event_times)
    event_indicators_gpu = cp.asarray(event_indicators)
    predictions_gpu = cp.asarray(test_predictions)
    
    start = time.time()
    cidx_gpu = vectorized_concordance_index_gpu(event_times_gpu, event_indicators_gpu, predictions_gpu, 1)
    gpu_time = time.time() - start
    print(f"   C-index: {cidx_gpu:.4f}")
    print(f"   Time: {gpu_time*1000:.2f} ms")
    print(f"   Speedup vs Numba: {numba_time/gpu_time:.2f}x")
    
    # Batch processing test
    print("\n" + "-"*80)
    print("BATCH PROCESSING (100 combinations)")
    print("-"*80)
    
    # Generate test combinations
    n_test = 100
    combinations = []
    for i in range(n_test):
        n_models = np.random.randint(5, 15)
        indices = np.random.choice(24, n_models, replace=False).tolist()
        combinations.append(indices)
    
    # Pre-load GPU data
    temporal_stacked_gpu = cp.asarray(temporal_stacked)
    
    # Numba batch
    print("\nNumba JIT batch:")
    start = time.time()
    for indices in combinations:
        ensemble = np.mean(temporal_stacked[indices], axis=0)
        predictions = ensemble[0, 2, :]
        _ = numba_concordance_index(event_times, event_indicators, predictions, 1)
    numba_batch_time = time.time() - start
    print(f"   Total time: {numba_batch_time:.2f} s")
    print(f"   Per combination: {numba_batch_time/n_test*1000:.2f} ms")
    
    # GPU batch
    print("\nGPU batch (data pre-loaded):")
    start = time.time()
    for indices in combinations:
        indices_gpu = cp.array(indices)
        ensemble_gpu = cp.mean(temporal_stacked_gpu[indices_gpu], axis=0)
        predictions_gpu = ensemble_gpu[0, 2, :]
        _ = vectorized_concordance_index_gpu(event_times_gpu, event_indicators_gpu, predictions_gpu, 1)
    gpu_batch_time = time.time() - start
    print(f"   Total time: {gpu_batch_time:.2f} s")
    print(f"   Per combination: {gpu_batch_time/n_test*1000:.2f} ms")
    print(f"   Speedup: {numba_batch_time/gpu_batch_time:.2f}x")
    
    # Estimate for full evaluation
    print("\n" + "-"*80)
    print("ESTIMATED TIME FOR FULL EVALUATION")
    print("-"*80)
    
    total_combinations = 16_777_215
    # Need to calculate 12 metrics (2 datasets × 2 events × 3 metrics)
    metrics_factor = 12
    
    # Conservative estimates
    numba_per_combo = numba_batch_time / n_test * metrics_factor
    gpu_per_combo = gpu_batch_time / n_test * metrics_factor
    
    numba_total_hours = total_combinations * numba_per_combo / 3600
    gpu_total_hours = total_combinations * gpu_per_combo / 3600
    
    print(f"\nUsing Numba JIT (parallel CPU):")
    print(f"   {numba_total_hours:.1f} hours ({numba_total_hours/24:.1f} days)")
    
    print(f"\nUsing GPU (vectorized + chunked):")
    print(f"   {gpu_total_hours:.1f} hours ({gpu_total_hours/24:.1f} days)")
    
    print(f"\nSpeedup: {numba_total_hours/gpu_total_hours:.1f}x")
    
    # Memory usage
    print("\n" + "-"*80)
    print("MEMORY ANALYSIS")
    print("-"*80)
    
    n_samples = len(event_times)
    full_comparison_memory = n_samples * n_samples * 4 / 1e9  # float32
    print(f"Full pairwise comparison matrix: {full_comparison_memory:.1f} GB")
    print(f"GPU memory available: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
    
    if full_comparison_memory > cp.cuda.runtime.memGetInfo()[1] / 1e9:
        print("✗ Full matrix too large - chunking required")
    else:
        print("✓ Full matrix fits in GPU memory")


if __name__ == "__main__":
    benchmark_vectorized_approaches()