"""
Fixed Numba JIT implementation for concordance index.
Addresses the parallel reduction bug in the original implementation.
"""

import numpy as np
import numba
from numba import jit, prange
import time
import pickle
import h5py
from datetime import timedelta


@jit(nopython=True)
def numba_concordance_index_serial(event_times, event_indicators, predictions, event_of_interest):
    """Serial Numba implementation for verification."""
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


@jit(nopython=True, parallel=True)
def numba_concordance_index_parallel_fixed(event_times, event_indicators, predictions, event_of_interest):
    """Fixed parallel Numba implementation using proper reduction."""
    n = len(event_times)
    
    # Pre-filter indices for event of interest
    event_indices = np.where(event_indicators == event_of_interest)[0]
    n_events = len(event_indices)
    
    if n_events == 0:
        return 0.5
    
    # Use thread-local accumulators
    n_threads = numba.config.NUMBA_NUM_THREADS
    concordant_local = np.zeros(n_threads, dtype=np.int64)
    discordant_local = np.zeros(n_threads, dtype=np.int64)
    tied_local = np.zeros(n_threads, dtype=np.int64)
    
    # Parallel loop over event indices
    for idx in prange(n_events):
        i = event_indices[idx]
        thread_id = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(n):
            if event_times[i] < event_times[j]:
                if predictions[i] > predictions[j]:
                    concordant_local[thread_id] += 1
                elif predictions[i] < predictions[j]:
                    discordant_local[thread_id] += 1
                else:
                    tied_local[thread_id] += 1
    
    # Reduce thread-local results
    concordant = np.sum(concordant_local)
    discordant = np.sum(discordant_local)
    tied = np.sum(tied_local)
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


@jit(nopython=True)
def numba_concordance_index_chunked(event_times, event_indicators, predictions, event_of_interest):
    """Chunked approach for better cache efficiency."""
    n = len(event_times)
    
    # Pre-filter indices
    event_mask = event_indicators == event_of_interest
    event_indices = np.where(event_mask)[0]
    n_events = len(event_indices)
    
    if n_events == 0:
        return 0.5
    
    concordant = 0
    discordant = 0
    tied = 0
    
    # Process in chunks for better cache locality
    chunk_size = 100
    
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        
        for idx in range(chunk_start, chunk_end):
            i = event_indices[idx]
            time_i = event_times[i]
            pred_i = predictions[i]
            
            # Vectorized inner loop
            for j in range(n):
                if time_i < event_times[j]:
                    pred_diff = pred_i - predictions[j]
                    if pred_diff > 0:
                        concordant += 1
                    elif pred_diff < 0:
                        discordant += 1
                    else:
                        tied += 1
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


def test_implementations():
    """Test all implementations for correctness and performance."""
    print("="*80)
    print("TESTING NUMBA IMPLEMENTATIONS")
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
    
    # Warm up JIT
    print("\nWarming up JIT...")
    small_sample = 100
    _ = numba_concordance_index_serial(event_times[:small_sample], 
                                      event_indicators[:small_sample], 
                                      test_predictions[:small_sample], 1)
    _ = numba_concordance_index_parallel_fixed(event_times[:small_sample], 
                                              event_indicators[:small_sample], 
                                              test_predictions[:small_sample], 1)
    _ = numba_concordance_index_chunked(event_times[:small_sample], 
                                       event_indicators[:small_sample], 
                                       test_predictions[:small_sample], 1)
    
    print("\n" + "-"*80)
    print("CORRECTNESS TEST (small sample)")
    print("-"*80)
    
    # Test on small sample for correctness
    n_test = 1000
    times_test = event_times[:n_test]
    indicators_test = event_indicators[:n_test]
    preds_test = test_predictions[:n_test]
    
    print(f"\nTesting on {n_test} samples...")
    
    # Serial (ground truth)
    start = time.time()
    cidx_serial = numba_concordance_index_serial(times_test, indicators_test, preds_test, 1)
    serial_time = time.time() - start
    print(f"Serial:          C-index = {cidx_serial:.6f}, Time = {serial_time*1000:.2f} ms")
    
    # Parallel fixed
    start = time.time()
    cidx_parallel = numba_concordance_index_parallel_fixed(times_test, indicators_test, preds_test, 1)
    parallel_time = time.time() - start
    print(f"Parallel fixed:  C-index = {cidx_parallel:.6f}, Time = {parallel_time*1000:.2f} ms")
    
    # Chunked
    start = time.time()
    cidx_chunked = numba_concordance_index_chunked(times_test, indicators_test, preds_test, 1)
    chunked_time = time.time() - start
    print(f"Chunked:         C-index = {cidx_chunked:.6f}, Time = {chunked_time*1000:.2f} ms")
    
    # Check correctness
    print("\nCorrectness check:")
    print(f"Parallel vs Serial: {'✓ PASS' if abs(cidx_parallel - cidx_serial) < 1e-6 else '✗ FAIL'}")
    print(f"Chunked vs Serial:  {'✓ PASS' if abs(cidx_chunked - cidx_serial) < 1e-6 else '✗ FAIL'}")
    
    print("\n" + "-"*80)
    print("PERFORMANCE TEST (full data)")
    print("-"*80)
    
    print(f"\nTesting on full {len(event_times)} samples...")
    
    # Serial
    print("\n1. Serial implementation:")
    start = time.time()
    cidx_serial_full = numba_concordance_index_serial(event_times, event_indicators, test_predictions, 1)
    serial_full_time = time.time() - start
    print(f"   C-index: {cidx_serial_full:.4f}")
    print(f"   Time: {serial_full_time:.2f} seconds")
    
    # Parallel fixed
    print("\n2. Parallel fixed implementation:")
    start = time.time()
    cidx_parallel_full = numba_concordance_index_parallel_fixed(event_times, event_indicators, test_predictions, 1)
    parallel_full_time = time.time() - start
    print(f"   C-index: {cidx_parallel_full:.4f}")
    print(f"   Time: {parallel_full_time:.2f} seconds")
    print(f"   Speedup: {serial_full_time/parallel_full_time:.2f}x")
    
    # Chunked
    print("\n3. Chunked implementation:")
    start = time.time()
    cidx_chunked_full = numba_concordance_index_chunked(event_times, event_indicators, test_predictions, 1)
    chunked_full_time = time.time() - start
    print(f"   C-index: {cidx_chunked_full:.4f}")
    print(f"   Time: {chunked_full_time:.2f} seconds")
    print(f"   Speedup: {serial_full_time/chunked_full_time:.2f}x")
    
    # Batch processing test
    print("\n" + "-"*80)
    print("BATCH PROCESSING TEST")
    print("-"*80)
    
    n_combinations = 100
    combinations = []
    for i in range(n_combinations):
        n_models = np.random.randint(5, 15)
        indices = np.random.choice(24, n_models, replace=False).tolist()
        combinations.append(indices)
    
    # Test fastest implementation
    best_impl = numba_concordance_index_chunked
    best_name = "Chunked"
    
    print(f"\nProcessing {n_combinations} combinations with {best_name} implementation...")
    start = time.time()
    
    for indices in combinations:
        ensemble = np.mean(temporal_stacked[indices], axis=0)
        predictions = ensemble[0, 2, :]
        _ = best_impl(event_times, event_indicators, predictions, 1)
    
    batch_time = time.time() - start
    per_combo_time = batch_time / n_combinations
    
    print(f"Total time: {batch_time:.2f} seconds")
    print(f"Per combination: {per_combo_time*1000:.2f} ms")
    
    # Estimate full evaluation
    print("\n" + "-"*80)
    print("FULL EVALUATION ESTIMATE")
    print("-"*80)
    
    total_combinations = 16_777_215
    # 12 metrics total (2 datasets × 2 events × 3 metrics types)
    metrics_factor = 12
    
    total_seconds = total_combinations * per_combo_time * metrics_factor
    total_hours = total_seconds / 3600
    total_days = total_hours / 24
    
    print(f"\nEstimated time for {total_combinations:,} combinations:")
    print(f"  {total_seconds:,.0f} seconds")
    print(f"  {total_hours:,.1f} hours")
    print(f"  {total_days:,.1f} days")
    
    print(f"\nCompared to original estimate of 4,488 days:")
    print(f"Speedup: {4488 / total_days:.0f}x faster!")


if __name__ == "__main__":
    # Set number of threads for optimal performance
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    print(f"Using {numba.config.NUMBA_NUM_THREADS} threads")
    
    test_implementations()