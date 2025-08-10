"""
Debug script to test why evaluation is getting stuck.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import time
from datetime import datetime
import numba
from numba import jit
from fill_metrics_numba_clean import numba_concordance_index, numba_brier_score, calculate_metrics_numba


def test_single_evaluation():
    """Test a single evaluation to see where it gets stuck."""
    
    print("="*80)
    print("DEBUGGING EVALUATION")
    print("="*80)
    
    # Load minimal data
    print("\n1. Loading data...")
    start = time.time()
    
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:5]  # Only load first 5 models
    
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    print(f"   Data loaded in {time.time() - start:.2f} seconds")
    print(f"   Stacked shape: {temporal_stacked.shape}")
    print(f"   Number of samples: {len(temporal_labels['event_times'])}")
    
    # Test ensemble creation
    print("\n2. Creating test ensemble...")
    start = time.time()
    model_indices = [0, 1, 2]
    ensemble = np.mean(temporal_stacked[model_indices], axis=0)
    print(f"   Ensemble created in {time.time() - start:.2f} seconds")
    print(f"   Ensemble shape: {ensemble.shape}")
    
    # Test metrics calculation
    print("\n3. Testing metrics calculation...")
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Test with small sample first
    print("\n   a) Testing with 100 samples...")
    start = time.time()
    small_metrics = calculate_metrics_numba(
        temporal_labels['event_times'][:100],
        temporal_labels['event_indicators'][:100],
        ensemble[:, :, :100],
        time_points
    )
    print(f"      Completed in {time.time() - start:.2f} seconds")
    print(f"      C-index Event 1: {small_metrics['cidx_event1']:.4f}")
    print(f"      C-index Event 2: {small_metrics['cidx_event2']:.4f}")
    
    # Test with full data
    print("\n   b) Testing with full data (42,953 samples)...")
    print("      This is where it might get stuck...")
    start = time.time()
    
    try:
        full_metrics = calculate_metrics_numba(
            temporal_labels['event_times'],
            temporal_labels['event_indicators'],
            ensemble,
            time_points
        )
        elapsed = time.time() - start
        print(f"      Completed in {elapsed:.2f} seconds")
        print(f"      C-index Event 1: {full_metrics['cidx_event1']:.4f}")
        print(f"      C-index Event 2: {full_metrics['cidx_event2']:.4f}")
        
        # Estimate for full evaluation
        print(f"\n4. Performance estimate:")
        print(f"   Single evaluation time: {elapsed:.2f} seconds")
        print(f"   Rate: {1/elapsed:.2f} evaluations/second")
        total_combinations = 16_777_191
        total_seconds = total_combinations * elapsed * 2  # x2 for both datasets
        print(f"   Estimated total time: {total_seconds/3600:.1f} hours ({total_seconds/86400:.1f} days)")
        
    except KeyboardInterrupt:
        print("\n   INTERRUPTED - Evaluation is taking too long!")
        elapsed = time.time() - start
        print(f"   Stuck after {elapsed:.2f} seconds")
    
    # Test reading the CSV file
    print("\n5. Testing CSV reading...")
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    
    print("   Reading first 5 rows...")
    df_head = pd.read_csv(results_file, nrows=5)
    print(f"   Columns: {list(df_head.columns)}")
    print(f"   First row model_indices: {df_head.iloc[0]['model_indices']}")
    
    # Test eval() on model_indices
    print("\n6. Testing model_indices parsing...")
    model_indices_str = df_head.iloc[0]['model_indices']
    print(f"   String: {model_indices_str}")
    print(f"   Type: {type(model_indices_str)}")
    
    try:
        parsed = eval(model_indices_str)
        print(f"   Parsed: {parsed}")
        print(f"   Type after parsing: {type(parsed)}")
    except Exception as e:
        print(f"   ERROR parsing: {e}")


def test_chunked_reading():
    """Test if the issue is with chunked CSV reading."""
    print("\n" + "="*80)
    print("TESTING CHUNKED CSV READING")
    print("="*80)
    
    results_file = 'results/ensemble_checkpoints/evaluation_results.csv'
    
    print("\nReading in chunks of 100...")
    chunk_count = 0
    row_count = 0
    
    try:
        for chunk in pd.read_csv(results_file, chunksize=100):
            chunk_count += 1
            row_count += len(chunk)
            
            if chunk_count <= 3:
                print(f"Chunk {chunk_count}: {len(chunk)} rows, indices {chunk.index[0]}-{chunk.index[-1]}")
            
            if chunk_count == 10:
                print(f"... processed {row_count} rows so far")
                break
                
    except Exception as e:
        print(f"ERROR reading chunks: {e}")


if __name__ == "__main__":
    # Set Numba threads
    numba.set_num_threads(4)  # Use fewer threads for testing
    print(f"Using {numba.config.NUMBA_NUM_THREADS} Numba threads")
    
    # Run tests
    test_single_evaluation()
    test_chunked_reading()