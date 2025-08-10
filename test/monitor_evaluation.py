"""
Monitor the progress of the ensemble evaluation.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta

def monitor_progress(output_file, total_rows, update_interval=10):
    """Monitor evaluation progress."""
    
    print("="*80)
    print("MONITORING ENSEMBLE EVALUATION PROGRESS")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Total combinations: {total_rows:,}")
    print(f"Update interval: {update_interval} seconds")
    print("="*80)
    
    start_time = time.time()
    last_count = 0
    
    while True:
        try:
            # Count rows (excluding header)
            if os.path.exists(output_file):
                current_count = sum(1 for line in open(output_file)) - 1
            else:
                current_count = 0
            
            # Calculate progress
            progress = current_count / total_rows * 100 if total_rows > 0 else 0
            
            # Calculate rate
            elapsed = time.time() - start_time
            rate = current_count / elapsed if elapsed > 0 else 0
            
            # Calculate ETA
            remaining = total_rows - current_count
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            # Print update
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Update:")
            print(f"  Processed: {current_count:,} / {total_rows:,} ({progress:.2f}%)")
            print(f"  Rate: {rate:.1f} rows/sec")
            print(f"  ETA: {eta_str}")
            
            # Check if completed
            if current_count >= total_rows:
                print("\n✓ EVALUATION COMPLETED!")
                break
            
            # Check if stalled
            if current_count == last_count:
                print("  ⚠️  No progress detected - evaluation may be stalled")
            
            last_count = current_count
            
            # Wait before next update
            time.sleep(update_interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(update_interval)

if __name__ == "__main__":
    # Configuration
    output_file = 'results/ensemble_checkpoints/evaluation_results_numba.csv'
    total_rows = 16_777_191  # From the evaluation output
    
    # Start monitoring
    monitor_progress(output_file, total_rows)