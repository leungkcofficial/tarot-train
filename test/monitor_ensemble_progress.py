#!/usr/bin/env python3
"""
Monitor the progress of the ensemble evaluation pipeline.
Run this in a separate terminal while the pipeline is running.
"""

import os
import pickle
import time
from datetime import datetime, timedelta
import sys

def monitor_progress():
    checkpoint_path = "results/final_deploy/ensemble_evaluation/evaluation_checkpoint.pkl"
    total_combinations = 16777191  # 2^24 - 24 - 1
    
    print("Ensemble Evaluation Progress Monitor")
    print("=" * 80)
    print(f"Total combinations to evaluate: {total_combinations:,}")
    print(f"Monitoring checkpoint file: {checkpoint_path}")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    last_combo_idx = -1
    start_time = None
    
    try:
        while True:
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    current_combo_idx = checkpoint_data['last_combo_idx']
                    checkpoint_time = checkpoint_data['timestamp']
                    
                    if current_combo_idx != last_combo_idx:
                        if start_time is None:
                            start_time = datetime.now()
                        
                        completed = current_combo_idx + 1
                        progress = 100 * completed / total_combinations
                        
                        # Calculate rate
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        
                        # Calculate ETA
                        remaining = total_combinations - completed
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta = timedelta(seconds=int(eta_seconds))
                        completion_time = datetime.now() + eta
                        
                        # Clear line and print update
                        sys.stdout.write('\r' + ' ' * 100 + '\r')
                        sys.stdout.write(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Progress: {completed:,}/{total_combinations:,} ({progress:.2f}%) | "
                            f"Rate: {rate:.0f} comb/s | "
                            f"ETA: {eta} | "
                            f"Complete: {completion_time.strftime('%H:%M:%S')}"
                        )
                        sys.stdout.flush()
                        
                        last_combo_idx = current_combo_idx
                        
                except Exception as e:
                    # Checkpoint file might be being written
                    pass
            else:
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                sys.stdout.write(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for pipeline to start...")
                sys.stdout.flush()
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if last_combo_idx >= 0:
            print(f"Last checkpoint: {last_combo_idx + 1:,} combinations completed")

if __name__ == "__main__":
    monitor_progress()