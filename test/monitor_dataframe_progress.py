"""
Monitor the progress of the DataFrame-based ensemble evaluation.
"""

import pandas as pd
import os
import time
from datetime import datetime, timedelta
import pickle


def monitor_progress():
    """Monitor the evaluation progress by reading the checkpoint and results files."""
    
    checkpoint_dir = "results/ensemble_checkpoints"
    checkpoint_file = os.path.join(checkpoint_dir, "evaluation_checkpoint.pkl")
    results_file = os.path.join(checkpoint_dir, "evaluation_results.csv")
    
    print("Monitoring ensemble evaluation progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_row = -1
    start_time = None
    
    try:
        while True:
            if os.path.exists(checkpoint_file):
                try:
                    # Read checkpoint
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                        current_row = checkpoint_data['last_completed_row']
                        checkpoint_time = checkpoint_data['timestamp']
                    
                    # Read results file to get total
                    if os.path.exists(results_file):
                        df = pd.read_csv(results_file, nrows=1)
                        total_rows = len(pd.read_csv(results_file))
                        
                        if current_row != last_row:
                            if start_time is None:
                                start_time = datetime.now()
                            
                            # Calculate progress
                            progress_pct = (current_row + 1) / total_rows * 100
                            
                            # Calculate rate
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed > 0 and last_row >= 0:
                                rows_processed = current_row - last_row
                                rate = rows_processed / (time.time() - last_update_time)
                                
                                # Calculate ETA
                                remaining = total_rows - current_row - 1
                                eta_seconds = remaining / rate if rate > 0 else 0
                                eta_str = str(timedelta(seconds=int(eta_seconds)))
                                
                                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"Progress: {current_row + 1:,}/{total_rows:,} "
                                      f"({progress_pct:.2f}%) | "
                                      f"Rate: {rate:.1f} combos/sec | "
                                      f"ETA: {eta_str}", end='', flush=True)
                            else:
                                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"Progress: {current_row + 1:,}/{total_rows:,} "
                                      f"({progress_pct:.2f}%)", end='', flush=True)
                            
                            last_row = current_row
                            last_update_time = time.time()
                    
                except Exception as e:
                    print(f"\nError reading checkpoint: {e}")
            
            elif os.path.exists(results_file):
                # Check if evaluation is complete
                try:
                    df = pd.read_csv(results_file)
                    if not df['temporal_ibs'].isna().any():
                        print(f"\n\nEvaluation completed! All {len(df):,} combinations evaluated.")
                        
                        # Show top results
                        if 'overall_cidx_avg' in df.columns:
                            print("\nTop 10 combinations by overall C-index:")
                            top_10 = df.nlargest(10, 'overall_cidx_avg')[
                                ['combination_id', 'n_models', 'overall_cidx_avg', 
                                 'temporal_cidx_avg', 'spatial_cidx_avg']
                            ]
                            print(top_10.to_string(index=False))
                        
                        break
                except Exception as e:
                    print(f"\nError reading results: {e}")
            
            else:
                print("\rWaiting for evaluation to start...", end='', flush=True)
            
            time.sleep(5)  # Check every 5 seconds
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


if __name__ == "__main__":
    monitor_progress()