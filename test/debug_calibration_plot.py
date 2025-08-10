#!/usr/bin/env python3
"""
Debug script to investigate calibration plot generation for mortality predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys
import os

# Add src to path
sys.path.append('src')
from util import load_predictions_from_hdf5

def debug_calibration_plot():
    """Debug the calibration plot generation for mortality predictions."""
    
    print("=== Calibration Plot Debug ===\n")
    
    # Load the prediction file
    prediction_file = '/mnt/dump/yard/projects/tarot2/results/test_predictions/spatial_test_predictions_20250721_234135.h5'
    
    # Load raw predictions
    with h5py.File(prediction_file, 'r') as f:
        predictions_array = f['predictions'][:]
        time_grid = f['time_grid'][:]
    
    # Extract mortality CIF (Cause 2)
    cif_cause2 = predictions_array[1]  # Shape: (5, 3125)
    
    print(f"Mortality CIF shape: {cif_cause2.shape}")
    print(f"Time grid: {time_grid}")
    
    # Simulate the calibration plot generation for Year 5 (index 4)
    time_idx = 4  # Year 5
    horizon = time_grid[time_idx]  # 1825 days
    pred_risks = cif_cause2[time_idx]  # Mortality predictions for Year 5
    
    print(f"\nAnalyzing Year 5 (horizon = {horizon} days):")
    print(f"Predicted risks shape: {pred_risks.shape}")
    print(f"Predicted risks range: [{pred_risks.min():.6f}, {pred_risks.max():.6f}]")
    print(f"Predicted risks mean: {pred_risks.mean():.6f}")
    
    # Create quantiles (same as calibration function)
    n_quantiles = 10
    quantile_labels = pd.qcut(pred_risks, n_quantiles, labels=False, duplicates='drop')
    unique_quantiles = np.unique(quantile_labels)
    
    print(f"\nQuantile analysis:")
    print(f"Number of unique quantiles: {len(unique_quantiles)}")
    
    predicted_risks = []
    for q in unique_quantiles:
        if pd.isna(q):
            continue
        
        mask = (quantile_labels == q)
        mean_pred_risk = np.mean(pred_risks[mask])
        predicted_risks.append(mean_pred_risk)
        
        print(f"Quantile {q}: {np.sum(mask)} samples, mean predicted risk = {mean_pred_risk:.6f}")
    
    # Simulate observed risks (for demonstration)
    # In reality, these would come from Aalen-Johansen estimator
    observed_risks = np.random.uniform(0, 0.002, len(predicted_risks))  # Simulate very low observed risks
    
    print(f"\nSimulated observed risks: {observed_risks}")
    print(f"Predicted risks: {predicted_risks}")
    
    # Test the axis scaling logic from the calibration function
    if len(observed_risks) > 0:
        min_risk = min(min(predicted_risks), min(observed_risks))
        max_risk = max(max(predicted_risks), max(observed_risks))
        
        print(f"\nAxis scaling analysis:")
        print(f"min_risk: {min_risk:.6f}")
        print(f"max_risk: {max_risk:.6f}")
        
        # This is the problematic logic from line 1756-1757
        x_limit = max(max_risk * 1.1, 0.1)
        y_limit = max(max_risk * 1.1, 0.1)
        
        print(f"X-axis limit: {x_limit:.6f}")
        print(f"Y-axis limit: {y_limit:.6f}")
        
        print(f"\n🔍 ISSUE IDENTIFIED:")
        print(f"When max_risk is very small ({max_risk:.6f}), the axis limits default to 0.1 (10%)")
        print(f"This makes tiny mortality risks appear as if they're near 100% on the plot!")
        
        # Create a test plot to demonstrate the issue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: With problematic axis scaling
        ax1.scatter(predicted_risks, observed_risks, alpha=0.7, s=50)
        ax1.plot([min_risk, max_risk], [min_risk, max_risk], 'r--', alpha=0.8, label='Perfect calibration')
        ax1.set_xlim(0, x_limit)
        ax1.set_ylim(0, y_limit)
        ax1.set_xlabel('Predicted Risk')
        ax1.set_ylabel('Observed Risk')
        ax1.set_title('PROBLEMATIC: Fixed 0.1 axis limit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: With corrected axis scaling
        corrected_x_limit = max_risk * 1.2  # Small buffer, but proportional to actual data
        corrected_y_limit = max_risk * 1.2
        
        ax2.scatter(predicted_risks, observed_risks, alpha=0.7, s=50)
        ax2.plot([min_risk, max_risk], [min_risk, max_risk], 'r--', alpha=0.8, label='Perfect calibration')
        ax2.set_xlim(0, corrected_x_limit)
        ax2.set_ylim(0, corrected_y_limit)
        ax2.set_xlabel('Predicted Risk')
        ax2.set_ylabel('Observed Risk')
        ax2.set_title('CORRECTED: Proportional axis limit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_plot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plot saved as 'calibration_plot_comparison.png'")
        print(f"Left plot shows the issue: tiny risks appear large due to 0.1 axis limit")
        print(f"Right plot shows the fix: proportional axis scaling")

if __name__ == "__main__":
    debug_calibration_plot()