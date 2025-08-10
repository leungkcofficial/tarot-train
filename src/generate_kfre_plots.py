#!/usr/bin/env python3
"""
Generate plots comparing predicted and observed risk for KFRE evaluation results.

This script reads the KFRE evaluation results from a CSV file and generates
bar plots comparing predicted and observed risk across different quantiles.
The plots are saved in PNG format with 1000 DPI.

For each dataset (train, spatial_test, temporal_test), it creates 4 plots:
- 4 variable 2 year (4v2y)
- 4 variable 5 year (4v5y)
- 8 variable 2 year (8v2y)
- 8 variable 5 year (8v5y)

Each plot shows the mean risk with 95% confidence intervals from bootstrap samples.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_stats(data):
    """
    Calculate mean and 95% confidence intervals for predicted and observed risk.
    
    Args:
        data (DataFrame): DataFrame containing bootstrap samples for a specific 
                         dataset, risk type, timepoint, and quantile
    
    Returns:
        dict: Dictionary containing mean and 95% confidence intervals
    """
    stats = {}
    
    # Calculate mean
    stats['pred_mean'] = data['predicted_risk'].mean()
    stats['obs_mean'] = data['observed_risk'].mean()
    
    # Calculate 95% confidence intervals using percentile method
    stats['pred_lower'] = np.percentile(data['predicted_risk'], 2.5)
    stats['pred_upper'] = np.percentile(data['predicted_risk'], 97.5)
    stats['obs_lower'] = np.percentile(data['observed_risk'], 2.5)
    stats['obs_upper'] = np.percentile(data['observed_risk'], 97.5)
    
    return stats


def generate_plot(df, dataset, risk_type, output_dir):
    """
    Generate a plot comparing predicted and observed risk for a specific dataset and risk type.
    
    Args:
        df (DataFrame): DataFrame containing all KFRE evaluation results
        dataset (str): Dataset name ('train', 'spatial_test', or 'temporal_test')
        risk_type (str): Risk type ('4v2y', '4v5y', '8v2y', or '8v5y')
        output_dir (str): Directory to save the plot
    """
    # Extract timepoint from risk_type (2yr or 5yr)
    if risk_type.endswith('2y'):
        timepoint = 730  # 2 years
        timepoint_str = '2yr'
    else:
        timepoint = 1825  # 5 years
        timepoint_str = '5yr'
    
    # Extract number of variables from risk_type (4v or 8v)
    variables = risk_type[:2]  # '4v' or '8v'
    
    # Filter data for the current dataset, risk type, and timepoint
    filtered_df = df[(df['dataset'] == dataset) & 
                     (df['risk'] == risk_type) & 
                     (df['timepoint'] == timepoint)]
    
    # Group by quantile and calculate statistics
    quantile_stats = {}
    for quantile in range(10):  # 0-9
        quantile_data = filtered_df[filtered_df['quantile'] == quantile]
        if not quantile_data.empty:
            quantile_stats[quantile] = calculate_stats(quantile_data)
    
    # Prepare data for plotting
    quantiles = list(quantile_stats.keys())
    pred_means = [quantile_stats[q]['pred_mean'] for q in quantiles]
    obs_means = [quantile_stats[q]['obs_mean'] for q in quantiles]
    pred_errors_lower = [quantile_stats[q]['pred_mean'] - quantile_stats[q]['pred_lower'] for q in quantiles]
    pred_errors_upper = [quantile_stats[q]['pred_upper'] - quantile_stats[q]['pred_mean'] for q in quantiles]
    obs_errors_lower = [quantile_stats[q]['obs_mean'] - quantile_stats[q]['obs_lower'] for q in quantiles]
    obs_errors_upper = [quantile_stats[q]['obs_upper'] - quantile_stats[q]['obs_mean'] for q in quantiles]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(quantiles))
    r2 = [x + bar_width for x in r1]
    
    # Convert risk values to percentages (0-100%)
    pred_means_pct = [x * 100 for x in pred_means]
    obs_means_pct = [x * 100 for x in obs_means]
    pred_errors_lower_pct = [x * 100 for x in pred_errors_lower]
    pred_errors_upper_pct = [x * 100 for x in pred_errors_upper]
    obs_errors_lower_pct = [x * 100 for x in obs_errors_lower]
    obs_errors_upper_pct = [x * 100 for x in obs_errors_upper]
    
    # Create the bars
    plt.bar(r1, pred_means_pct, width=bar_width, label='Predicted Risk', color='blue', alpha=0.7)
    plt.bar(r2, obs_means_pct, width=bar_width, label='Observed Risk', color='red', alpha=0.7)
    
    # Add error bars
    plt.errorbar(r1, pred_means_pct, yerr=[pred_errors_lower_pct, pred_errors_upper_pct], fmt='o', color='black', capsize=5)
    plt.errorbar(r2, obs_means_pct, yerr=[obs_errors_lower_pct, obs_errors_upper_pct], fmt='o', color='black', capsize=5)
    
    # Add labels and title
    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('Risk (%)', fontsize=14)
    plt.title(f'KFRE {variables} {timepoint_str} - {dataset.replace("_", " ").title()}', fontsize=16)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width/2 for r in range(len(quantiles))], quantiles)
    
    # Add a legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = f'{output_dir}/{dataset}_{variables}_{timepoint_str}.png'
    plt.savefig(output_file, dpi=1000)
    plt.close()
    
    print(f'Plot saved to {output_file}')


def main():
    """
    Main function to generate all plots.
    """
    # Load the data
    df = pd.read_csv('kfre_evaluation_results.csv')
    
    # Create output directories
    for dataset in ['train', 'spatial_test', 'temporal_test']:
        os.makedirs(f'results/kfre_evaluation/{dataset}', exist_ok=True)
    
    # Generate plots for each dataset and risk type
    for dataset in ['train', 'spatial_test', 'temporal_test']:
        for risk_type in ['4v2y', '4v5y', '8v2y', '8v5y']:
            generate_plot(df, dataset, risk_type, f'results/kfre_evaluation/{dataset}')


if __name__ == '__main__':
    main()