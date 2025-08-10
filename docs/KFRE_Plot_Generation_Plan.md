# KFRE Plot Generation Plan

## Overview

This document outlines the plan for creating a Python script that will generate plots comparing predicted and observed risk across different quantiles for the KFRE (Kidney Failure Risk Equation) evaluation results.

## Requirements

1. Create a Python script in `./src` directory
2. Generate plots comparing predicted and observed risk in different quantiles
3. Save output in `./results` directory in PNG format with 1000 DPI
4. Create separate plots for train_df, spatial_test_df, and temporal_test_df
5. For each dataset, create 4 plots:
   - 4 variable 2 year (4v2y)
   - 4 variable 5 year (4v5y)
   - 8 variable 2 year (8v2y)
   - 8 variable 5 year (8v5y)
6. For each plot:
   - X-axis: quantiles
   - Y-axis: risk
   - Each bar should show the mean with 95% percentile from 500 bootstrap samples

## Data Source

The data is stored in `kfre_evaluation_results.csv` with the following columns:
- dataset: train, spatial_test, temporal_test
- bootstrap: bootstrap iteration number (0-499)
- timepoint: 730 (2 years) or 1825 (5 years)
- quantile: 0-9 (risk stratification groups)
- risk: 4v2y, 4v5y, 8v2y, 8v5y (different KFRE models)
- null_brier_score: baseline Brier score
- predicted_risk: risk predicted by the KFRE model
- observed_risk: actual observed risk
- brier_score: Brier score for the model
- c_index: concordance index
- chi2_stat: chi-squared statistic

## Bootstrap Confidence Interval Calculation

The 95% confidence intervals will be calculated using the percentile method from the bootstrap samples:

1. For each combination of dataset, risk type, timepoint, and quantile, we have 500 bootstrap samples.
2. For each bootstrap sample, we have values for predicted_risk and observed_risk.
3. To calculate the 95% confidence interval:
   - Sort all 500 values for predicted_risk (or observed_risk) in ascending order
   - The lower bound (2.5th percentile) is the value at position 500 * 0.025 = 12.5 (rounded to 13th value)
   - The upper bound (97.5th percentile) is the value at position 500 * 0.975 = 487.5 (rounded to 488th value)

This percentile method directly uses the empirical distribution of the bootstrap samples without assuming any particular distribution shape. It's particularly useful for non-normally distributed data and provides a robust estimate of uncertainty.

The calculation in code:
```python
# For predicted risk
pred_lower = np.percentile(bootstrap_samples['predicted_risk'], 2.5)
pred_upper = np.percentile(bootstrap_samples['predicted_risk'], 97.5)

# For observed risk
obs_lower = np.percentile(bootstrap_samples['observed_risk'], 2.5)
obs_upper = np.percentile(bootstrap_samples['observed_risk'], 97.5)
```

The error bars on the plots will then show these confidence intervals, providing a visual representation of the uncertainty in the risk estimates.

## Implementation Plan

### 1. Script Structure

```python
#!/usr/bin/env python3
"""
Generate plots comparing predicted and observed risk for KFRE evaluation results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
```

### 2. Data Loading and Processing

```python
# Load the CSV file
df = pd.read_csv('kfre_evaluation_results.csv')

# Define datasets, risk types, and timepoints
datasets = ['train', 'spatial_test', 'temporal_test']
risk_types = ['4v2y', '4v5y', '8v2y', '8v5y']
timepoints = {730: '2yr', 1825: '5yr'}

# Create output directories if they don't exist
for dataset in datasets:
    os.makedirs(f'results/kfre_evaluation/{dataset}', exist_ok=True)
```

### 3. Function to Calculate Statistics

```python
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
```

### 4. Function to Generate Plots

```python
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
    
    # Create the bars
    plt.bar(r1, pred_means, width=bar_width, label='Predicted Risk', color='blue', alpha=0.7)
    plt.bar(r2, obs_means, width=bar_width, label='Observed Risk', color='red', alpha=0.7)
    
    # Add error bars
    plt.errorbar(r1, pred_means, yerr=[pred_errors_lower, pred_errors_upper], fmt='o', color='black', capsize=5)
    plt.errorbar(r2, obs_means, yerr=[obs_errors_lower, obs_errors_upper], fmt='o', color='black', capsize=5)
    
    # Add labels and title
    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('Risk', fontsize=14)
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
```

### 5. Main Function

```python
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
```

## Expected Output

The script will generate 12 plots in total:

1. `results/kfre_evaluation/train/train_4v_2yr.png`
2. `results/kfre_evaluation/train/train_4v_5yr.png`
3. `results/kfre_evaluation/train/train_8v_2yr.png`
4. `results/kfre_evaluation/train/train_8v_5yr.png`
5. `results/kfre_evaluation/spatial_test/spatial_test_4v_2yr.png`
6. `results/kfre_evaluation/spatial_test/spatial_test_4v_5yr.png`
7. `results/kfre_evaluation/spatial_test/spatial_test_8v_2yr.png`
8. `results/kfre_evaluation/spatial_test/spatial_test_8v_5yr.png`
9. `results/kfre_evaluation/temporal_test/temporal_test_4v_2yr.png`
10. `results/kfre_evaluation/temporal_test/temporal_test_4v_5yr.png`
11. `results/kfre_evaluation/temporal_test/temporal_test_8v_2yr.png`
12. `results/kfre_evaluation/temporal_test/temporal_test_8v_5yr.png`

Each plot will show:
- Quantiles (0-9) on the x-axis
- Predicted and observed risk on the y-axis
- Error bars representing the 95% confidence intervals from bootstrap samples
- Clear labels and title

## Next Steps

After reviewing this plan, we should:

1. Switch to Code mode to implement the actual Python script
2. Test the script with the provided CSV file
3. Review the generated plots for accuracy and clarity
4. Make any necessary adjustments to the script or plot formatting