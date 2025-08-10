# KFRE Evaluation Implementation Plan

## Objective
Create a function that evaluates KFRE risk predictions by:
1. Using train_df, spatial_test_df, and temporal_test_df as input
2. Bootstrapping 500 times
3. Getting rows with available KFRE risk
4. Calculating null Brier score, Brier score, c-index, and chi2 statistics
5. Outputting results in CSV format

## Implementation Steps

### 1. Function Signature
```python
def evaluate_kfre(
    train_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    output_path: str = "kfre_evaluation_results.csv",
    n_bootstrap: int = 500,
    n_quantiles: int = 10
) -> pd.DataFrame:
    """
    Evaluate KFRE risk predictions on the given datasets.
    
    Args:
        train_df: Training dataset
        spatial_test_df: Spatial test dataset
        temporal_test_df: Temporal test dataset
        output_path: Path to save the evaluation results CSV
        n_bootstrap: Number of bootstrap iterations
        n_quantiles: Number of quantiles for risk stratification
        
    Returns:
        DataFrame containing evaluation results
    """
```

### 2. Calculate KFRE Risk Scores
```python
# Import required libraries
from src.KFRE import KFRECalculator
from src.util import df_event_focus
from src.model_evaluator import NullModel
from lifelines import KaplanMeierFitter
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index
from scipy.stats import chi2
import numpy as np
import pandas as pd
import os

# Calculate KFRE risk scores for all datasets
kfre_calculator = KFRECalculator("src/default_master_df_mapping.yml")
train_df_kfre = kfre_calculator.add_kfre_risk(train_df)
spatial_test_df_kfre = kfre_calculator.add_kfre_risk(spatial_test_df)
temporal_test_df_kfre = kfre_calculator.add_kfre_risk(temporal_test_df)

# Define datasets dictionary for easier iteration
datasets = {
    'train': train_df_kfre,
    'spatial_test': spatial_test_df_kfre,
    'temporal_test': temporal_test_df_kfre
}
```

### 3. Initialize Results Storage
```python
# Initialize results list
results = []

# Define risk columns and time points
risk_cols = ['4v2y', '4v5y', '8v2y', '8v5y']
time_points = np.array([730, 1825])  # 2 years and 5 years in days
```

### 4. Bootstrap Loop
```python
# For each bootstrap iteration
for b in range(n_bootstrap):
    print(f"Processing bootstrap {b+1} of {n_bootstrap}...")
    
    # For each dataset
    for dataset_name, dataset in datasets.items():
        # Sample with replacement
        bootstrap_indices = np.random.choice(dataset.index, size=len(dataset), replace=True)
        df_bootstrap = dataset.loc[bootstrap_indices].copy()
        
        # For each time point
        for t in time_points:
            # For each risk column
            for risk_col in risk_cols:
                # Get rows with non-null risk values
                df_focus = df_bootstrap[df_bootstrap[risk_col].notnull()].copy()
                
                # Focus on dialysis events (endpoint = 1)
                df_focus = df_event_focus(df_focus, 'endpoint', 1)
                
                # Skip if no data
                if len(df_focus) == 0:
                    continue
                
                # Calculate null Brier score
                null_brier = NullModel.calculate_null_brier(
                    df_focus['duration'].values,
                    df_focus['endpoint'].values,
                    time_grid=np.array([t]),
                    event_of_interest=1
                )
                
                # Divide into quantiles
                df_focus['quantile'] = pd.qcut(
                    df_focus[risk_col], 
                    n_quantiles, 
                    labels=False, 
                    duplicates='drop'
                )
                
                # For each quantile
                for q in df_focus['quantile'].unique():
                    # Get subset for this quantile
                    quantile_df = df_focus[df_focus['quantile'] == q].copy()
                    
                    # Calculate observed risk using Kaplan-Meier
                    kmf = KaplanMeierFitter()
                    kmf.fit(quantile_df['duration'], quantile_df['endpoint'])
                    
                    # Get observed risk at time point
                    if t in kmf.cumulative_density_.index:
                        observed_risk = kmf.cumulative_density_.loc[t].values[0]
                    else:
                        # Find closest time point
                        closest_idx = kmf.cumulative_density_.index.get_indexer([t], method='nearest')[0]
                        closest_time = kmf.cumulative_density_.index[closest_idx]
                        observed_risk = kmf.cumulative_density_.loc[closest_time].values[0]
                    
                    # Calculate predicted risk (mean of KFRE risk in this quantile)
                    predicted_risk = quantile_df[risk_col].mean()
                    
                    # Calculate Brier score for the entire dataset
                    brier_score = brier_score_loss(
                        (df_focus['duration'] <= t) & (df_focus['endpoint'] == 1),
                        df_focus[risk_col]
                    )
                    
                    # Calculate c-index
                    c_index = concordance_index(
                        df_focus['duration'],
                        -df_focus[risk_col],
                        df_focus['endpoint']
                    )
                    
                    # Calculate chi2 statistic
                    # Group by quantile
                    grouped = df_focus.groupby('quantile')
                    observed_events = grouped['endpoint'].sum()
                    expected_events = grouped[risk_col].sum()
                    n = grouped.size()
                    chi2_stat = np.sum(((observed_events - expected_events) ** 2) / (expected_events * (1 - expected_events / n)))
                    
                    # Store results
                    results.append({
                        'dataset': dataset_name,
                        'bootstrap': b,
                        'timepoint': t,
                        'quantile': q,
                        'risk': risk_col,
                        'null_brier_score': null_brier,
                        'predicted_risk': predicted_risk,
                        'observed_risk': observed_risk,
                        'brier_score': brier_score,
                        'c_index': c_index,
                        'chi2_stat': chi2_stat
                    })
```

### 5. Save Results to CSV
```python
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(output_path, index=False)

return results_df
```

## Output Format
The CSV file will have the following columns:
1. dataset: 'train', 'spatial_test', or 'temporal_test'
2. bootstrap: the order of bootstrap for the test (0-499)
3. timepoint: the time point (730 or 1825 days)
4. quantile: the quantile of predicted risk
5. risk: '4v2y', '4v5y', '8v2y', or '8v5y'
6. null_brier_score: the null Brier score
7. predicted_risk: the mean predicted risk
8. observed_risk: the mean observed risk
9. brier_score: the Brier score of the KFRE
10. c_index: concordance index of KFRE
11. chi2_stat: the chi2 test of predicted and observed risk

## Integration with kfre_eval.py
This implementation will be integrated into the existing `kfre_eval.py` file, replacing the incomplete implementation there. The function will be called from the training pipeline to evaluate KFRE risk predictions on the training and test datasets.