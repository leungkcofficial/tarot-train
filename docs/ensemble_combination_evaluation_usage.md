# Ensemble Combination Evaluation Pipeline Usage Guide

## Overview

The ensemble combination evaluation pipeline evaluates ALL possible combinations of your 24 CIF arrays to find the best performing ensemble. It uses bootstrap evaluation with three key metrics:

- **Integrated Brier Score (IBS)** - Lower is better
- **Concordance Index (C-index)** - Higher is better  
- **Negative Log-Likelihood (NLL)** - Lower is better

## Understanding the Scale

With 24 models, there are **2^24 - 1 = 16,777,215** possible combinations! This includes:
- Single model predictions
- Pairs of models
- Triplets of models
- ... up to all 24 models combined

## Prerequisites

1. Run `final_deploy_v2.py` first to generate individual model predictions
2. Ensure predictions are saved in `results/final_deploy/predictions/`
3. Have sufficient computational resources (evaluating all combinations will take significant time)

## Basic Usage

```python
python pipelines/ensemble_combination_evaluation.py
```

## Configuration Options

Edit the script to modify these parameters:

```python
# In main() function:
predictions_dir = "results/final_deploy/predictions"  # Where model predictions are stored
output_dir = "results/final_deploy/ensemble_evaluation"  # Where to save results
max_combinations = 1000  # Limit for testing (set to None for ALL combinations)
n_bootstraps = 10  # Number of bootstrap iterations per combination
```

## Recommended Approach

### 1. Start Small (Testing)
```python
max_combinations = 1000  # Evaluate first 1000 combinations
```

### 2. Medium Scale (Exploratory)
```python
max_combinations = 10000  # Evaluate 10,000 combinations
```

### 3. Full Scale (Production)
```python
max_combinations = None  # Evaluate ALL 16.7M combinations
# WARNING: This will take a very long time!
```

## Output Files

The pipeline generates several output files:

### 1. Full Results (Pickle)
```
ensemble_evaluation_results_YYYYMMDD_HHMMSS.pkl
```
Contains complete results including:
- Model combinations
- Bootstrap scores for each metric
- Mean and 95% CI for each metric

### 2. Summary CSV
```
ensemble_evaluation_summary_YYYYMMDD_HHMMSS.csv
```
Columns:
- `combination_id`: Unique identifier
- `n_models`: Number of models in combination
- `model_ids`: Comma-separated model IDs
- `ibs_mean`, `ibs_ci_lower`, `ibs_ci_upper`
- `cidx_event1_mean`, `cidx_event1_ci_lower`, `cidx_event1_ci_upper`
- `cidx_event2_mean`, `cidx_event2_ci_lower`, `cidx_event2_ci_upper`
- `nll_mean`, `nll_ci_lower`, `nll_ci_upper`

### 3. Intermediate Results
```
intermediate_results_YYYYMMDD_HHMMSS.pkl
```
Saved every 1000 combinations for recovery if interrupted

## Analyzing Results

### Load and Analyze Results
```python
import pickle
import pandas as pd

# Load full results
with open('results/final_deploy/ensemble_evaluation/ensemble_evaluation_results_*.pkl', 'rb') as f:
    results = pickle.load(f)

# Load summary
summary = pd.read_csv('results/final_deploy/ensemble_evaluation/ensemble_evaluation_summary_*.csv')

# Find best by IBS
best_ibs = summary.nsmallest(10, 'ibs_mean')
print("Best 10 combinations by IBS:")
print(best_ibs[['combination_id', 'n_models', 'model_ids', 'ibs_mean']])

# Find best by C-index Event 1
best_cidx1 = summary.nlargest(10, 'cidx_event1_mean')
print("\nBest 10 combinations by C-index Event 1:")
print(best_cidx1[['combination_id', 'n_models', 'model_ids', 'cidx_event1_mean']])
```

### Understanding Model Composition
```python
# Analyze which models appear most frequently in top combinations
top_100 = summary.nsmallest(100, 'ibs_mean')
model_frequency = {}

for idx, row in top_100.iterrows():
    models = row['model_ids'].split(',')
    for model in models:
        model_frequency[model] = model_frequency.get(model, 0) + 1

print("Model frequency in top 100 combinations:")
for model, freq in sorted(model_frequency.items(), key=lambda x: x[1], reverse=True):
    print(f"Model {model}: {freq} times")
```

## Computational Considerations

### Time Estimates
- 1 combination ≈ 0.1-0.5 seconds (depends on data size and bootstraps)
- 1,000 combinations ≈ 2-8 minutes
- 10,000 combinations ≈ 20-80 minutes
- ALL combinations ≈ 20-80 hours

### Memory Requirements
- Results are saved incrementally
- Peak memory usage depends on data size
- Approximately 1-2 GB for typical datasets

### Parallelization Options
To speed up evaluation, you could modify the script to:
1. Use multiprocessing for bootstrap evaluation
2. Distribute combinations across multiple machines
3. Use GPU for metric calculations (if applicable)

## Interpreting Results

### Best Practices
1. **Don't rely on a single metric** - Look at performance across all metrics
2. **Consider model diversity** - Ensembles with diverse models often perform better
3. **Check confidence intervals** - Narrow CIs indicate more stable performance
4. **Validate on holdout data** - Test best combinations on completely new data

### Example Interpretation
```
Combination 5432 (12 models)
IBS: 0.1234 (95% CI: 0.1180 - 0.1288)
C-index Event 1: 0.7856 (95% CI: 0.7712 - 0.7999)
C-index Event 2: 0.8123 (95% CI: 0.7989 - 0.8257)
NLL: 0.4567 (95% CI: 0.4423 - 0.4711)

Models: 1,3,5,7,9,11,25,27,29,31,33,35
Composition: {'deepsurv': 6, 'deephit': 6}
```

This shows:
- Balanced mix of DeepSurv and DeepHit models
- Good performance across all metrics
- Reasonable confidence intervals

## Next Steps

After finding the best combinations:

1. **Implement the chosen ensemble** in production
2. **Consider weighted averaging** based on individual model performance
3. **Implement stacking** as mentioned in your plans
4. **Monitor performance** on new data

## Troubleshooting

### Out of Memory
- Reduce `n_bootstraps`
- Process in smaller batches
- Use a machine with more RAM

### Taking Too Long
- Start with `max_combinations` limit
- Use sampling strategies (e.g., random sampling of combinations)
- Implement parallel processing

### Missing Predictions
- Ensure `final_deploy_v2.py` completed successfully
- Check that all 24 model predictions exist in the predictions directory