# Balance and Optimization Configuration Explanation

## Current Situation

The balance and optimization settings are stored in the **CSV file** (`model_config.csv`), not in the individual JSON files:

```csv
Model No. | Algorithm | Structure | Balancing Method      | Prediction Endpoint | Optimization target
1         | DeepSurv  | ANN       | None                  | Event 1            | Concordance Index
5         | DeepSurv  | ANN       | NearMiss version 1    | Event 1            | Concordance Index
```

## How final_deploy_v2.py Handles This

The `create_model_specific_config()` function **automatically** extracts these settings:

```python
# From the CSV "Balancing Method" column
'balance': {
    'enable': balancing_method != 'None',
    'method': 'near_miss' if 'NearMiss' in balancing_method else ...,
    'near_miss_version': int(balancing_method.split()[-1]) if 'NearMiss version' in balancing_method else 1
}

# From the CSV "Optimization target" column
'optimization': {
    'metric': 'cidx' if 'Concordance' in optimization_target else 'loglik'
}
```

## Your Options

### Option 1: Keep Current Approach (Recommended) ✅

**No changes needed!** The pipeline already works correctly:

1. Reads balance/optimization from CSV
2. Combines with hyperparameters from JSON
3. Creates complete configuration for each model

**Advantages:**
- Already implemented and working
- Single source of truth (CSV)
- No need to modify 36 JSON files

### Option 2: Add to JSON Files

If you prefer to have everything in the JSON files, use the provided script:

```bash
# First, see what would be added (dry run)
python pipelines/update_model_configs.py

# To actually update the files
# Edit the script and change: update_json_configs(update_files=True)
```

**Example updated JSON:**
```json
{
    "model_type": "deepsurv",
    "input_dim": 11,
    "hidden_dims": [119, 124, 14],
    "dropout": 0.03022661174579847,
    "model_path": "/path/to/model_weights.pt",
    "balance": {
        "enable": true,
        "method": "near_miss",
        "sampling_strategy": "majority",
        "near_miss_version": 1
    },
    "optimization": {
        "metric": "cidx",
        "n_trials": 50,
        "patience": 10,
        "seed": 42
    }
}
```

### Option 3: Hybrid Approach

Keep balance/optimization in CSV but modify `final_deploy_v2.py` to also check JSON files:

```python
def create_model_specific_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing code ...
    
    # Check if balance/optimization exist in JSON, otherwise use CSV
    if 'balance' in model_details:
        config['balance'] = model_details['balance']
    else:
        # Use existing logic from CSV
        config['balance'] = {
            'enable': balancing_method != 'None',
            # ... etc
        }
```

## Recommendation

**Use Option 1** - The current implementation already handles everything correctly:

1. **CSV** contains: Algorithm, Structure, Balancing Method, Endpoint, Optimization Target
2. **JSON** contains: Network architecture, hyperparameters, model paths
3. **final_deploy_v2.py** combines both automatically

This separation makes sense because:
- CSV provides a quick overview of all models
- JSON contains detailed technical parameters
- The pipeline handles the integration seamlessly

## To Run the Pipeline

```bash
# With current approach (recommended)
python pipelines/final_deploy_v2.py

# Or with retraining
# Edit the pipeline to set retrain_models=True
```

The pipeline will automatically use the correct balance and optimization settings for each model from the CSV file.