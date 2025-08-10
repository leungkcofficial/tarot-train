# Multi-Model Deployment Pipeline V2 - Usage Guide

## Overview
The `final_deploy_v2.py` pipeline is an enhanced version that properly handles model-specific configurations from the CSV and JSON files. It can either:
1. Load pre-trained models and use them for predictions (default)
2. Retrain models with their specific configurations before predictions

## Key Improvements

### Model-Specific Configuration Handling
- Each model uses its own configuration from the JSON files and CSV
- Balancing methods are applied per model based on the CSV specification
- Optimization targets (Concordance Index vs Log-likelihood) are respected
- Network architectures (ANN vs LSTM) are loaded from individual configs

### Configuration Mapping
The pipeline creates model-specific configurations that include:
- **Model Type**: DeepSurv or DeepHit
- **Target Endpoint**: Event 1, Event 2, or Both (for DeepHit)
- **Network Structure**: ANN or LSTM with specific architectures
- **Balancing Method**: None, NearMiss (versions 1-3), or KNN
- **Optimization Metric**: Concordance Index or Log-likelihood

## Usage

### Basic Usage (Load Pre-trained Models)
```bash
python pipelines/final_deploy_v2.py
```

### Retrain Models with Specific Configurations
```python
from pipelines.final_deploy_v2 import multi_model_deploy_pipeline

# Modify the pipeline to retrain models
@pipeline(enable_cache=False)
def multi_model_deploy_pipeline_retrain():
    # ... (same data preparation steps)
    
    ensemble_results = deploy_multiple_models_v2(
        train_df_preprocessed=train_df_preprocessed,
        temporal_test_df_preprocessed=temporal_test_df_preprocessed,
        spatial_test_df_preprocessed=spatial_test_df_preprocessed,
        retrain_models=True  # Enable retraining
    )
    
    return ensemble_results

# Run the pipeline
pipeline = multi_model_deploy_pipeline_retrain()
pipeline.run()
```

## Model Configuration Structure

### From CSV (model_config.csv)
```
Model No. | Algorithm | Structure | Balancing Method      | Prediction Endpoint | Optimization target
1         | DeepSurv  | ANN       | None                  | Event 1            | Concordance Index
5         | DeepSurv  | ANN       | NearMiss version 1    | Event 1            | Concordance Index
28        | DeepHit   | ANN       | NearMiss version 3    | Both               | Concordance Index
```

### From JSON (model{i}_details_*.json)
```json
{
    "model_type": "deepsurv",
    "input_dim": 11,
    "hidden_dims": [119, 124, 14],
    "output_dim": 1,
    "dropout": 0.03022661174579847,
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "Adam",
    "model_path": "/path/to/model_weights.pt"
}
```

## How Configuration is Applied

1. **Loading Phase**:
   - Reads model metadata from CSV
   - Loads hyperparameters from JSON files
   - Combines both to create complete configuration

2. **Model Creation**:
   - Uses network architecture from JSON (hidden_dims, dropout, etc.)
   - Applies model type (DeepSurv/DeepHit) from JSON
   - Sets target endpoint based on CSV

3. **Training (if enabled)**:
   - Applies balancing method from CSV
   - Uses optimization metric from CSV
   - Applies all hyperparameters from JSON

## Differences from Original Pipeline

| Aspect | Original (final_deploy.py) | V2 (final_deploy_v2.py) |
|--------|---------------------------|-------------------------|
| Configuration | Uses global hyperparameter_config.yml | Uses individual model configs |
| Balancing | Not applied during deployment | Can apply model-specific balancing |
| Retraining | Always loads pre-trained | Optional retraining with configs |
| Flexibility | Fixed configuration | Per-model configuration |

## Output Structure

Same as the original pipeline:
```
results/final_deploy/
├── individual_predictions/
│   ├── model{i}_spatial_predictions_*.h5
│   ├── model{i}_temporal_predictions_*.h5
│   └── metadata files...
├── ensemble_predictions/
│   ├── ensemble_spatial_predictions_*.h5
│   ├── ensemble_temporal_predictions_*.h5
│   └── deployment_log_*.json
```

## Deployment Log

The deployment log now includes:
- Whether models were retrained
- Individual model configurations used
- Balancing methods applied
- Optimization targets

## Recommendations

1. **For Production Deployment**: Use `retrain_models=False` to load existing trained models
2. **For Model Updates**: Use `retrain_models=True` to retrain with latest data
3. **For Testing**: Start with a subset of models by filtering the CSV

## Example: Using Specific Models Only

```python
# In deploy_multiple_models_v2 function
model_configs = load_all_model_configurations(config_csv_path, config_dir)

# Filter to specific models
model_configs = [c for c in model_configs if c['model_no'] in [1, 2, 3, 28, 29, 30]]
```

## Troubleshooting

### Issue: Models using wrong configuration
**Solution**: Use final_deploy_v2.py which creates model-specific configurations

### Issue: Balancing not applied correctly
**Solution**: Set `retrain_models=True` to apply model-specific balancing during training

### Issue: Memory issues during retraining
**Solution**: Process models in smaller batches or reduce batch_size in configurations