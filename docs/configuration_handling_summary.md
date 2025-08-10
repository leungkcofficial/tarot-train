# Configuration Handling in Multi-Model Deployment

## The Issue
The original `final_deploy.py` was designed to load pre-trained models, but based on your feedback about the `deploy_model` step in the training pipeline, we discovered that:

1. The `deploy_model` step actually retrains models with optimized hyperparameters
2. It uses the global `hyperparameter_config.yml` for all models
3. This means all models would use the same balancing configuration, which is incorrect

## The Solution: final_deploy_v2.py

### Key Features

1. **Model-Specific Configuration Creation**
   ```python
   def create_model_specific_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
       # Creates a configuration that mimics hyperparameter_config.yml structure
       # But uses model-specific settings from CSV and JSON
   ```

2. **Flexible Model Loading/Training**
   ```python
   def create_and_train_model(
       model_config: Dict[str, Any],
       train_data: pd.DataFrame,
       feature_cols: List[str],
       device: str = 'cpu',
       retrain: bool = True  # Can choose to retrain or just load
   ) -> Any:
   ```

3. **Configuration Sources**
   - **From CSV**: Algorithm, Structure, Balancing Method, Endpoint, Optimization Target
   - **From JSON**: Network architecture, hyperparameters, model paths

### Configuration Mapping Example

For Model 5 (DeepSurv, ANN, NearMiss version 1, Event 1, Concordance Index):

```python
{
    'model_type': 'deepsurv',
    'target_endpoint': 1,  # Event 1
    'network': {
        'type': 'ann',
        'default': {
            'hidden_dims': [119, 124, 14],  # From JSON
            'dropout': 0.03022661174579847,  # From JSON
            # ... other hyperparameters
        }
    },
    'optimization': {
        'metric': 'cidx'  # Concordance Index from CSV
    },
    'balance': {
        'enable': True,
        'method': 'near_miss',
        'near_miss_version': 1  # From CSV "NearMiss version 1"
    }
}
```

## Usage Options

### Option 1: Load Pre-trained Models (Default)
```python
ensemble_results = deploy_multiple_models_v2(
    train_df_preprocessed=train_df_preprocessed,
    temporal_test_df_preprocessed=temporal_test_df_preprocessed,
    spatial_test_df_preprocessed=spatial_test_df_preprocessed,
    retrain_models=False  # Just load existing weights
)
```

### Option 2: Retrain with Specific Configurations
```python
ensemble_results = deploy_multiple_models_v2(
    # ... same data arguments
    retrain_models=True  # Retrain each model with its specific config
)
```

## Benefits

1. **Correct Configuration**: Each model uses its intended settings
2. **Flexibility**: Can choose between loading or retraining
3. **Reproducibility**: Model-specific configurations are preserved
4. **Compatibility**: Works with existing model files and CSV structure

## When to Use Each Option

- **Use `retrain_models=False`** when:
  - Models are already properly trained
  - You want fast predictions
  - You're in production

- **Use `retrain_models=True`** when:
  - You need to ensure models use correct configurations
  - You have new training data
  - You're updating the models

## Files Created

1. **pipelines/final_deploy_v2.py** - Enhanced pipeline with proper configuration handling
2. **pipelines/final_deploy_v2_usage.md** - Detailed usage guide
3. **pipelines/configuration_handling_summary.md** - This summary

The original `final_deploy.py` remains unchanged and can still be used if you're certain the models were trained with correct configurations.