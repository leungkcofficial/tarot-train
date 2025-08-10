# CKD Risk Prediction Pipelines

This directory contains the ZenML pipelines for CKD risk prediction using deep learning survival models.

## Pipeline Structure

The pipeline has been split into two separate pipelines to allow for better separation of concerns:

1. **Training Pipeline (`training_pipeline.py`)**: Focuses on hyperparameter optimization to find the best model configuration.
2. **Model Deployment Pipeline (`model_deployment_pipeline.py`)**: Takes the best hyperparameters and trains, evaluates, exports, and registers the final model.

This separation allows you to:
- Run extensive hyperparameter optimization experiments without retraining the final model each time
- Train multiple final models with different hyperparameter configurations
- Simplify the pipeline structure for better maintainability

## Training Pipeline

The training pipeline (`training_pipeline.py`) is responsible for:

1. Loading and preprocessing the data
2. Running hyperparameter optimization to find the best hyperparameters
3. Returning the best hyperparameters for use in the model deployment pipeline

### Usage

```python
from pipelines.training_pipeline import train_pipeline

# Run the training pipeline
pipeline_instance = train_pipeline(
    model_type="deepsurv",  # Options: "deepsurv", "deephit", "kfre"
    run_hyperparameter_optimization=True
)

# Get the best hyperparameters from the pipeline output
best_hyperparams = pipeline_instance.get_output().get("best_hyperparams", {})
```

## Model Deployment Pipeline

The model deployment pipeline (`model_deployment_pipeline.py`) is responsible for:

1. Loading and preprocessing the data
2. Training the model with the best hyperparameters
3. Evaluating the model on test data
4. Exporting the model to PyTorch and ONNX formats
5. Registering the model with MLflow
6. Running model comparison if requested

### Usage

```python
from pipelines.model_deployment_pipeline import model_deployment_pipeline

# Run the model deployment pipeline
deployment_pipeline_instance = model_deployment_pipeline(
    model_type="deepsurv",  # Options: "deepsurv", "deephit", "kfre"
    best_hyperparams=best_hyperparams,  # From training pipeline
    run_model_comparison=True
)
```

## Complete Example

See `examples/dl_survival_pipeline_example.py` for a complete example of how to use both pipelines together.

## Supported Models

The pipelines support the following model types:

- **DeepSurv**: A Cox proportional hazards deep neural network for survival analysis
- **DeepHit**: A deep neural network for survival analysis with competing risks
- **KFRE**: Kidney Failure Risk Equation (baseline model, no training required)

## Configuration

### YAML Configuration Files

The pipelines use YAML configuration files for better organization and maintainability:

1. **Feature Configuration**: Both pipelines read the feature columns from `src/default_master_df_mapping.yml`. This file contains a `features` key that lists all the features to be used for model development, ensuring consistency in feature selection across the entire pipeline.

2. **Hyperparameter Configuration**: Both pipelines read model configuration and hyperparameter search space from `src/hyperparameter_config.yml`. This file contains:
   - Default model type
   - Network architecture configuration for each model type
   - Hyperparameter search space for optimization
   - Optimization settings (number of trials, patience, etc.)

Example of the hyperparameter configuration file:

```yaml
# Default model type to use
default_model_type: "deepsurv"

# Network architecture configuration
network:
  default:
    hidden_dims: [128, 64, 32]
    num_layers: 3
    dropout: 0.2
    batch_size: 64
    learning_rate: 0.001
    epochs: 100
  
  deephit:
    alpha: 0.2
    sigma: 0.1
    time_grid: [30, 60, 90, 180, 365, 730, 1095, 1460, 1825]

# Hyperparameter search space for optimization
search_space:
  common:
    learning_rate:
      type: "float"
      min: 0.0001
      max: 0.01
      log: true
    # ... other hyperparameters
```

### Environment Variables

Some configuration can still be overridden using environment variables:

- `EPOCHS`: Override the number of epochs for training
- `N_TRIALS`: Override the number of trials for hyperparameter optimization
- `OUTPUT_DIR`: Directory to save model outputs (default: "model_output")

Note: The time grid for DeepHit models and evaluation time points is now defined in the YAML configuration file, ensuring consistency throughout the pipeline.