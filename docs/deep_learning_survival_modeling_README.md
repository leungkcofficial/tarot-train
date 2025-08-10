# Deep Learning Survival Modeling for CKD Risk Prediction

This document provides an overview of the deep learning survival modeling implementation for CKD risk prediction. The implementation uses the PyCox library to train and evaluate DeepSurv and DeepHit models for predicting the risk of end-stage renal disease (ESRD).

## Overview

The implementation consists of several components:

1. **Neural Network Architectures**: Defined in `src/nn_architectures.py`
2. **Survival Utilities**: Utility functions in `src/survival_utils.py`
3. **ZenML Steps**:
   - `steps/survival_dataset_builder.py`: Converts pandas DataFrames to PyCox datasets
   - `steps/model_init.py`: Initializes neural networks
   - `steps/hyperparameter_optimization.py`: Optimizes hyperparameters using Optuna
   - `steps/model_train_surv.py`: Trains models with optimal hyperparameters
   - `steps/model_eval_surv.py`: Evaluates models and generates reports
   - `steps/model_export.py`: Exports models to PyTorch and ONNX formats
   - `steps/model_register.py`: Registers models in MLflow
   - `steps/dl_model_comparison.py`: Compares KFRE, DeepSurv, and DeepHit models
4. **Pipeline Integration**:
   - `pipelines/training_pipeline.py`: Focuses on hyperparameter optimization
   - `pipelines/model_deployment_pipeline.py`: Handles model training, evaluation, export, and registration
   - `examples/dl_survival_pipeline_example.py`: Example script showing how to use both pipelines together
5. **Testing**: Test script in `tests/test_dl_survival_pipeline.py`

## Models

### DeepSurv

DeepSurv is a deep learning-based implementation of the Cox proportional hazards model. It uses a neural network to learn a nonlinear relationship between patient covariates and the risk of an event (ESRD). The model outputs a risk score for each patient, which can be used to rank patients by risk.

### DeepHit

DeepHit is a deep learning model for competing risks survival analysis. It can handle multiple competing events and does not make the proportional hazards assumption. The model outputs a probability mass function (PMF) for each patient, which can be used to calculate the cumulative incidence function (CIF) for each event.

## Usage

### Running the Pipelines

#### Option 1: Using the Example Script

The easiest way to run both pipelines is to use the example script:

```bash
python -m examples.dl_survival_pipeline_example
```

This script will:
1. Run the training pipeline to find the best hyperparameters
2. Use those hyperparameters to train and deploy the best model

#### Option 2: Running Pipelines Separately

To run the training pipeline for hyperparameter optimization:

```bash
python -m pipelines.training_pipeline --model_type deepsurv --run_hyperparameter_optimization True
```

To run the model deployment pipeline with specific hyperparameters:

```bash
python -m pipelines.model_deployment_pipeline --model_type deepsurv --best_hyperparams '{"learning_rate": 0.001, "num_layers": 3, "hidden_units": 128, "dropout": 0.2, "optimizer": "Adam", "batch_size": 64}'
```

Available model types for both pipelines:
- `kfre`: Kidney Failure Risk Equation (baseline)
- `deepsurv`: Deep learning Cox proportional hazards model
- `deephit`: Deep learning competing risks model

### Testing

To test the implementation, use the provided test script:

```bash
python tests/test_dl_survival_pipeline.py --model deepsurv --hyperopt
```

Options:
- `--model`: Model type to test (`deepsurv`, `deephit`, or `both`)
- `--hyperopt`: Flag to run hyperparameter optimization

## Configuration

The models are now configured using YAML configuration files for better organization and maintainability:

### 1. Feature Configuration
Features are read from `src/default_master_df_mapping.yml`, which contains a `features` key that lists all the features to be used for model development.

### 2. Hyperparameter Configuration
Model hyperparameters and search spaces are defined in `src/hyperparameter_config.yml`, which includes:

- **Default Model Type**: The default model type to use
- **Network Architecture**: Default and model-specific network parameters
- **Hyperparameter Search Space**: Configuration for Optuna hyperparameter optimization
- **Optimization Settings**: Number of trials, patience, etc.

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

### Environment Variable Overrides
Some configuration can still be overridden using environment variables:

- `EPOCHS`: Override the number of epochs for training
- `N_TRIALS`: Override the number of trials for hyperparameter optimization
- `OUTPUT_DIR`: Directory to save model outputs (default: "model_output")

Note: The time grid for DeepHit models and evaluation time points is now defined in the YAML configuration file, eliminating the need for a separate `TIME_HORIZONS` variable.

## Pipeline Steps

### 1. Dataset Building

The `survival_dataset_builder` step converts pandas DataFrames to PyCox datasets. It handles:
- Feature selection
- Train/validation splitting
- Feature scaling
- Conversion to PyCox format

### 2. Model Initialization

The `model_init` step initializes neural networks for DeepSurv or DeepHit. It creates:
- Neural network architecture
- Optimizer
- Loss function
- PyCox model wrapper

### 3. Hyperparameter Optimization

The `hyperparameter_optimization` step uses Optuna to find optimal hyperparameters based on the search space defined in `src/hyperparameter_config.yml`. It optimizes:
- Learning rate
- Number of layers
- Hidden units
- Dropout rate
- Batch size
- Optimizer type
- DeepHit-specific parameters (alpha, sigma)

The search space for each parameter is configurable through the YAML file, allowing for easy adjustment of the optimization process without code changes. The step supports:
- Different parameter types (float, int, categorical)
- Log-scale search for appropriate parameters
- Custom ranges for each parameter
- Model-specific parameter spaces

### 4. Model Training

The `model_train_surv` step trains the model with optimal hyperparameters. It includes:
- Early stopping
- Training curve visualization
- Model checkpointing
- MLflow logging

### 5. Model Evaluation

The `model_eval_surv` step evaluates the model on test data. It calculates:
- Concordance index
- Integrated Brier score
- Time-dependent AUC
- Calibration plots
- Survival curves

### 6. Model Export

The `model_export` step exports the model to PyTorch and ONNX formats. It creates:
- PyTorch model file
- ONNX model file
- Model parameters file
- Inference script
- README file

### 7. Model Registration

The `model_register` step registers the model in MLflow. It handles:
- Model versioning
- Model staging
- Artifact logging
- Registration summary

### 8. Model Comparison

The `dl_model_comparison` step compares KFRE, DeepSurv, and DeepHit models. It generates:
- Comparison tables
- Comparison plots
- Radar plots
- Statistical tests

## Pipeline Architecture

The deep learning survival modeling steps are now organized into two separate pipelines:

### 1. Training Pipeline (`pipelines/training_pipeline.py`)
This pipeline focuses solely on hyperparameter optimization:
- Processes and prepares data
- Builds survival datasets
- Performs hyperparameter optimization using Optuna
- Returns the best hyperparameters for model training

### 2. Model Deployment Pipeline (`pipelines/model_deployment_pipeline.py`)
This pipeline handles model training and deployment with the best hyperparameters:
- Processes and prepares data
- Builds survival datasets
- Trains the model with optimal hyperparameters
- Evaluates model performance
- Exports the model for deployment
- Registers the model in MLflow
- Compares model performance with other models (optional)

### Example Usage
The `examples/dl_survival_pipeline_example.py` script demonstrates how to use both pipelines together:
1. Run the training pipeline to find the best hyperparameters
2. Extract the best hyperparameters from the pipeline output
3. Run the model deployment pipeline with the best hyperparameters
4. Analyze the evaluation results and export paths

This separation of concerns allows for more flexibility and better organization of the workflow.

## Evaluation Metrics

The implementation uses the following metrics for model evaluation:

- **Concordance Index (C-index)**: Measures the model's ability to rank patients by risk
- **Integrated Brier Score (IBS)**: Measures the model's calibration and discrimination
- **Time-dependent AUC**: Measures the model's discrimination at specific time points
- **Calibration Plots**: Assess the model's calibration at specific time points
- **Survival Curves**: Visualize the model's predictions for individual patients

## Dependencies

- PyCox: Deep learning survival analysis library
- PyTorch: Deep learning framework
- Optuna: Hyperparameter optimization framework
- MLflow: Model tracking and registration
- ZenML: Pipeline management
- ONNX: Model export format
- Matplotlib, Seaborn: Visualization
- Pandas, NumPy: Data manipulation

## References

1. Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. "Time-to-event prediction with neural networks and Cox regression." Journal of Machine Learning Research 20.129 (2019): 1-30.
2. Changhee Lee, William R. Zame, Jinsung Yoon, and Mihaela van der Schaar. "DeepHit: A deep learning approach to survival analysis with competing risks." AAAI Conference on Artificial Intelligence (2018).
3. Faraggi, D., & Simon, R. (1995). A neural network model for survival data. Statistics in Medicine, 14(1), 73-82.