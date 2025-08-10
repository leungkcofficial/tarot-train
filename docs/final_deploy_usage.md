# Multi-Model Deployment Pipeline Usage Guide

## Overview
The `final_deploy.py` pipeline loads multiple pre-trained survival analysis models and generates ensemble predictions for CKD risk assessment.

## Prerequisites

### Required Files
1. **Model Configuration CSV**: `results/final_deploy/model_config/model_config.csv`
2. **Model JSON Files**: In `results/final_deploy/model_config/` directory
   - `model{i}_details_*.json` - Model hyperparameters and architecture
   - `model{i}_optimization_metrics_*.json` - Model performance metrics
3. **Model Weights**: Referenced in the JSON files (usually in `models/` directory)
4. **Data Files**: Raw data files in `data/` directory

### Dependencies
- PyTorch
- PyCox
- ZenML
- NumPy, Pandas, H5py
- Other dependencies from the main project

## Running the Pipeline

### Basic Usage
```bash
python pipelines/final_deploy.py
```

### Using ZenML
```python
from pipelines.final_deploy import multi_model_deploy_pipeline

# Create and run the pipeline
pipeline = multi_model_deploy_pipeline()
pipeline.run()
```

## Pipeline Steps

1. **Data Ingestion**: Loads raw data files
2. **Data Cleaning**: Cleans and preprocesses data
3. **Data Merging**: Combines all data sources
4. **Data Splitting**: Creates train/temporal test/spatial test sets
5. **Data Imputation**: Handles missing values
6. **Data Preprocessing**: Final preprocessing for model input
7. **Multi-Model Deployment**: 
   - Loads all model configurations
   - Creates and loads each model
   - Generates predictions
   - Processes predictions (CIF conversion, time extraction)
   - Creates ensemble predictions
   - Saves results

## Output Structure

```
results/final_deploy/
├── individual_predictions/
│   ├── model1_spatial_predictions_[timestamp].h5
│   ├── model1_temporal_predictions_[timestamp].h5
│   ├── model1_spatial_metadata_[timestamp].csv
│   ├── model1_temporal_metadata_[timestamp].csv
│   └── ... (for all 36 models)
├── ensemble_predictions/
│   ├── ensemble_spatial_predictions_[timestamp].h5
│   ├── ensemble_temporal_predictions_[timestamp].h5
│   ├── ensemble_spatial_metadata_[timestamp].csv
│   ├── ensemble_temporal_metadata_[timestamp].csv
│   └── deployment_log_[timestamp].json
```

## Understanding the Predictions

### Individual Model Predictions
- **DeepSurv Models**: CIF predictions at 5 time points (365, 730, 1095, 1460, 1825 days)
  - Shape: (5, n_samples)
  - Values: Cumulative incidence (risk) probabilities [0, 1]
  
- **DeepHit Models**: Competing risks CIF predictions
  - Shape: (2, 5, n_samples)
  - Dimension 0: Endpoints (0: RRT/eGFR<15, 1: Mortality)
  - Dimension 1: Time points (365, 730, 1095, 1460, 1825 days)
  - Dimension 2: Samples

### Ensemble Predictions
- Shape: (2, 5, n_samples)
- Averaged predictions across all models
- Format matches DeepHit competing risks structure

## Configuration Options

### In `deploy_multiple_models` step:
- `config_csv_path`: Path to model configuration CSV
- `config_dir`: Directory containing model JSON files
- `master_df_mapping_path`: Path to feature mapping YAML
- `ensemble_method`: 'average' (default) or 'weighted'
- `batch_size`: Batch size for prediction processing (default: 1000)

## Troubleshooting

### Common Issues

1. **Missing Model Files**
   - Error: `FileNotFoundError: Model weights not found`
   - Solution: Ensure all model weight files exist in the paths specified in JSON configs

2. **Memory Issues**
   - Error: `CUDA out of memory`
   - Solution: Reduce `batch_size` or use CPU by setting `CUDA_VISIBLE_DEVICES=""`

3. **Incomplete Model Groups**
   - Warning: `Incomplete group (algorithm, structure, balancing, optimization)`
   - Solution: Ensure both Event 1 and Event 2 models exist for each DeepSurv configuration

4. **Shape Mismatches**
   - Error: Shape mismatch in predictions
   - Solution: Check that all models were trained with the same feature set

## Extending the Pipeline

### Adding New Ensemble Methods
Edit the `ensemble_predictions` function to add new methods:

```python
def ensemble_predictions(all_predictions, method='average', weights=None):
    if method == 'average':
        # Existing implementation
    elif method == 'weighted':
        # Weighted averaging
    elif method == 'voting':
        # Add voting logic
    elif method == 'stacking':
        # Add stacking logic
```

### Processing Subset of Models
Modify the model configuration loading to filter models:

```python
# Example: Only process ANN models
model_configs = [c for c in model_configs if c['structure'] == 'ANN']
```

## Performance Considerations

1. **GPU Usage**: The pipeline automatically uses GPU if available
2. **Memory Management**: Models are loaded and processed one at a time
3. **Batch Processing**: Predictions are generated in batches to manage memory
4. **Cleanup**: GPU memory is cleared after each model

## Validation

After running the pipeline:

1. Check the deployment log for the number of models processed
2. Verify prediction shapes in the HDF5 files
3. Compare individual model predictions with ensemble
4. Validate that all expected output files were created

## Example Analysis Script

```python
import h5py
import pandas as pd
import numpy as np

# Load ensemble predictions
with h5py.File('results/final_deploy/ensemble_predictions/ensemble_temporal_predictions_[timestamp].h5', 'r') as f:
    predictions = f['predictions'][:]
    time_grid = f['time_grid'][:]

# Load metadata
metadata = pd.read_csv('results/final_deploy/ensemble_predictions/ensemble_temporal_metadata_[timestamp].csv')

# Analyze predictions
print(f"Predictions shape: {predictions.shape}")
print(f"Time points: {time_grid}")
print(f"Number of samples: {len(metadata)}")

# Get 5-year risk for RRT/eGFR<15 (Event 1)
five_year_risk_event1 = predictions[0, -1, :]  # Last time point (1825 days)
print(f"5-year RRT risk: mean={five_year_risk_event1.mean():.3f}, std={five_year_risk_event1.std():.3f}")

# Get 5-year risk for Mortality (Event 2)
five_year_risk_event2 = predictions[1, -1, :]
print(f"5-year mortality risk: mean={five_year_risk_event2.mean():.3f}, std={five_year_risk_event2.std():.3f}")