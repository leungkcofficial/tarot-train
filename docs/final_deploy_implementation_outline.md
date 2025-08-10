# Implementation Outline for Multi-Model Deployment Pipeline

## Core Functions to Implement

### 1. Model Configuration Loading

```python
def load_all_model_configurations(config_csv_path, config_dir):
    """
    Load model configurations from CSV and corresponding JSON files.
    
    Returns:
        List of dicts with model metadata and configurations
    """
    # Read CSV file
    # For each row, load corresponding JSON files
    # Return combined information
```

### 2. Model Creation and Loading

```python
def create_model_from_config(model_config, model_details, device='cpu'):
    """
    Create and load a model based on configuration.
    
    Handles:
    - DeepSurv (binary endpoint)
    - DeepHit (competing risks)
    - ANN and LSTM architectures
    """
    # Extract model parameters
    # Create network architecture
    # Load saved weights
    # Return initialized model
```

### 3. Prediction Generation

```python
def generate_model_predictions(model, test_data, model_type, endpoint=None, batch_size=1000):
    """
    Generate predictions handling different model types.
    
    Returns:
    - DeepSurv: (1825, n_samples) survival probabilities
    - DeepHit: (2, 5, n_samples) CIF predictions
    """
    # Process data in batches
    # Handle different prediction methods
    # Return predictions in correct format
```

### 4. CIF Conversion for DeepSurv

```python
def convert_survival_to_cif(survival_probs):
    """
    Convert survival probabilities to cumulative incidence function.
    
    CIF = 1 - survival_probability
    """
    # Ensure valid range [0, 1]
    # Convert to CIF
    # Return CIF array
```

### 5. Time Point Extraction

```python
def extract_time_points(predictions, time_grid, target_times=[365, 730, 1095, 1460, 1825]):
    """
    Extract predictions at specific time points.
    
    For DeepSurv: Extract from 1825 time points
    For DeepHit: Already at correct time points
    """
    # Find indices of target times
    # Extract predictions
    # Return (5, n_samples) array
```

### 6. Prediction Stacking

```python
def stack_deepsurv_predictions(predictions_dict, model_configs):
    """
    Stack DeepSurv predictions by grouping criteria.
    
    Groups by: Algorithm, Structure, Balancing Method, Optimization Target
    Stacks Event 1 and Event 2 predictions
    
    Returns: List of (2, 5, n_samples) arrays
    """
    # Group models by criteria
    # Stack Event 1 and Event 2 for each group
    # Return stacked predictions
```

### 7. Ensemble Implementation

```python
def ensemble_predictions(all_predictions, method='average', weights=None):
    """
    Apply ensemble method to predictions.
    
    Input: (24, 2, 5, n_samples) array
    Output: (2, 5, n_samples) array
    
    Methods:
    - 'average': Simple mean
    - 'weighted': Weighted average
    - 'voting': Majority voting (future)
    """
    # Apply ensemble method
    # Return final predictions
```

### 8. Main Pipeline Function

```python
def deploy_multiple_models(
    config_csv_path,
    config_dir,
    train_df_preprocessed,
    temporal_test_df_preprocessed,
    spatial_test_df_preprocessed,
    ensemble_method='average'
):
    """
    Main pipeline for multi-model deployment.
    
    Steps:
    1. Load all model configurations
    2. Create and load models
    3. Generate predictions for each model
    4. Process predictions (CIF conversion, time extraction)
    5. Stack predictions appropriately
    6. Apply ensemble method
    7. Save results
    """
```

## Data Flow

### DeepSurv Models (24 total)
1. Load model → Generate predictions (1825, n_samples)
2. Convert to CIF: CIF = 1 - survival_prob
3. Extract 5 time points → (5, n_samples)
4. Group by characteristics (12 groups)
5. Stack Event 1 & 2 → (2, 5, n_samples) per group

### DeepHit Models (12 total)
1. Load model → Generate predictions (2, 5, n_samples)
2. Already in correct format (CIF)

### Ensemble Process
1. Combine all 24 model groups → (24, 2, 5, n_samples)
2. Apply ensemble method → (2, 5, n_samples)
3. Save individual and ensemble predictions

## Key Considerations

### Memory Management
- Process predictions in batches
- Clear GPU memory between models
- Save intermediate results

### Error Handling
- Validate model configurations
- Check prediction shapes
- Handle missing models gracefully
- Log all operations

### Output Structure
```
results/final_deploy/
├── individual_predictions/
│   ├── model{i}_spatial_predictions_{timestamp}.h5
│   ├── model{i}_temporal_predictions_{timestamp}.h5
│   └── model{i}_metadata_{timestamp}.json
├── ensemble_predictions/
│   ├── ensemble_spatial_predictions_{timestamp}.h5
│   ├── ensemble_temporal_predictions_{timestamp}.h5
│   └── ensemble_metadata_{timestamp}.json
└── deployment_log_{timestamp}.json
```

## Integration with Existing Code

### Reuse from existing modules:
- `steps.model_deploy.prepare_survival_dataset`
- `src.nn_architectures.create_network`
- `src.util.save_predictions_to_hdf5`
- `src.competing_risks_evaluation.save_competing_risks_predictions`
- `pycox.models.CoxPH` and `pycox.models.DeepHit`

### New utilities needed:
- CIF conversion function
- Time point extraction
- Model grouping logic
- Ensemble methods

## Testing Strategy

1. Test with subset of models first (e.g., 2-3 models)
2. Verify prediction shapes at each step
3. Compare individual vs ensemble performance
4. Validate output files and metadata