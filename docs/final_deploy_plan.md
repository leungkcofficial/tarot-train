# Multi-Model Deployment Pipeline Design

## Overview
This pipeline will load multiple pre-trained models, generate predictions from each, and ensemble them together for final predictions on spatial and temporal test datasets.

## Architecture

### 1. Model Configuration Loading
- Read `model_config.csv` to get model metadata
- Load corresponding JSON files for each model's hyperparameters
- Group models by their characteristics for proper stacking

### 2. Model Types and Prediction Shapes

#### DeepSurv Models (24 models)
- Binary endpoint models (Event 1 or Event 2)
- Raw prediction shape: (1825, n_samples) - survival probabilities
- Need to convert to CIF (Cumulative Incidence Function)
- Extract predictions at time points: [365, 730, 1095, 1460, 1825]
- Final shape per model: (5, n_samples)

#### DeepHit Models (12 models)
- Competing risks models (Both endpoints)
- Raw prediction shape: (2, 5, n_samples) - CIF for each endpoint
- Already in correct format

### 3. Stacking Strategy

#### Step 1: Group DeepSurv models
Groups based on (Algorithm, Structure, Balancing Method, Optimization Target):
- Each group will have 2 models (Event 1 and Event 2)
- Stack them to create shape (2, 5, n_samples)
- Total: 12 stacked DeepSurv groups

#### Step 2: Combine with DeepHit
- 12 stacked DeepSurv groups + 12 DeepHit models = 24 total
- Final stacked shape: (24, 2, 5, n_samples)

### 4. Ensemble Methods

#### Simple Averaging (Current)
- Average across all 24 models
- Final output: (2, 5, n_samples)

#### Future Methods (Extensible)
- Weighted averaging based on performance metrics
- Voting mechanisms
- Stacking with meta-learner

### 5. Output Structure
```
results/
├── final_deploy/
│   ├── individual_predictions/
│   │   ├── model1_spatial_predictions_[timestamp].h5
│   │   ├── model1_temporal_predictions_[timestamp].h5
│   │   ├── model2_spatial_predictions_[timestamp].h5
│   │   └── ...
│   ├── ensemble_predictions/
│   │   ├── ensemble_spatial_predictions_[timestamp].h5
│   │   ├── ensemble_temporal_predictions_[timestamp].h5
│   │   └── ensemble_metadata_[timestamp].json
│   └── deployment_log_[timestamp].json
```

## Implementation Steps

### Phase 1: Model Loading
1. Parse model_config.csv
2. Load model JSON configurations
3. Initialize models with saved weights

### Phase 2: Prediction Generation
1. Load preprocessed test datasets
2. Generate predictions for each model
3. Handle different output formats

### Phase 3: Prediction Processing
1. Convert DeepSurv predictions to CIF
2. Extract specific time points
3. Stack predictions according to grouping rules

### Phase 4: Ensemble
1. Combine all predictions into (24, 2, 5, n_samples) array
2. Apply ensemble method
3. Generate final (2, 5, n_samples) predictions

### Phase 5: Output
1. Save individual model predictions
2. Save ensemble predictions
3. Generate metadata and logs

## Key Functions

```python
def load_model_configurations(config_csv_path, config_dir):
    """Load all model configurations from CSV and JSON files"""
    
def create_model_from_config(model_config, model_details):
    """Create and load a single model from configuration"""
    
def generate_predictions(model, test_data, model_type, endpoint=None):
    """Generate predictions handling different model types"""
    
def convert_deepsurv_to_cif(survival_probs, time_points):
    """Convert survival probabilities to CIF"""
    
def extract_time_points(predictions, target_times=[365, 730, 1095, 1460, 1825]):
    """Extract predictions at specific time points"""
    
def stack_predictions(predictions_dict, model_configs):
    """Stack predictions according to grouping rules"""
    
def ensemble_predictions(stacked_predictions, method='average'):
    """Apply ensemble method to predictions"""
    
def save_predictions(predictions, metadata, output_path):
    """Save predictions and metadata"""
```

## Error Handling
- Validate model configurations before loading
- Check prediction shapes at each step
- Handle missing models gracefully
- Log all operations for debugging

## Performance Considerations
- Process models in batches to manage memory
- Use GPU when available
- Save intermediate results for recovery