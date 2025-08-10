# Baseline Hazard Computation Pipeline Plan

## Overview
We need to create a new pipeline to compute baseline hazards for all 24 DeepSurv models in the `/results/final_deploy/models` directory. The baseline hazards were not saved during the original model deployment, but they are essential for making predictions with DeepSurv models.

## Current Situation

### Models
- **Total DeepSurv Models**: 24 (Models 1-24)
  - 12 ANN models (Models 1-12)
  - 12 LSTM models (Models 13-24)
- **Events**: Each model predicts either Event 1 or Event 2
- **Model Files**: Located in `/results/final_deploy/models/`
- **Naming Pattern**: `Ensemble_model{N}_DeepSurv_{Structure}_Event_{E}_{timestamp}.pt`

### Model Configurations
- Configuration details are in `/results/final_deploy/model_config/`
- Each model has:
  - Details JSON: `model{N}_details_{timestamp}.json`
  - Optimization metrics JSON: `model{N}_optimization_metrics_{timestamp}.json`

## Pipeline Architecture

### 1. Data Processing (Reuse from ensemble_deploy_pipeline.py)
The pipeline will reuse the exact same data processing steps:
1. `ingest_data` - Load all data sources
2. `clean_data` - Clean and preprocess data
3. `merge_data` - Merge into master dataframe
4. `split_data` - Split into train/test sets
5. `impute_data` - Impute missing values
6. `preprocess_data` - Final preprocessing

### 2. New Step: compute_all_baseline_hazards
This step will:
1. Load model configurations from CSV
2. For each DeepSurv model:
   - Load the model weights
   - Determine architecture (ANN vs LSTM)
   - Prepare training data for the specific event
   - Compute baseline hazards
   - Save baseline hazards as `.pkl` file

### 3. Key Considerations

#### Model Architecture Detection
- **ANN Models**: Standard feed-forward networks
- **LSTM Models**: Require sequence data preparation
  - Each LSTM model has its own sequence_length specified in the model config
  - Sequence lengths vary (e.g., 8 or 10) and must be read from each model's configuration
  - Need to create sequences from training data using the model-specific sequence_length

#### Event-Specific Data
- Each model is trained for a specific event (1 or 2)
- Must filter training data for the correct event
- Use columns: `event{N}`, `duration{N}`

#### Feature Columns
Standard features used:
```python
feature_cols = [
    'gender', 'creatinine', 'hemoglobin', 'phosphate',
    'age_at_obs', 'bicarbonate', 'albumin',
    'uacr', 'cci_score_total', 'ht', 'observation_period'
]
```

#### Output Format
For each model, save:
- `baseline_hazards_{model_name}_{timestamp}.pkl` containing:
  - `baseline_hazards_`: The baseline hazard values
  - `baseline_cumulative_hazards_`: Cumulative baseline hazards

## Implementation Details

### Pipeline Structure
```python
@pipeline(enable_cache=True)
def compute_baseline_hazards_pipeline():
    # Data processing steps (cached)
    # ... (same as ensemble_deploy_pipeline)
    
    # Compute baseline hazards for all models
    baseline_results = compute_all_baseline_hazards(
        train_df_preprocessed=train_df_preprocessed,
        model_config_csv="results/final_deploy/model_config/model_config.csv",
        models_dir="results/final_deploy/models",
        output_dir="results/final_deploy/models"
    )
    
    return baseline_results
```

### Error Handling
- Continue processing even if one model fails
- Log detailed error messages
- Return summary of successful/failed models

### Memory Management
- Process models sequentially to avoid memory issues
- Clear GPU memory between models
- Use appropriate batch sizes for computation

## Testing Strategy
1. Test with a single ANN model first
2. Test with a single LSTM model
3. Run full pipeline for all 24 models
4. Verify baseline hazards can be loaded and used for predictions

## Expected Outputs
- 24 baseline hazard files in `/results/final_deploy/models/`
- Summary report of computation results
- Log file with any errors or warnings