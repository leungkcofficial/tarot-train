# Baseline Hazard Computation Pipeline

## Overview

This pipeline computes baseline hazards for all 24 DeepSurv models in the `/results/final_deploy/models` directory. The baseline hazards are essential for making predictions with DeepSurv models but were not saved during the original model deployment.

## Files Created

1. **`pipelines/compute_baseline_hazard.py`** - Main ZenML pipeline that orchestrates the computation
2. **`steps/compute_all_baseline_hazards.py`** - Step that computes baseline hazards for all models
3. **`run_baseline_hazard_computation.py`** - Script to execute the pipeline
4. **`test_baseline_hazard_single_model.py`** - Test script for verifying the implementation

## How to Run

### Option 1: Run the Full Pipeline

```bash
python run_pipeline.py
```

Note: The `run_pipeline.py` has been configured to run the `compute_baseline_hazards_pipeline()`. Make sure line 16 is uncommented and other pipelines are commented out if you only want to run the baseline hazard computation.

This will:
1. Load and preprocess the training data (cached if already processed)
2. Process all 24 DeepSurv models sequentially
3. Compute baseline hazards for each model
4. Save baseline hazards as `.pkl` files in `/results/final_deploy/models/`
5. Generate a summary report

### Option 2: Test with Individual Models First

If you want to test the implementation before running the full pipeline:

1. First, you need to find the path to preprocessed training data from a previous pipeline run
2. Update the `train_df_path` variable in `test_baseline_hazard_single_model.py`
3. Run the test:

```bash
python test_baseline_hazard_single_model.py
```

## Output Files

For each model, the pipeline creates a baseline hazard file:
- **Filename format**: `baseline_hazards_model{N}_{timestamp}.pkl`
- **Location**: `/results/final_deploy/models/`

Each file contains:
- `baseline_hazards_`: The baseline hazard values
- `baseline_cumulative_hazards_`: Cumulative baseline hazards
- `model_config`: The model's configuration for reference
- `model_info`: Metadata about the model

## Summary Report

The pipeline generates a summary JSON file:
- **Filename**: `baseline_hazards_summary_{timestamp}.json`
- **Location**: `/results/final_deploy/models/`

The summary includes:
- Total models processed
- Number of successful/failed computations
- List of baseline hazard files created
- Details of any errors

## Technical Details

### Model Architecture Handling

The pipeline correctly handles both architectures:

1. **ANN Models (1-12)**:
   - Use standard feature extraction
   - Direct computation of baseline hazards

2. **LSTM Models (13-24)**:
   - Read sequence_length from model configuration (varies from 8-10)
   - Create sequences using model-specific sequence_length
   - Compute baseline hazards on sequential data

### Key Features

- **Caching**: Data processing steps are cached by ZenML for efficiency
- **Error Handling**: Continues processing even if individual models fail
- **Memory Management**: Processes models sequentially to avoid memory issues
- **GPU Support**: Automatically uses GPU if available

### Feature Columns Used

```python
feature_cols = [
    'gender', 'creatinine', 'hemoglobin', 'phosphate',
    'age_at_obs', 'bicarbonate', 'albumin',
    'uacr', 'cci_score_total', 'ht', 'observation_period'
]
```

## Troubleshooting

### Common Issues

1. **Memory Error**: If you run out of memory, the pipeline processes models sequentially to minimize memory usage. Ensure you have sufficient RAM/GPU memory.

2. **Missing Model Files**: Ensure all model files exist in `/results/final_deploy/models/`

3. **Missing Config Files**: Ensure all configuration JSON files exist in `/results/final_deploy/model_config/`

4. **Data Processing Error**: If data processing fails, check that all required data files are available

### Verification

To verify baseline hazards were computed correctly:

```python
import pickle

# Load a baseline hazard file
with open('results/final_deploy/models/baseline_hazards_model1_*.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Baseline hazards shape: {data['baseline_hazards_'].shape}")
print(f"Model info: {data['model_info']}")
```

## Next Steps

After computing baseline hazards, the models are ready for:
1. Making predictions on new data
2. Integration into the ensemble pipeline
3. Model evaluation and validation

The baseline hazard files should be kept alongside the model weight files for deployment.