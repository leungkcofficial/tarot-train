# Ensemble vs KFRE vs Null Model Comparison

This set of scripts compares the performance of the ensemble model against KFRE (Kidney Failure Risk Equation) models and a null model baseline.

## Overview

The analysis compares:
1. **Ensemble Model**: Deep learning ensemble predictions (average of 24 models)
2. **KFRE 4-variable**: Traditional risk equation using age, sex, eGFR, and ACR
3. **KFRE 8-variable**: Extended equation with additional lab values
4. **Null Model**: Aalen-Johansen estimator as baseline

## Scripts

### 1. `extract_test_datasets.py`
Extracts and saves the test datasets from the data processing pipeline using the exact same data processing steps as the main pipeline.

**Usage:**
```bash
python extract_test_datasets.py
```

**Output:**
- `results/final_deploy/temporal_test_df_imputed.pkl` - Temporal test dataset (imputed, original clinical values)
- `results/final_deploy/spatial_test_df_imputed.pkl` - Spatial test dataset (imputed, original clinical values)
- `results/final_deploy/temporal_test_df_preprocessed.pkl` - Temporal test dataset (preprocessed, scaled values)
- `results/final_deploy/spatial_test_df_preprocessed.pkl` - Spatial test dataset (preprocessed, scaled values)
- CSV versions for inspection

**Important:**
- The **imputed** datasets contain original clinical values and are used for KFRE calculation
- The **preprocessed** datasets contain scaled values and are used for model predictions
- This script uses the ZenML pipeline steps to ensure consistency with the main training pipeline

### 2. `calculate_kfre_predictions.py`
Calculates KFRE predictions for both test datasets using the imputed data (original clinical values).

**Usage:**
```bash
python calculate_kfre_predictions.py
```

**Prerequisites:**
- Imputed test datasets must exist (run `extract_test_datasets.py` first)
- YAML mapping file at `src/default_master_df_mapping.yml`

**Note:** KFRE requires original clinical values (age, eGFR, ACR, etc.), not scaled values, so this script uses the imputed datasets.

**Output:**
- `results/kfre_predictions/temporal_kfre_predictions.csv`
- `results/kfre_predictions/spatial_kfre_predictions.csv`
- Summary statistics in JSON format

### 3. `compare_ensemble_kfre_null.py`
Main analysis script that performs the comparison.

**Usage:**
```bash
python compare_ensemble_kfre_null.py
```

**Prerequisites:**
- Ensemble CIF files at `results/full_ensemble/`
- Ground truth labels at `results/final_deploy/`
- KFRE predictions (run `calculate_kfre_predictions.py` first)

**Output:**
- `results/ensemble_kfre_comparison/comparison_results.json` - All metrics and quantile analysis
- Calibration plots for 2-year and 5-year predictions

## Analysis Details

### Metrics Calculated
1. **Brier Score**: Measures calibration of probabilistic predictions
2. **C-index**: Measures discrimination ability
3. **IPA (Index of Prediction Accuracy)**: Relative improvement over null model
   - IPA = 1 - (Brier_model / Brier_null)

### Quantile Analysis
- Predictions divided into 10 quantiles
- For each quantile:
  - Mean predicted risk (for each model)
  - Mean observed risk
  - Sample size

### Visualization
Bar plots showing:
- X-axis: Risk quantiles (Q1-Q10)
- Y-axis: Risk percentage (0-100%)
- Bars: Predicted risks from each model
- Line: Observed risk

## Running the Complete Analysis

### Option 1: Use the Runner Script (Recommended)
```bash
python run_ensemble_kfre_comparison.py
```

This script will:
- Check for prerequisites
- Extract test datasets (if needed)
- Calculate KFRE predictions (if needed)
- Run the comparison analysis
- Display a summary of results

### Option 2: Run Steps Manually
```bash
# Step 1: Extract test datasets (if not already available)
python extract_test_datasets.py

# Step 2: Calculate KFRE predictions
python calculate_kfre_predictions.py

# Step 3: Run the comparison analysis
python compare_ensemble_kfre_null.py
```

## Expected Results Structure

```
results/
├── final_deploy/
│   ├── temporal_test_df_imputed.pkl
│   ├── spatial_test_df_imputed.pkl
│   ├── temporal_test_df_preprocessed.pkl
│   ├── spatial_test_df_preprocessed.pkl
│   ├── temporal_test_labels.csv
│   └── spatial_test_labels.csv
├── kfre_predictions/
│   ├── temporal_kfre_predictions.csv
│   ├── spatial_kfre_predictions.csv
│   └── kfre_predictions_summary.json
└── ensemble_kfre_comparison/
    ├── comparison_results.json
    ├── temporal_730days_calibration.png
    ├── temporal_1825days_calibration.png
    ├── spatial_730days_calibration.png
    └── spatial_1825days_calibration.png
```

## Interpreting Results

### JSON Output Structure
```json
{
  "timestamp": "...",
  "results": {
    "temporal": {
      "730days": {
        "metrics": {
          "ensemble": {"brier_score": ..., "c_index": ..., "ipa": ...},
          "kfre_4v": {...},
          "kfre_8v": {...}
        },
        "quantile_analysis": {
          "ensemble": [
            {"quantile": 1, "predicted_risk": ..., "observed_risk": ...},
            ...
          ]
        }
      }
    }
  }
}
```

### Key Insights to Look For
1. **IPA > 0**: Model performs better than null baseline
2. **Higher C-index**: Better discrimination
3. **Lower Brier score**: Better calibration
4. **Calibration plots**: Predicted risk should align with observed risk

## Troubleshooting

### Missing Test Datasets
If you get an error about missing test datasets:
1. Run `extract_test_datasets.py` first
2. Ensure you have access to the raw data files
3. Check that all pipeline dependencies are installed

### KFRE Calculation Errors
If KFRE predictions fail:
1. Check that required columns exist in test datasets
2. Verify the YAML mapping file is correct
3. Look for missing lab values in the data

### Memory Issues
For large datasets:
1. Process datasets separately (temporal first, then spatial)
2. Reduce the number of quantiles if needed
3. Close other applications to free memory

## Dependencies

- pandas
- numpy
- h5py
- matplotlib
- seaborn
- lifelines
- scikit-learn
- scipy
- zenml (for data extraction)

Install with:
```bash
pip install pandas numpy h5py matplotlib seaborn lifelines scikit-learn scipy zenml
```

## Additional Scripts

### `run_ensemble_kfre_comparison.py`
A convenience script that runs all steps in sequence with progress tracking and error handling.

**Features:**
- Checks for prerequisite files
- Skips steps if outputs already exist (with option to regenerate)
- Provides clear progress indicators
- Summarizes results at the end

**Usage:**
```bash
python run_ensemble_kfre_comparison.py
```

## Notes

- The data extraction script uses the exact same ZenML pipeline steps as the main training pipeline to ensure consistency
- **Important**: KFRE calculations use the imputed datasets (original clinical values), not the preprocessed datasets (scaled values)
- KFRE predictions are calculated using both 4-variable and 8-variable equations
- The analysis focuses on dialysis risk (Event 1) at 2 and 5 years
- Quantile analysis divides predictions into 10 groups for calibration assessment

## Data Flow

1. **Raw Data** → `ingest_data()` → Multiple DataFrames
2. **Cleaning** → `clean_data()` → Cleaned DataFrames
3. **Merging** → `merge_data()` → Merged DataFrame
4. **Splitting** → `split_data()` → Train/Temporal Test/Spatial Test
5. **Imputation** → `impute_data()` → **Imputed datasets (used for KFRE)**
6. **Preprocessing** → `preprocess_data()` → **Preprocessed datasets (used for models)**
- matplotlib
- seaborn
- lifelines
- scikit-learn
- scipy

Install with:
```bash
pip install pandas numpy h5py matplotlib seaborn lifelines scikit-learn scipy