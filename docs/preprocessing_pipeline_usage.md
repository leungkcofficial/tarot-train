# CKD Preprocessing Pipeline Documentation

This document explains how to use the CKD preprocessing pipeline to transform raw patient data into the format expected by the CKD risk prediction models.

## Overview

The CKD preprocessing pipeline handles all data transformations that were applied during model training, including:
- Missing value imputation (using MICE for lab values)
- Log transformation of skewed features
- Min-max scaling
- Feature engineering (age calculations, observation periods)
- Categorical variable encoding

## Installation

```bash
# Required dependencies
pip install numpy pandas scikit-learn scipy xgboost
```

## Quick Start

### 1. Download Required Files

Download these files from the GitHub repository:
- `ckd_preprocessor.pkl` - The fitted preprocessing pipeline
- Model weights (`.pt` files) from `models/` directory
- Baseline hazards (`.pkl` files) from `models/` directory

### 2. Basic Usage

```python
from src.ckd_preprocessor import CKDPreprocessor

# Load the preprocessor
preprocessor = CKDPreprocessor.load('ckd_preprocessor.pkl')

# Prepare patient data
patient_data = {
    # Demographics
    'gender': 1,  # 1=Male, 0=Female
    'dob': '1960-01-01',
    'date': '2024-01-15',
    
    # Laboratory values
    'creatinine': 1.5,      # mg/dL
    'hemoglobin': 12.0,     # g/dL
    'albumin': 3.8,         # g/dL
    'egfr': 45,             # mL/min/1.73m²
    
    # Comorbidities
    'ht': 1,  # Hypertension (1=Yes, 0=No)
    'dm': 0,  # Diabetes (1=Yes, 0=No)
    
    # Add other required variables...
}

# Transform the data
preprocessed = preprocessor.transform(patient_data)

# Now use preprocessed data with your models
# model.predict(preprocessed)
```

## Input Variables

The preprocessor expects the following input variables:

### Core Laboratory Values (11 variables)
| Variable | Description | Unit | Normal Range |
|----------|-------------|------|--------------|
| `creatinine` | Serum creatinine | **µmol/L** | 50-120 |
| `hemoglobin` | Hemoglobin | **g/dL** | 12-16 (F), 14-18 (M) |
| `albumin` | Serum albumin | **g/L** | 35-50 |
| `a1c` | Hemoglobin A1c | **%** | <5.7 |
| `phosphate` | Serum phosphate | **mmol/L** | 0.8-1.5 |
| `calcium` | Serum calcium | **mmol/L** | 2.1-2.6 |
| `ca_adjusted` | Adjusted calcium | **mmol/L** | 2.1-2.6 |
| `bicarbonate` | Serum bicarbonate | **mmol/L** | 22-28 |
| `upcr` | Urine protein-creatinine ratio | **g/g** | <0.2 |
| `uacr` | Urine albumin-creatinine ratio | **mg/g** | <30 |
| `egfr` | Estimated GFR | **mL/min/1.73m²** | >90 |

### Demographics
| Variable | Description | Values |
|----------|-------------|--------|
| `key` | Patient identifier | String |
| `gender` | Patient gender | 0=Female, 1=Male |
| `dob` | Date of birth | YYYY-MM-DD |
| `date` | Observation date | YYYY-MM-DD |
| `first_sub_60_date` | Date when eGFR first <60 | YYYY-MM-DD |

### Comorbidities
| Variable | Description | Values |
|----------|-------------|--------|
| `ht` | Hypertension | 0=No, 1=Yes |
| `dm` | Diabetes mellitus | 0=No, 1=Yes |

### Charlson Comorbidity Index (CCI) Components
All CCI components are binary (0=No, 1=Yes):
- `myocardial_infarction`
- `congestive_heart_failure`
- `peripheral_vascular_disease`
- `cerebrovascular_disease`
- `dementia`
- `chronic_pulmonary_disease`
- `rheumatic_disease`
- `peptic_ulcer_disease`
- `mild_liver_disease`
- `diabetes_wo_complication`
- `renal_mild_moderate`
- `diabetes_w_complication`
- `hemiplegia_paraplegia`
- `any_malignancy`
- `liver_severe`
- `renal_severe`
- `hiv`
- `metastatic_cancer`
- `aids`
- `cci_score_total` (Total CCI score)

## Handling Missing Values

The preprocessor automatically handles missing values:

1. **Laboratory values**: Uses MICE (Multiple Imputation by Chained Equations) with XGBoost
2. **Demographics**: Uses patient-specific historical values when available
3. **Comorbidities**: Forward/backward fills within patient history

You can pass `None` or `np.nan` for missing values:

```python
patient_data = {
    'creatinine': 1.5,
    'albumin': None,  # Will be imputed
    'uacr': np.nan,   # Will be imputed
    # ...
}
```

## Complete Example

```python
import numpy as np
import pandas as pd
from src.ckd_preprocessor import CKDPreprocessor

# Load preprocessor
preprocessor = CKDPreprocessor.load('ckd_preprocessor.pkl')

# Create patient data with some missing values
patient = {
    # Patient identifiers
    'key': 'PATIENT001',
    'date': '2024-01-15',
    'dob': '1955-03-20',
    'gender': 1,
    'first_sub_60_date': '2020-06-01',
    
    # Lab values (some missing)
    'creatinine': 186,      # µmol/L
    'hemoglobin': 11.5,     # g/dL
    'albumin': None,        # g/L (Missing - will be imputed)
    'a1c': 7.2,            # %
    'phosphate': 1.3,       # mmol/L
    'calcium': 2.3,         # mmol/L
    'ca_adjusted': None,    # mmol/L (Missing - will be imputed)
    'bicarbonate': 22,      # mmol/L
    'upcr': 1.8,           # g/g
    'uacr': None,          # mg/g (Missing - will be imputed)
    'egfr': 35,            # mL/min/1.73m²
    
    # Comorbidities
    'ht': 1,
    'dm': 1,
    
    # CCI components (set to 0 if unknown)
    'myocardial_infarction': 0,
    'congestive_heart_failure': 1,
    'peripheral_vascular_disease': 0,
    'cerebrovascular_disease': 0,
    'dementia': 0,
    'chronic_pulmonary_disease': 0,
    'rheumatic_disease': 0,
    'peptic_ulcer_disease': 0,
    'mild_liver_disease': 0,
    'diabetes_wo_complication': 0,
    'renal_mild_moderate': 1,
    'diabetes_w_complication': 1,
    'hemiplegia_paraplegia': 0,
    'any_malignancy': 0,
    'liver_severe': 0,
    'renal_severe': 0,
    'hiv': 0,
    'metastatic_cancer': 0,
    'aids': 0,
    'cci_score_total': 3
}

# Transform data
preprocessed = preprocessor.transform(patient)

print(f"Input features: {len(patient)}")
print(f"Output features: {preprocessed.shape[1]}")
print(f"Missing values handled: {preprocessed.isna().sum().sum()} remaining")
```

## Batch Processing

To process multiple patients at once:

```python
# Create DataFrame with multiple patients
patients_df = pd.DataFrame([
    patient1_data,
    patient2_data,
    patient3_data
])

# Transform all at once
preprocessed_df = preprocessor.transform(patients_df)
```

## Integration with Models

After preprocessing, use the transformed data with your models:

```python
# Example with DeepSurv model
import torch
from pycox.models import DeepSurv

# Load model (example)
model = DeepSurv.load_model('model_weights.pt')

# Make predictions
X = preprocessed.values
survival_probs = model.predict_surv(X, time_points=[365, 730, 1095])
```

## Troubleshooting

### Common Issues

1. **Missing required columns**
   ```python
   # The preprocessor will add missing columns with NaN
   # These will be imputed based on training data statistics
   ```

2. **Different column order**
   ```python
   # The preprocessor handles column ordering automatically
   ```

3. **Data type issues**
   ```python
   # Ensure dates are strings in 'YYYY-MM-DD' format
   # Numeric values should be float or int
   ```

### Debugging

Check preprocessing information:

```python
# Get preprocessor info
info = preprocessor.get_preprocessing_info()
print(f"Total features expected: {info['n_features']}")
print(f"MICE imputer fitted: {info['imputation']['mice_fitted']}")
print(f"Log transformed columns: {info['transformations']['n_log_transformed']}")
print(f"MinMax scaled columns: {info['transformations']['n_minmax_scaled']}")
```

## Advanced Usage

### Custom Imputation

If you want to override the automatic imputation:

```python
# Preprocess without certain columns
patient_data_partial = {k: v for k, v in patient_data.items() 
                       if k not in ['albumin', 'uacr']}

# Transform
preprocessed = preprocessor.transform(patient_data_partial)

# Manually set values after transformation if needed
# (Not recommended - use automatic imputation when possible)
```

### Feature Names

Get the list of features after preprocessing:

```python
feature_names = preprocessor.get_feature_names()
print(f"Features: {feature_names}")
```

## File Structure

When deploying, ensure you have:

```
your_project/
├── src/
│   └── ckd_preprocessor.py
├── models/
│   ├── ckd_preprocessor.pkl
│   ├── model_weights_*.pt
│   └── baseline_hazards_*.pkl
└── your_prediction_script.py
```

## Support

For issues or questions:
1. Check that all required dependencies are installed
2. Ensure input data matches the expected format
3. Verify that the preprocessor file is not corrupted
4. Check the model compatibility with the preprocessed features

## Version Information

The preprocessor includes version information:

```python
preprocessor = CKDPreprocessor.load('ckd_preprocessor.pkl')
# Version and creation date are stored in the file
```

This ensures compatibility between the preprocessing pipeline and trained models.