# CKD Risk Prediction - Preprocessing Pipeline

This preprocessing pipeline transforms raw patient data into the format required by the CKD risk prediction models. It handles missing value imputation, feature scaling, and all transformations that were applied during model training.

## 🚀 Quick Start

### 1. Extract the Preprocessing Pipeline

First, run the extraction script to create the preprocessor from your training data:

```bash
python src/extract_preprocessing_pipeline.py
```

This will create `results/final_deploy/ckd_preprocessor.pkl` containing all preprocessing parameters.

### 2. Use the Preprocessor

```python
from src.ckd_preprocessor import CKDPreprocessor

# Load preprocessor
preprocessor = CKDPreprocessor.load('results/final_deploy/ckd_preprocessor.pkl')

# Transform patient data
patient_data = {
    'creatinine': 1.5,
    'hemoglobin': 12.0,
    'albumin': 3.8,
    # ... other variables
}

preprocessed = preprocessor.transform(patient_data)
```

## 📋 Required Input Variables

The preprocessor expects these variables (missing values are automatically imputed):

**Laboratory Values (with units):**
- `creatinine` (µmol/L), `hemoglobin` (g/dL), `albumin` (g/L)
- `a1c` (%), `phosphate` (mmol/L), `calcium` (mmol/L)
- `ca_adjusted` (mmol/L), `bicarbonate` (mmol/L)
- `upcr` (g/g), `uacr` (mg/g), `egfr` (mL/min/1.73m²)

**Demographics:**
- `gender` (0=Female, 1=Male)
- `dob`, `date`, `first_sub_60_date` (dates as 'YYYY-MM-DD')

**Comorbidities:**
- `ht` (hypertension), `dm` (diabetes)
- All CCI components (see full documentation)

## 🔧 What the Preprocessor Does

1. **Imputation**: 
   - MICE imputation for lab values using XGBoost
   - Patient-specific imputation for demographics
   - Forward/backward fill for medical history

2. **Transformations**:
   - Log transformation for skewed features
   - Min-max scaling for continuous variables
   - Binary encoding for categorical variables

3. **Feature Engineering**:
   - Calculates `age_at_obs` from DOB
   - Calculates `observation_period` from dates

## 📁 Files for GitHub Release

Include these files in your GitHub repository:

```
models/
├── ckd_preprocessor.pkl          # Preprocessing pipeline
├── model_weights_*.pt             # Model weights
└── baseline_hazards_*.pkl         # Baseline hazards for survival models

src/
├── ckd_preprocessor.py            # Preprocessor class
└── nn_architectures.py            # Neural network definitions

examples/
└── predict_with_preprocessor.py   # Example usage

docs/
└── preprocessing_pipeline_usage.md # Full documentation
```

## 📊 Example: Complete Prediction Pipeline

```python
# 1. Load preprocessor
preprocessor = CKDPreprocessor.load('models/ckd_preprocessor.pkl')

# 2. Prepare patient data (with correct units!)
patient = {
    'creatinine': 186,      # µmol/L (not mg/dL)
    'hemoglobin': 11.5,     # g/dL
    'albumin': 36,          # g/L (not g/dL)
    'egfr': 35,            # mL/min/1.73m²
    # ... (see documentation for full list)
}

# 3. Preprocess
preprocessed = preprocessor.transform(patient)

# 4. Load model and predict
model = load_your_model('models/model_weights.pt')
prediction = model.predict(preprocessed)
```

## 🐛 Troubleshooting

- **Missing columns**: The preprocessor adds them automatically
- **Missing values**: Imputed using training data statistics
- **Wrong data types**: Ensure dates are strings, numbers are float/int

## 📚 Full Documentation

See [preprocessing_pipeline_usage.md](docs/preprocessing_pipeline_usage.md) for:
- Complete variable list with descriptions
- Handling missing values
- Batch processing
- Integration with different model types
- Advanced usage examples

## ⚙️ Requirements

```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
scipy>=1.6.0
xgboost>=1.3.0  # Optional but recommended for better imputation
```

## 📄 License

This preprocessing pipeline is part of the CKD Risk Prediction project.