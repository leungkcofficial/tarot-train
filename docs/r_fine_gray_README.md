# Baseline Competing-Risk Dashboard — CKD Dialysis & Mortality

## Overview

This module implements Fine-Gray competing risk models for CKD patients, providing risk estimates for dialysis and mortality at 1-5 year horizons. It fits models, generates visualizations, and exports model objects for reuse.

The module uses rpy2 to interface with R's fastcmprsk package for Fine-Gray modeling, providing a seamless Python interface while leveraging R's robust competing risk modeling capabilities.

## Features

- **Data Validation**: Validates input data structure and handles missing values
- **Fine-Gray Model Fitting**: Fits competing risk models for dialysis and death endpoints
- **Risk Prediction**: Predicts cumulative incidence at 1-5 year horizons
- **Visualization**: Creates bar charts showing risk percentages
- **Model Export**: Saves fitted models as RDS files for reproducibility and reuse
- **Metadata Export**: Saves model metadata including checksums and version information
- **Thread Control**: Respects environment variables and allows explicit thread count specification
- **Comprehensive Logging**: Detailed logging with progress reporting for long-running operations
- **Error Handling**: Robust error handling, particularly for R-Python interoperability

## Requirements

### Python Dependencies

- Python ≥ 3.7
- pandas
- numpy
- matplotlib
- rpy2

### R Dependencies

- R ≥ 4.2
- fastcmprsk
- ggplot2
- jsonlite
- tools

## Installation

1. Ensure R is installed on your system
2. Install required R packages:
   ```R
   install.packages(c("fastcmprsk", "ggplot2", "jsonlite"))
   ```
3. Install Python dependencies:
   ```bash
   pip install pandas numpy matplotlib rpy2
   ```

## Usage

### Pipeline Integration (Recommended)

```python
# In a pipeline context (e.g., training_pipeline.py)
from src.r_fine_gray import run_baseline_cif

# Previous pipeline steps that produce a dataframe
# ...

# Run the Fine-Gray analysis directly on the in-memory dataframe
fine_gray_results = run_baseline_cif(
    df=processed_dataframe,  # DataFrame already in memory
    feature_cols=selected_features,
    output_path=output_directory,
    seed=random_seed
)

# Continue with pipeline using the results
# ...
```

### Standalone Usage

```python
import pandas as pd
from src.r_fine_gray import run_baseline_cif

# Load your data (only needed for standalone usage)
df = pd.read_csv("your_data.csv")

# Run the analysis
results = run_baseline_cif(
    df,
    output_path="./output",
    seed=42,
    n_threads=None,  # Auto-detect
    silent=False
)

# Access the results
print(f"Dialysis risks: {results['dialysis_risks']}")
print(f"Death risks: {results['death_risks']}")
print(f"Visualization saved to: {results['visualization_path']}")
```

### Command Line Usage

```bash
python -m src.r_fine_gray --input your_data.csv --output ./output --seed 42
```

> **Note**: The command line interface is provided for convenience in standalone scenarios. For integration into pipelines, direct DataFrame usage is recommended for efficiency and better data lineage tracking.

### Loading Saved Models

```python
import pandas as pd
from src.r_fine_gray import load_and_predict

# Load your new data
new_df = pd.read_csv("new_data.csv")

# Define feature columns (must match those used in model training)
feature_cols = ['gender', 'creatinine', 'hemoglobin', 'phosphate', 'age',
                'calcium', 'bicarbonate', 'a1c', 'albumin',
                'uacr', 'cci_score_total', 'ht', 'observation_period']

# Load model and predict
predictions = load_and_predict(
    model_path="./output/models/fg_fit_dialysis.rds",
    df=new_df,
    feature_cols=feature_cols,
    time_horizons=[365, 730, 1095, 1460, 1825],  # 1-5 years
    seed=42
)

# Calculate mean risk
mean_risk = predictions.mean() * 100
print(f"Mean dialysis risk: {mean_risk}")
```

## Data Format

The input dataframe must contain the following columns:

- `duration`: Time to event or censoring (days, non-negative)
- `endpoint`: Event indicator (0 = censored, 1 = dialysis, 2 = death)
- Feature columns as specified in the mapping file or passed to the function

## Output Files

The module generates the following output files:

- `baseline_cif.png`: Bar chart visualization of risks
- `baseline_cif.csv`: CSV file with risk percentages
- `models/fg_fit_dialysis.rds`: Saved dialysis model
- `models/fg_fit_death.rds`: Saved death model
- `models/model_meta.json`: Model metadata

## Thread Control

The module respects the following environment variables for thread control:

- `OMP_NUM_THREADS`: Number of OpenMP threads
- `MKL_NUM_THREADS`: Number of MKL threads

Alternatively, you can specify the number of threads directly in the function call.

## Error Handling

The module includes comprehensive error handling, particularly for R-Python interoperability issues. Errors are logged with detailed messages to help diagnose and resolve issues.

## Performance

The module is optimized for performance, with the following benchmarks:

- 10k patients × 15 features × 200 bootstraps: ≤ 45 seconds on 16 vCPU, 32 GB RAM

## Reproducibility

The module ensures reproducibility through:

- Fixed random seed control
- Model object persistence
- Metadata including checksums
- Version information

## License

This module is part of the CKD Risk Prediction project.