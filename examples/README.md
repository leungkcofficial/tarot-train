# Fine-Gray Competing Risk Model Examples

This directory contains example scripts demonstrating how to use the `r_fine_gray.py` module for competing risk analysis in CKD patients.

## Prerequisites

Before running these examples, make sure you have:

1. Python 3.7 or higher
2. R 4.2 or higher
3. Required Python packages:
   - pandas
   - numpy
   - matplotlib
   - rpy2
4. Required R packages:
   - fastcmprsk
   - ggplot2
   - jsonlite
   - tools

You can install the required R packages in R with:

```R
install.packages(c("fastcmprsk", "ggplot2", "jsonlite"))
```

And the required Python packages with:

```bash
pip install pandas numpy matplotlib rpy2
```

## Example Scripts

### 1. Simple Train and Predict Example

**File:** `simple_train_predict_example.py`

This script demonstrates how to:
- Fit Fine-Gray models on `train_df`
- Make predictions on `spatial_test_df` and `temporal_test_df`
- Compare risks across datasets

This is the simplest example and is recommended for most users. It provides a function `run_fine_gray_analysis()` that takes your dataframes as input and returns the results.

```python
from examples.simple_train_predict_example import run_fine_gray_analysis

# Assuming you have train_df, spatial_test_df, and temporal_test_df loaded
results = run_fine_gray_analysis(train_df, spatial_test_df, temporal_test_df)
```

### 2. Train and Predict Example with ZenML

**File:** `train_and_predict_example.py`

This script demonstrates how to use the module with ZenML for artifact management. It's useful if you're using ZenML in your pipeline.

### 3. Jupyter Notebook Example

**File:** `fine_gray_notebook_example.ipynb`

This Jupyter notebook provides a step-by-step walkthrough of the Fine-Gray analysis process, with explanations and visualizations. It's useful for interactive exploration and understanding the workflow.

## Data Format

All examples expect dataframes with the following structure:

```
key       date   icd10        dob  gender  endpoint  endpoint_date  first_sub_60_date  hemoglobin  calcium  duration  observation_period  age  age_at_obs  creatinine  a1c  albumin  phosphate  upcr  uacr  bicarbonate  cci_score_total  ht
```

The key columns are:
- `duration`: Time to event or censoring (days)
- `endpoint`: Event indicator (0 = censored, 1 = dialysis, 2 = death)
- Feature columns as specified in `src/default_master_df_mapping.yml`

## Usage in Training Pipeline

To integrate with your training pipeline, you can import the `run_baseline_cif` function directly:

```python
from src.r_fine_gray import run_baseline_cif

# In your pipeline
def my_pipeline_step(train_df, ...):
    # Previous pipeline steps
    # ...
    
    # Run Fine-Gray analysis
    fine_gray_results = run_baseline_cif(
        df=train_df,
        feature_cols=selected_features,  # Optional, uses default from YAML if not provided
        output_path="./output",
        seed=42
    )
    
    # Continue with pipeline
    # ...
```

## Troubleshooting

If you encounter errors related to R or rpy2, make sure:
1. R is installed and in your PATH
2. The required R packages are installed
3. rpy2 is installed and compatible with your R version

### Common Errors and Solutions

1. **Error: `Error in Crisk(duration, endpoint, failcode=1, cencode=0, data=rdf)`**
   - This error occurs because the `Crisk()` function doesn't accept a `data` parameter.
   - Solution: This is fixed in the latest version of the module by extracting columns from the dataframe first.

2. **Error: `Error in cr$time : $ operator is invalid for atomic vectors`**
   - This error occurs because the `Crisk()` function is returning an atomic vector instead of a list or data frame.
   - Solution: This is fixed in the latest version of the module by checking if `cr` is an atomic vector before accessing its elements.

3. **Error: `Error in fastCrr(cr, feature_df, B = 200) : unused argument (B = 200)`**
   - This error occurs because the installed version of the fastcmprsk package doesn't support the `B` parameter for bootstrap variance.
   - Solution: This is fixed in the latest version of the module by checking if the `fastCrr()` function supports the `B` parameter before using it.

4. **Error: `Error in formula.default(object, env = baseenv()) : invalid formula`**
   - This error occurs because the `fastCrr()` function expects a formula as its first argument, not a `Crisk` object.
   - Solution: This is fixed in the latest version of the module by creating a proper formula for the `fastCrr()` function.

5. **Error: `Failed to fit Fine-Gray model for event 1`**
   - This error can occur when the dataset is very large or when there are convergence issues with the model.
   - Solution: This is addressed in the latest version of the module by:
     - Always sampling the data to a manageable size (5,000 rows) with stratification by status
     - Adding a timeout mechanism to prevent the model fitting from running indefinitely
     - Providing more informative error messages with traceback information
     - Enhanced logging of model fitting progress

If you encounter other errors, please check the logs for detailed error messages and report them to the development team.

## Performance Considerations

The Fine-Gray model fitting can be computationally intensive, especially with large datasets. The module includes several optimizations, but **for very large datasets (>100,000 rows), we strongly recommend pre-sampling your data before passing it to the module**.

### Recommended Approach for Large Datasets

```python
import pandas as pd
import numpy as np

def stratified_sample(df, target_size=5000, random_state=42):
    """
    Create a stratified sample of the dataframe based on the endpoint column.
    
    Args:
        df: Input dataframe
        target_size: Target sample size
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled dataframe
    """
    # Get endpoint distribution
    endpoint_counts = df['endpoint'].value_counts()
    
    # Calculate sampling fraction
    fraction = min(1.0, target_size / len(df))
    
    # Sample with stratification
    sampled_indices = []
    for endpoint_value, count in endpoint_counts.items():
        # Get indices for this endpoint value
        indices = df[df['endpoint'] == endpoint_value].index
        
        # Calculate how many to sample
        n_sample = min(len(indices), int(count * fraction))
        
        # Sample indices
        if n_sample > 0:
            sampled = np.random.choice(indices, n_sample, replace=False)
            sampled_indices.extend(sampled)
    
    # Return sampled dataframe
    return df.loc[sampled_indices]

# Example usage
np.random.seed(42)
train_df_sampled = stratified_sample(train_df, target_size=5000)
spatial_test_df_sampled = stratified_sample(spatial_test_df, target_size=5000)
temporal_test_df_sampled = stratified_sample(temporal_test_df, target_size=5000)

# Use sampled dataframes with the module
from examples.simple_train_predict_example import run_fine_gray_analysis
results = run_fine_gray_analysis(train_df_sampled, spatial_test_df_sampled, temporal_test_df_sampled)
```

### Other Performance Optimizations

1. **Timeout Mechanism**: A 10-minute timeout prevents the model from running indefinitely.

2. **Thread Control**: The module respects environment variables for thread control and allows explicit thread count specification.

3. **Progress Logging**: Detailed progress logging helps track the model fitting process.

For production use with very large datasets, consider using a more powerful machine with additional computational resources.