# Balancing Method NaN Error Fix

## Problem Description

When processing Model 1 (and other models with NaN balancing methods), the pipeline encounters the error:
```
Error processing model 1: argument of type 'float' is not iterable
```

### Root Cause

The error occurs because:
1. Some models have `NaN` (float type) as their balancing method in the CSV
2. The code tries to use the `in` operator directly on this float value:
   ```python
   if 'None' in balancing_method:  # Fails when balancing_method is float NaN
   ```
3. Python cannot iterate over a float, causing the error

### Affected Models

Models with NaN balancing methods in the CSV:
- Model 1: DeepSurv - ANN, Event 1, Balancing: nan
- And potentially others with similar configuration

## Solution

### Fixed Code (in `final_deploy_v2_fixed.py`)

The fix involves checking for NaN values before using the `in` operator:

```python
# Handle NaN balancing method - check if it's NaN or string 'nan'
if pd.isna(balancing_method) or str(balancing_method).lower() == 'nan':
    balancing_method = 'None'

# Now safe to use 'in' operator
if 'NearMiss' in str(balancing_method):
    # Process NearMiss balancing
```

### Key Changes

1. **In `prepare_data_for_model` function**:
   - Added NaN check before processing balancing method
   - Convert NaN to 'None' string for consistent handling

2. **In `stack_predictions_by_group` function**:
   - Similar NaN check for balancing when grouping models

3. **String conversion for safety**:
   - Always use `str(balancing_method)` when using `in` operator
   - This prevents errors even if NaN check is missed

## Usage

**IMPORTANT**: Use the fixed version of the pipeline:

```bash
# Correct - use the fixed version
python pipelines/final_deploy_v2_fixed.py

# NOT this (contains the bug)
python pipelines/final_deploy_v2.py
```

## Verification

Run the diagnostic script to verify the fix:
```bash
python pipelines/diagnose_pipeline_error.py
```

This will:
1. Show the error with direct `in` operator usage
2. Demonstrate the fix with NaN checking
3. Confirm which pipeline files contain the fix

## Additional Notes

- The CSV file uses "None" (string) for some models and NaN (float) for others
- The fixed code handles both cases consistently
- All balancing-related string operations now use `str()` conversion for safety