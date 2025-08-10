# Feature Evaluation Enhancement Plan

## Current Understanding

The project appears to be a CKD (Chronic Kidney Disease) Risk Prediction system with:
- Feature selection and evaluation capabilities
- Survival analysis components
- XGBoost modeling with SHAP analysis
- Existing VIF (Variance Inflation Factor) calculation for multicollinearity detection

## Requested Enhancements

We need to add two new functions to `src/feature_eval.py`:

1. **AIC/BIC Calculation Function**: To calculate and report Akaike's Information Criterion and Bayesian Information Criterion for each feature
2. **LASSO Regression Function**: To implement LASSO regression with cross-validation, coefficient path tracking, and visualization

## Detailed Function Specifications

### 1. AIC/BIC Calculation Function

```python
def calculate_information_criteria(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    method: str = 'univariate',  # 'univariate', 'full', or 'stepwise'
    criterion: str = 'aic',      # 'aic', 'bic', or 'both'
    save_path: Optional[Union[str, Path]] = None,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Calculate AIC and BIC for Cox proportional hazards models.
    
    This function can:
    1. Calculate criteria for one-variable 'mini-models' (univariate screening)
    2. Calculate criteria for a full model with all features
    3. Perform stepwise/best-subset selection using AIC/BIC as stopping rule
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and target
        duration_col (str): Name of the survival duration column
        event_col (str): Name of the event indicator column
        features (List[str], optional): List of feature names to evaluate.
                                      If None, uses all numeric columns in the DataFrame.
        method (str, optional): Method for model building:
                              - 'univariate': Fit one model per feature
                              - 'full': Fit one model with all features
                              - 'stepwise': Perform stepwise selection
        criterion (str, optional): Criterion to use: 'aic', 'bic', or 'both'
        save_path (str or Path, optional): Path to save the results as CSV.
                                         If None, results are not saved.
        return_dataframe (bool, optional): If True, returns a DataFrame with results.
                                         If False, returns a dictionary. Default is True.
    
    Returns:
        pd.DataFrame or Dict: AIC and BIC values for models, sorted by criterion value.
    """
```

### 2. LASSO Regression Function

```python
def perform_cox_lasso(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    cv_folds: int = 10,
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 1.0,  # 1.0 for LASSO, between 0-1 for elastic net
    max_iter: int = 10000,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Cox-LASSO regression with cross-validation and generate visualizations.
    
    This function:
    1. Determines the optimal regularization parameter (lambda) through cross-validation
    2. Tracks coefficient changes as lambda increases
    3. Visualizes the relationship between partial likelihood, lambda, and number of features
    4. Exports raw data as CSV and visualizations as PNG
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and target
        duration_col (str): Name of the survival duration column
        event_col (str): Name of the event indicator column
        features (List[str], optional): List of feature names to use.
                                      If None, uses all numeric columns in the DataFrame.
        cv_folds (int, optional): Number of cross-validation folds. Default is 10.
        alphas (List[float], optional): List of alpha values to try.
                                      If None, uses a logarithmic sequence.
        l1_ratio (float, optional): Elastic net mixing parameter (0 <= l1_ratio <= 1).
                                  l1_ratio=1 corresponds to LASSO. Default is 1.0.
        max_iter (int, optional): Maximum number of iterations. Default is 10000.
        output_dir (str or Path, optional): Directory to save outputs.
                                          If None, uses current directory.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
        Dict[str, Any]: Dictionary containing results and paths to saved files
    """
```

## Implementation Details

### AIC/BIC Function Implementation

The function will:
1. Support Cox proportional hazards models for survival data
2. Use `lifelines` or `statsmodels` for model fitting
3. Calculate AIC and BIC for:
   - Individual feature models (univariate screening)
   - Full model with all features
   - Stepwise selection models
4. Return and optionally save results

### Cox-LASSO Implementation

The function will:
1. Use `sksurv.linear_model.CoxnetSurvivalAnalysis` or `lifelines.CoxPHFitter` with L1 penalty
2. Implement 10-fold cross-validation (user-configurable)
3. Generate visualizations:
   - Coefficient path plot showing which predictors enter/exit as λ varies
     - x-axis: log(λ)
     - y-axis: coefficient values
     - Legend sorted by entry order
   - Cross-validated deviance curve
     - x-axis: log(λ)
     - y-axis: mean CV deviance with 1 SE error bars
     - Secondary axis showing number of non-zero coefficients
4. Export raw data as CSV and visualizations as PNG

## Dependencies

The implementation will require:
- `lifelines` or `sksurv` for survival analysis
- `statsmodels` for AIC/BIC calculation
- `scikit-learn` for cross-validation
- `matplotlib` and `seaborn` for visualizations
- `pandas` and `numpy` for data handling

## Integration with Existing Code

These functions will be added to `src/feature_eval.py` and can be called from `steps/feature_selection.py` similar to how the existing VIF functions are used.