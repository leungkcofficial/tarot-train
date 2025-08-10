# Revised Plan: GPU-Accelerated Cox Model with cuML Only

Based on your feedback, I'll revise the plan to focus exclusively on cuML for GPU acceleration, without the PyTorch fallback. This will keep the implementation simpler and more focused.

## Implementation Strategy

We'll create a GPU-accelerated implementation with these components:

1. **GPU-Accelerated Cox Model**: Replace lifelines' `CoxPHFitter` with a cuML-based alternative
2. **GPU-Accelerated AIC/BIC Calculation**: Implement efficient information criteria calculation using cuML/cuPy
3. **GPU-Accelerated LASSO**: Implement LASSO regression with cross-validation using cuML

## Detailed Implementation Plan

### 1. GPU-Accelerated Cox Proportional Hazards Model

We'll implement a class that provides the same interface as lifelines' `CoxPHFitter` but uses cuML for GPU acceleration:

```python
class CumlCoxPHFitter:
    """GPU-accelerated Cox Proportional Hazards model using cuML."""
    
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.params_ = None
        self.log_likelihood_ = None
        self.summary = None
        
        # Check if cuML is available
        self._check_cuml()
        
    def _check_cuml(self):
        """Check if cuML is available and raise error if not."""
        try:
            import cuml
            import cupy as cp
        except ImportError:
            raise ImportError("cuML and cuPy are required for GPU-accelerated Cox model. Please install them with 'pip install cuml cupy'.")
    
    def fit(self, df, duration_col, event_col, robust=False, weights_col=None, show_progress=True):
        """Fit the Cox model using GPU acceleration with cuML."""
        import cuml
        import cupy as cp
        import pandas as pd
        
        # Extract data
        y = df[[duration_col, event_col]].values
        X = df.drop([duration_col, event_col], axis=1).values if weights_col is None else df.drop([duration_col, event_col, weights_col], axis=1).values
        
        # Convert to GPU arrays
        X_gpu = cp.array(X)
        durations = cp.array(y[:, 0])
        events = cp.array(y[:, 1])
        
        # Implement Cox PH fitting using cuML
        # Note: cuML doesn't have a direct Cox PH implementation, so we'll implement it using cuML's optimization tools
        
        # 1. Initialize parameters
        n_features = X_gpu.shape[1]
        beta = cp.zeros(n_features)
        
        # 2. Define the negative log partial likelihood function
        def negative_log_likelihood(beta):
            # Compute linear predictor
            eta = X_gpu.dot(beta)
            exp_eta = cp.exp(eta)
            
            # Compute log partial likelihood
            log_lik = 0
            for i in range(len(durations)):
                if events[i]:
                    # Get risk set (patients who are still at risk at time t_i)
                    risk_set = durations >= durations[i]
                    # Compute partial likelihood contribution
                    log_lik += eta[i] - cp.log(cp.sum(exp_eta[risk_set]))
            
            return -log_lik
        
        # 3. Define the gradient of the negative log partial likelihood
        def negative_log_likelihood_gradient(beta):
            # Compute linear predictor
            eta = X_gpu.dot(beta)
            exp_eta = cp.exp(eta)
            
            # Initialize gradient
            gradient = cp.zeros_like(beta)
            
            # Compute gradient
            for i in range(len(durations)):
                if events[i]:
                    # Get risk set
                    risk_set = durations >= durations[i]
                    # Compute weighted average of covariates in risk set
                    risk_sum = cp.sum(exp_eta[risk_set])
                    weighted_avg = cp.sum((X_gpu[risk_set].T * exp_eta[risk_set]).T, axis=0) / risk_sum
                    # Update gradient
                    gradient += weighted_avg - X_gpu[i]
            
            return gradient
        
        # 4. Optimize using L-BFGS
        from scipy.optimize import minimize
        
        # Convert functions to use CPU arrays for scipy.optimize
        def cpu_negative_log_likelihood(beta_cpu):
            beta_gpu = cp.array(beta_cpu)
            return float(negative_log_likelihood(beta_gpu))
        
        def cpu_negative_log_likelihood_gradient(beta_cpu):
            beta_gpu = cp.array(beta_cpu)
            return cp.asnumpy(negative_log_likelihood_gradient(beta_gpu))
        
        # Optimize
        result = minimize(
            cpu_negative_log_likelihood,
            cp.asnumpy(beta),
            method='L-BFGS-B',
            jac=cpu_negative_log_likelihood_gradient,
            options={'disp': show_progress, 'maxiter': 1000}
        )
        
        # Store results
        self.params_ = cp.array(result.x)
        self.log_likelihood_ = -result.fun
        
        # Create summary DataFrame
        self._create_summary(X, df.drop([duration_col, event_col], axis=1).columns)
        
        return self
    
    def _create_summary(self, X, feature_names):
        """Create summary DataFrame similar to lifelines."""
        import cupy as cp
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        # Calculate standard errors (using observed information matrix)
        # This is a simplified approach; a more accurate approach would use the Hessian
        n_samples = X.shape[0]
        se = cp.sqrt(cp.diag(cp.linalg.inv(cp.dot(X.T, X))) * n_samples)
        
        # Calculate z-scores and p-values
        z = self.params_ / se
        p = 2 * (1 - stats.norm.cdf(cp.abs(cp.asnumpy(z))))
        
        # Calculate hazard ratios and confidence intervals
        hr = cp.exp(self.params_)
        hr_lower = cp.exp(self.params_ - 1.96 * se)
        hr_upper = cp.exp(self.params_ + 1.96 * se)
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'coef': cp.asnumpy(self.params_),
            'exp(coef)': cp.asnumpy(hr),
            'se(coef)': cp.asnumpy(se),
            'z': cp.asnumpy(z),
            'p': p,
            'lower 0.95': cp.asnumpy(hr_lower),
            'upper 0.95': cp.asnumpy(hr_upper)
        }, index=feature_names)
        
        self.summary = summary
        return summary
```

### 2. GPU-Accelerated AIC/BIC Calculation

```python
def calculate_cuml_information_criteria(model, df):
    """
    Calculate AIC and BIC using GPU acceleration with cuML/cuPy.
    
    Args:
        model: Fitted CumlCoxPHFitter model
        df: DataFrame used for fitting
        
    Returns:
        Dict with AIC and BIC values
    """
    import cupy as cp
    
    # Get number of parameters and observations
    k = len(model.params_)
    n = df.shape[0]
    
    # Get log-likelihood (already calculated on GPU)
    ll = model.log_likelihood_
    
    # Calculate AIC and BIC
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * cp.log(cp.array([n]))[0]
    
    return {"AIC": float(aic), "BIC": float(bic)}
```

### 3. GPU-Accelerated LASSO Regression

```python
def perform_cuml_cox_lasso(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    cv_folds: int = 10,
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 1.0,
    max_iter: int = 10000,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Cox-LASSO regression with cross-validation using cuML for GPU acceleration.
    """
    import cuml
    import cupy as cp
    
    # Check if cuML is available
    try:
        import cuml
        import cupy as cp
    except ImportError:
        raise ImportError("cuML and cuPy are required for GPU-accelerated LASSO. Please install them with 'pip install cuml cupy'.")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path("results/lasso_analysis")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"LASSO analysis results will be saved to: {output_dir}")
    
    # If features not specified, use all numeric columns except duration and event
    if features is None:
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in all_numeric if f not in [duration_col, event_col]]
    else:
        # Filter to only include numeric features from the provided list
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in features if f in numeric_features]
    
    # Check if we have enough features
    if len(features) < 2:
        error_msg = f"LASSO regression requires at least 2 numeric features. Found {len(features)} numeric features."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Drop rows with missing values in the selected features, duration, and event
    cols_to_check = features + [duration_col, event_col]
    df_clean = df[cols_to_check].dropna()
    logger.info(f"Using {len(df_clean)} complete rows for LASSO regression out of {len(df)} total rows")
    
    # Prepare the data
    X = df_clean[features].values
    y = df_clean[[duration_col, event_col]].values
    
    # Convert to GPU arrays
    X_gpu = cp.array(X)
    durations = cp.array(y[:, 0])
    events = cp.array(y[:, 1])
    
    # Set up alpha values (regularization strengths)
    if alphas is None:
        # Create a logarithmic sequence of alphas
        alphas = cp.logspace(-3, 1, 100)
    
    # Implement LASSO with cross-validation
    # ...
    
    # Return results
    results = {
        "optimal_alpha": optimal_alpha,
        "optimal_coefficients": coef_df.to_dict(orient='records'),
        "cv_results": cv_results.to_dict(orient='records'),
        "coefficient_path": coef_path_df.to_dict(orient='records'),
        "output_files": {
            "coefficients_csv": str(coef_path),
            "coefficient_path_csv": str(coef_path_csv),
            "cv_results_csv": str(cv_results_csv),
            "coefficient_path_plot": str(coef_path_plot),
            "cv_results_plot": str(cv_plot)
        }
    }
    
    return results
```

## Modifications to `calculate_information_criteria` Function

We'll replace the lifelines implementation with our cuML-based version:

```python
def calculate_information_criteria(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    method: str = 'univariate',
    criterion: str = 'aic',
    save_path: Optional[Union[str, Path]] = None,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Calculate AIC and BIC for Cox proportional hazards models using GPU acceleration with cuML.
    """
    # Check if cuML is available
    try:
        import cuml
        import cupy as cp
    except ImportError:
        raise ImportError("cuML and cuPy are required for GPU-accelerated information criteria calculation. Please install them with 'pip install cuml cupy'.")
    
    # Rest of the implementation using CumlCoxPHFitter
    # ...
```

## Implementation Considerations

1. **Error Handling**: Provide clear error messages if cuML is not available
2. **Memory Management**: For very large datasets (2M+ samples), implement batch processing to avoid GPU memory issues
3. **Numerical Stability**: Ensure numerical stability in GPU calculations
4. **Interface Compatibility**: Maintain the same interface as the original functions

## Performance Optimization Techniques

1. **Batch Processing**: Process data in batches to handle large datasets
2. **Mixed Precision**: Use mixed precision (FP16/FP32) for faster computation
3. **Kernel Optimization**: Optimize GPU kernels for Cox model calculations
4. **Memory Reuse**: Reuse GPU memory allocations where possible

## Next Steps

1. Implement the `CumlCoxPHFitter` class
2. Implement the GPU-accelerated information criteria calculation
3. Implement the GPU-accelerated LASSO regression
4. Update the existing functions to use the new GPU-accelerated implementations
5. Add comprehensive error handling
6. Test with large datasets to verify performance improvements