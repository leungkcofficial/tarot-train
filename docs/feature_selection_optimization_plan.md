# Optimizing AIC/BIC Calculation in Feature Selection

Based on the provided feedback and "triage sheet", this document outlines a comprehensive plan to optimize the AIC/BIC calculation in the feature selection process. The current implementation is commented out because it's too slow, despite using GPU acceleration.

## Current Performance Issues

1. **One model-fit per feature**: Each call to `CoxPHFitter/CumlCoxPHFitter` launches a fresh optimizer and scans the whole risk set
2. **Pandas → GPU → Pandas shuttling**: Data is constantly moving between host and device
3. **Stepwise forward selection**: Requires fitting hundreds of Cox models
4. **Robust=TRUE and ridge penalizer=0.01**: Doubles gradient work and slows convergence
5. **10-fold CV for λ**: Already addressed, but still a consideration

## Optimization Plan

### 1. Create a Fast Univariate Screening Function

Instead of fitting full Cox models for each feature, we'll implement a fast screening approach using log-rank statistics:

```python
def fast_univariate_screening(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: List[str],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fast univariate screening using log-rank test statistics.
    
    Args:
        df: Input DataFrame
        duration_col: Name of the survival duration column
        event_col: Name of the event indicator column
        features: List of feature names to evaluate
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        DataFrame with features ranked by test statistic
    """
    from joblib import Parallel, delayed
    from lifelines.statistics import proportional_hazard_test
    
    def score_stat(col):
        # Create binary endpoint for proportional_hazard_test
        df_copy = df[[col, duration_col, event_col]].copy()
        df_copy['event_binary'] = (df_copy[event_col] > 0).astype(int)
        
        res = proportional_hazard_test(
            cph=None,  # lifelines will build an internal null model
            training_df=df_copy,
            duration_col=duration_col,
            event_col='event_binary',
            time_transform="rank"
        )
        return col, res.test_statistic, res.p_value
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(score_stat)(col) for col in features)
    
    # Create DataFrame with results
    result_df = pd.DataFrame(results, columns=['Feature', 'Chi_Square', 'P_Value'])
    result_df = result_df.sort_values('Chi_Square', ascending=False)
    
    return result_df
```

### 2. Create a Fast AIC/BIC Calculation Function

For features that pass the initial screening, we'll calculate AIC/BIC with minimal optimization:

```python
def fast_calculate_ic(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: List[str],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fast calculation of AIC/BIC for Cox models.
    
    Args:
        df: Input DataFrame
        duration_col: Name of the survival duration column
        event_col: Name of the event indicator column
        features: List of feature names to evaluate
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        DataFrame with AIC and BIC values for each feature
    """
    from joblib import Parallel, delayed
    from lifelines import CoxPHFitter
    
    def fast_single_ic(col):
        # Create binary endpoint
        df_copy = df[[col, duration_col, event_col]].copy()
        df_copy['event_binary'] = (df_copy[event_col] > 0).astype(int)
        
        # One Newton iteration to get β & log-lik
        cph = CoxPHFitter(penalizer=0.0)  # No ridge penalty for faster convergence
        cph.fit(
            df_copy[[col, duration_col, 'event_binary']], 
            duration_col=duration_col, 
            event_col='event_binary',
            robust=False,  # Turn off robust for faster computation
            show_progress=False, 
            step_size=1.0, 
            max_steps=1  # Just one step for fast approximation
        )
        
        ll = cph.log_likelihood_
        k = 1  # One coefficient
        n = df_copy.shape[0]
        
        return col, -2*ll + 2*k, -2*ll + k*np.log(n)  # Feature, AIC, BIC
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(fast_single_ic)(col) for col in features)
    
    # Create DataFrame with results
    result_df = pd.DataFrame(results, columns=['Feature', 'AIC', 'BIC'])
    
    return result_df
```

### 3. Implement a Fast Multivariable Model Function

For the final model with selected features:

```python
def fast_multivariable_model(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: List[str]
) -> Dict:
    """
    Fit a multivariable Cox model with the selected features.
    
    Args:
        df: Input DataFrame
        duration_col: Name of the survival duration column
        event_col: Name of the event indicator column
        features: List of selected feature names
        
    Returns:
        Dictionary with model results
    """
    from lifelines import CoxPHFitter
    
    # Create binary endpoint
    df_copy = df[features + [duration_col, event_col]].copy()
    df_copy['event_binary'] = (df_copy[event_col] > 0).astype(int)
    
    # Fit the final model with proper settings
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        df_copy[features + [duration_col, 'event_binary']], 
        duration_col=duration_col, 
        event_col='event_binary',
        robust=True,  # Use robust for final model
        show_progress=False
    )
    
    # Calculate AIC and BIC
    ll = cph.log_likelihood_
    k = len(features)
    n = df_copy.shape[0]
    aic = -2*ll + 2*k
    bic = -2*ll + k*np.log(n)
    
    # Get coefficients and p-values
    summary = cph.summary.reset_index()
    summary = summary.rename(columns={'index': 'Feature'})
    
    return {
        'model': cph,
        'AIC': aic,
        'BIC': bic,
        'log_likelihood': ll,
        'summary': summary
    }
```

### 4. Implement a Complete Fast Feature Selection Pipeline

```python
def fast_feature_selection(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: List[str],
    top_n: int = 10,
    output_dir: Optional[Path] = None,
    n_jobs: int = -1
) -> Dict:
    """
    Fast feature selection pipeline for Cox models.
    
    Args:
        df: Input DataFrame
        duration_col: Name of the survival duration column
        event_col: Name of the event indicator column
        features: List of feature names to evaluate
        top_n: Number of top features to select for multivariable model
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Dictionary with results
    """
    import time
    
    results = {}
    
    # 1. Fast univariate screening
    start_time = time.time()
    print("Performing fast univariate screening...")
    screening_results = fast_univariate_screening(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        features=features,
        n_jobs=n_jobs
    )
    print(f"Univariate screening completed in {time.time() - start_time:.2f} seconds")
    
    if output_dir:
        screening_path = output_dir / "fast_univariate_screening.csv"
        screening_results.to_csv(screening_path, index=False)
        print(f"Saved screening results to {screening_path}")
    
    results['screening'] = screening_results
    
    # 2. Fast AIC/BIC calculation for top features
    start_time = time.time()
    print(f"Calculating AIC/BIC for top {top_n} features...")
    top_features = screening_results['Feature'].head(top_n).tolist()
    ic_results = fast_calculate_ic(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        features=top_features,
        n_jobs=n_jobs
    )
    print(f"AIC/BIC calculation completed in {time.time() - start_time:.2f} seconds")
    
    if output_dir:
        ic_path = output_dir / "fast_aic_bic.csv"
        ic_results.to_csv(ic_path, index=False)
        print(f"Saved AIC/BIC results to {ic_path}")
    
    results['ic'] = ic_results
    
    # 3. Fit multivariable model with top features
    start_time = time.time()
    print("Fitting multivariable model with top features...")
    model_results = fast_multivariable_model(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        features=top_features
    )
    print(f"Multivariable model fitting completed in {time.time() - start_time:.2f} seconds")
    
    if output_dir:
        summary_path = output_dir / "fast_multivariable_model.csv"
        model_results['summary'].to_csv(summary_path, index=False)
        print(f"Saved model summary to {summary_path}")
    
    results['model'] = model_results
    
    return results
```

## Expected Performance Improvements

With these optimizations, we expect:

1. **Univariate screening**: Reduced from hours to seconds by using log-rank statistics instead of full model fitting
2. **AIC/BIC calculation**: Reduced from minutes to seconds by using minimal optimization and parallel processing
3. **Multivariable model**: Reduced from minutes to seconds by fitting only one model with selected features
4. **Overall**: Reduced from hours to minutes for the entire feature selection process

## Implementation Strategy

1. First, implement the new functions in `src/feature_eval.py`
2. Then, update the commented-out code in `steps/feature_selection.py` to use these functions
3. Test with a small dataset to ensure correctness
4. Run with the full dataset to verify performance improvements

## Conclusion

This optimization plan addresses all the bottlenecks identified in the triage sheet:
- Replaces full optimization with score/log-rank statistics for screening
- Eliminates data shuttling between CPU and GPU
- Replaces stepwise selection with a more efficient approach
- Turns off robust=TRUE and ridge penalizer for ranking
- Uses parallel processing for faster computation

The result should be a feature selection process that completes in minutes rather than hours, even on large datasets.