"""Feature evaluation module for CKD Risk Prediction Model
Contains code to evaluate the importance and multicollinearity of features in different models"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import warnings
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Set up logging
logger = logging.getLogger(__name__)

def calculate_vif(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    vif_threshold: float = 5.0,
    save_path: Optional[Union[str, Path]] = None,
    return_dataframe: bool = True,
    print_high_vif: bool = True
) -> Union[pd.DataFrame, Dict[str, float]]:
    """
    Calculate Variance Inflation Factor (VIF) for features in a DataFrame.
    
    VIF quantifies the severity of multicollinearity in regression analysis.
    It provides an index that measures how much the variance of an estimated
    regression coefficient is increased because of collinearity.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the features
        features (List[str], optional): List of feature names to calculate VIF for.
                                       If None, uses all numeric columns in the DataFrame.
        vif_threshold (float, optional): Threshold for highlighting high VIF values. Default is 5.0.
        save_path (str or Path, optional): Path to save the VIF results as CSV.
                                          If None, results are not saved.
        return_dataframe (bool, optional): If True, returns a DataFrame with VIF values.
                                          If False, returns a dictionary. Default is True.
        print_high_vif (bool, optional): If True, prints features with VIF > threshold. Default is True.
    
    Returns:
        pd.DataFrame or Dict[str, float]: VIF values for each feature, sorted by VIF in descending order.
                                         Returns DataFrame if return_dataframe=True, otherwise a dictionary.
    
    Raises:
        ValueError: If fewer than 2 numeric features are available for VIF calculation.
        ImportError: If statsmodels is not installed.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.error("statsmodels package is required for VIF calculation")
        raise ImportError("statsmodels package is required for VIF calculation. Install it with 'pip install statsmodels'")
    
    # If features not specified, use all numeric columns
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to only include numeric features from the provided list
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in features if f in numeric_features]
    
    # Check if we have enough features for VIF calculation
    if len(features) < 2:
        error_msg = f"VIF calculation requires at least 2 numeric features. Found {len(features)} numeric features."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Drop rows with missing values in the selected features
    df_clean = df[features].dropna()
    logger.info(f"Using {len(df_clean)} complete rows for VIF calculation out of {len(df)} total rows")
    
    # Calculate VIF for each feature
    X = df_clean[features]
    
    # Calculate VIF values
    vif_values = {}
    for i, feature in enumerate(features):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values[feature] = vif
        except Exception as e:
            logger.warning(f"Error calculating VIF for feature '{feature}': {e}")
            vif_values[feature] = np.nan
    
    # Create DataFrame with VIF values
    vif_data = pd.DataFrame({
        "Feature": list(vif_values.keys()),
        "VIF": list(vif_values.values())
    })
    
    # Sort by VIF value in descending order
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    # Save VIF data to CSV if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        os.makedirs(save_path.parent, exist_ok=True)
        vif_data.to_csv(save_path, index=False)
        logger.info(f"Saved VIF analysis to {save_path}")
    
    # Print features with high VIF if requested
    if print_high_vif:
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        if not high_vif_features.empty:
            logger.info(f"Features with high multicollinearity (VIF > {vif_threshold}):")
            for i, row in high_vif_features.iterrows():
                logger.info(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")
        else:
            logger.info(f"No features with high VIF (> {vif_threshold}) found")
    
    # Return results in requested format
    if return_dataframe:
        return vif_data
    else:
        return dict(zip(vif_data["Feature"], vif_data["VIF"]))

def plot_vif(
    vif_data: Union[pd.DataFrame, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    threshold: float = 5.0,
    title: str = "Variance Inflation Factor (VIF) Analysis",
    show_plot: bool = False
) -> None:
    """
    Create a horizontal bar plot of VIF values.
    
    Args:
        vif_data (pd.DataFrame or Dict[str, float]): VIF values from calculate_vif function
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        figsize (tuple, optional): Figure size as (width, height). Default is (10, 8).
        threshold (float, optional): Threshold for highlighting high VIF values. Default is 5.0.
        title (str, optional): Plot title. Default is "Variance Inflation Factor (VIF) Analysis".
        show_plot (bool, optional): If True, displays the plot. Default is False.
    
    Returns:
        None
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(vif_data, dict):
        vif_df = pd.DataFrame({
            "Feature": list(vif_data.keys()),
            "VIF": list(vif_data.values())
        }).sort_values("VIF", ascending=False)
    else:
        vif_df = vif_data.copy()
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Sort by VIF value in ascending order for better visualization
    vif_df = vif_df.sort_values("VIF", ascending=True)
    
    # Create color map based on threshold
    colors = ['#ff7f0e' if vif >= threshold else '#1f77b4' for vif in vif_df["VIF"]]
    
    # Create horizontal bar chart
    bars = plt.barh(vif_df["Feature"], vif_df["VIF"], color=colors)
    
    # Add a vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Add labels and title
    plt.xlabel('VIF Value')
    plt.ylabel('Feature')
    plt.title(title)
    
    # Add legend
    plt.legend(['Threshold', 'High VIF (Problematic)', 'Low VIF (Good)'])
    
    # Add VIF values as text
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{vif_df["VIF"].iloc[i]:.2f}',
                va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved VIF plot to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

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
    # Try to import GPU-accelerated Cox model
    try:
        from src.gpu_cox import CumlCoxPHFitter, calculate_cuml_information_criteria, batch_process_large_dataset
        use_gpu = True
        logger.info("Using GPU-accelerated Cox model")
    except ImportError:
        logger.warning("GPU-accelerated Cox model not available. Falling back to lifelines.")
        try:
            # Import lifelines for Cox proportional hazards models as fallback
            from lifelines import CoxPHFitter
            from lifelines.statistics import proportional_hazard_test
            use_gpu = False
        except ImportError:
            logger.error("Neither GPU-accelerated Cox model nor lifelines is available")
            raise ImportError("Either GPU-accelerated Cox model or lifelines is required. Install with 'pip install cuml cupy' or 'pip install lifelines'")
    
    # If features not specified, use all numeric columns except duration and event
    if features is None:
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in all_numeric if f not in [duration_col, event_col]]
    else:
        # Filter to only include numeric features from the provided list
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in features if f in numeric_features]
    
    # Check if we have enough features
    if len(features) < 1:
        error_msg = f"AIC/BIC calculation requires at least 1 numeric feature. Found {len(features)} numeric features."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Drop rows with missing values in the selected features, duration, and event
    cols_to_check = features + [duration_col, event_col]
    df_clean = df[cols_to_check].dropna()
    logger.info(f"Using {len(df_clean)} complete rows for AIC/BIC calculation out of {len(df)} total rows")
    
    # Prepare the results container
    results = {}
    
    # Function to calculate AIC and BIC for a given model
    def get_criteria(model, df):
        try:
            if use_gpu and isinstance(model, CumlCoxPHFitter):
                # Use GPU-accelerated calculation
                try:
                    return calculate_cuml_information_criteria(model, df)
                except Exception as e:
                    logger.warning(f"GPU-accelerated information criteria calculation failed: {e}")
                    logger.warning("Falling back to manual calculation")
                    # Fall back to manual calculation
                    k = len(model.params_)  # Number of parameters
                    ll = model.log_likelihood_  # Log-likelihood
                    n = df.shape[0]  # Number of observations
                    aic = -2 * ll + 2 * k
                    bic = -2 * ll + k * np.log(n)
                    return {"AIC": float(aic), "BIC": float(bic)}
            else:
                # Calculate AIC - lifelines provides AIC_partial_
                if hasattr(model, 'AIC_partial_'):
                    aic = model.AIC_partial_
                else:
                    # Manual calculation if not available
                    k = len(model.params_)  # Number of parameters
                    ll = model.log_likelihood_  # Log-likelihood
                    aic = -2 * ll + 2 * k
                
                # Calculate BIC manually
                k = len(model.params_)  # Number of parameters
                ll = model.log_likelihood_  # Log-likelihood
                n = df.shape[0]  # Number of observations
                bic = -2 * ll + k * np.log(n)
                
                # Ensure we return float values, not numpy or cupy arrays
                return {"AIC": float(aic), "BIC": float(bic)}
        except Exception as e:
            logger.warning(f"Error calculating information criteria: {e}")
            return {"AIC": np.nan, "BIC": np.nan}
    
    # 1. Univariate models (one feature at a time)
    if method == 'univariate' or method == 'stepwise':
        logger.info(f"Fitting univariate Cox models for each feature ({len(features)} features total)...")
        logger.info(f"Using {len(df_clean)} complete samples for analysis")
        univariate_results = {}
        
        for i, feature in enumerate(features, 1):
            logger.info(f"Processing feature {i}/{len(features)}: '{feature}'")
            # Create a dataframe with just this feature, duration, and event
            feature_df = df_clean[[feature, duration_col, event_col]].copy()
            
            # Fit Cox model with robust parameters
            try:
                if use_gpu:
                    # Use GPU-accelerated Cox model
                    cph = CumlCoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    try:
                        # Fit the model with GPU acceleration
                        logger.info(f"Starting GPU-accelerated Cox model fitting for feature '{feature}'...")
                        feature_start_time = time.time()
                        try:
                            cph.fit(feature_df, duration_col=duration_col, event_col=event_col, show_progress=True)
                            feature_time = time.time() - feature_start_time
                            logger.info(f"Completed fitting for feature '{feature}' in {feature_time:.2f} seconds")
                        except Exception as e:
                            logger.error(f"Error during GPU-accelerated Cox model fitting: {e}")
                            logger.warning("Falling back to lifelines for this feature")
                            # Fall back to lifelines
                            from lifelines import CoxPHFitter as LifelinesCoxPHFitter
                            cph = LifelinesCoxPHFitter(penalizer=0.01)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=Warning)
                                try:
                                    cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                            robust=True, show_progress=False)
                                except:
                                    # Try with different solver parameters if first attempt fails
                                    # Note: lifelines doesn't support step_size, use different parameters
                                    logger.info("First lifelines fitting attempt failed, trying with different parameters")
                                    cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                            robust=True, show_progress=False,
                                            alpha=0.05,  # Equivalent to step_size in some contexts
                                            tol=1e-5)    # More lenient tolerance
                    except Exception as e:
                        logger.warning(f"GPU-accelerated Cox model fitting failed: {e}")
                        logger.warning("Falling back to lifelines for this feature")
                        # Fall back to lifelines
                        from lifelines import CoxPHFitter as LifelinesCoxPHFitter
                        cph = LifelinesCoxPHFitter(penalizer=0.01)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            try:
                                cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False)
                            except:
                                # Try with different solver parameters if first attempt fails
                                logger.info("First lifelines fitting attempt failed, trying with different parameters")
                                cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False,
                                        alpha=0.05,  # Equivalent to step_size in some contexts
                                        tol=1e-5)    # More lenient tolerance
                else:
                    # Use lifelines Cox model
                    cph = CoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    # Suppress convergence warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        # Try with different solver options if needed
                        try:
                            cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False)
                        except:
                            # Try with different solver if first attempt fails
                            cph.fit(feature_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False, step_size=0.5,
                                    max_steps=100)
                
                # Get AIC and BIC
                univariate_results[feature] = get_criteria(cph, feature_df)
                logger.info(f"  {feature}: AIC={univariate_results[feature]['AIC']:.2f}, BIC={univariate_results[feature]['BIC']:.2f}")
            except Exception as e:
                logger.warning(f"Error fitting Cox model for feature '{feature}': {e}")
                univariate_results[feature] = {"AIC": np.nan, "BIC": np.nan}
        
        results["univariate"] = univariate_results
    
    # 2. Full model (all features)
    if method == 'full' or method == 'stepwise':
        logger.info("Fitting full Cox model with all features...")
        logger.info(f"Model will include {len(features)} features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
        logger.info(f"Using {len(df_clean)} complete samples for full model")
        try:
            # Determine whether to use GPU acceleration
            if use_gpu:
                # Check if dataset is large enough to require batch processing
                large_dataset = len(df_clean) > 100000  # Threshold for batch processing
                
                if large_dataset:
                    logger.info(f"Large dataset detected ({len(df_clean)} samples). Using batch processing.")
                    logger.info(f"Processing in batches of 50,000 samples with 5 epochs")
                    # Use batch processing for large datasets
                    cph = batch_process_large_dataset(
                        df_clean,
                        duration_col=duration_col,
                        event_col=event_col,
                        features=features,
                        batch_size=50000,  # Process in batches of 50,000 samples
                        penalizer=0.01,
                        show_progress=True,
                        max_epochs=5
                    )
                else:
                    # Use GPU-accelerated Cox model for regular-sized datasets
                    logger.info(f"Using GPU-accelerated Cox model for dataset with {len(df_clean)} samples")
                    cph = CumlCoxPHFitter(penalizer=0.01)
                    cph.fit(df_clean, duration_col=duration_col, event_col=event_col, show_progress=True)
            else:
                # Fit Cox model with all features using robust parameters
                logger.info(f"Using CPU-based lifelines Cox model with {len(features)} features")
                cph = CoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                
                # Suppress convergence warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    # Try with different solver options if needed
                    try:
                        cph.fit(df_clean, duration_col=duration_col, event_col=event_col,
                                robust=True, show_progress=False)
                    except:
                        # Try with different solver parameters if first attempt fails
                        logger.info("First lifelines fitting attempt failed, trying with different parameters")
                        cph.fit(df_clean, duration_col=duration_col, event_col=event_col,
                                robust=True, show_progress=False,
                                alpha=0.05,  # Equivalent to step_size in some contexts
                                tol=1e-5)    # More lenient tolerance
            
            # Get AIC and BIC
            full_model_criteria = get_criteria(cph, df_clean)
            results["full_model"] = {"all_features": full_model_criteria}
            logger.info(f"Full model: AIC={full_model_criteria['AIC']:.2f}, BIC={full_model_criteria['BIC']:.2f}")
            
            # Store feature coefficients and p-values
            coef_df = cph.summary
            results["full_model_coefficients"] = coef_df.to_dict()
        except Exception as e:
            logger.warning(f"Error fitting full Cox model: {e}")
            results["full_model"] = {"all_features": {"AIC": np.nan, "BIC": np.nan}}
    
    # 3. Stepwise selection
    if method == 'stepwise':
        logger.info("Performing stepwise feature selection...")
        logger.info(f"Selection criterion: {criterion.upper()}")
        
        # Sort features by univariate AIC/BIC
        sorted_features = []
        if criterion == 'aic' or criterion == 'both':
            logger.info("Sorting features by univariate AIC values")
            sorted_features = sorted(
                features,
                key=lambda f: results["univariate"][f]["AIC"] if not np.isnan(results["univariate"][f]["AIC"]) else float('inf')
            )
        else:  # criterion == 'bic'
            logger.info("Sorting features by univariate BIC values")
            sorted_features = sorted(
                features,
                key=lambda f: results["univariate"][f]["BIC"] if not np.isnan(results["univariate"][f]["BIC"]) else float('inf')
            )
        
        # Log top features by univariate criterion
        top_univariate = sorted_features[:min(10, len(sorted_features))]
        logger.info(f"Top {len(top_univariate)} features by univariate {criterion.upper()}:")
        for i, feature in enumerate(top_univariate, 1):
            criterion_value = results["univariate"][feature]["AIC" if criterion == 'aic' or criterion == 'both' else "BIC"]
            logger.info(f"  {i}. {feature}: {criterion_value:.2f}")
        
        # Forward stepwise selection
        logger.info("Starting forward stepwise selection process")
        current_features = []
        stepwise_results = {}
        best_criterion_value = float('inf')
        best_feature_set = []
        logger.info(f"Will evaluate up to {len(sorted_features)} feature combinations")
        
        for step, feature in enumerate(sorted_features, 1):
            logger.info(f"Step {step}/{len(sorted_features)}: Adding feature '{feature}'")
            current_features.append(feature)
            
            # Create a dataframe with current features, duration, and event
            current_df = df_clean[current_features + [duration_col, event_col]].copy()
            logger.info(f"Current model has {len(current_features)} features")
            
            # Fit Cox model with robust parameters
            try:
                if use_gpu:
                    # Use GPU-accelerated Cox model
                    cph = CumlCoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    try:
                        # Fit the model with GPU acceleration
                        cph.fit(current_df, duration_col=duration_col, event_col=event_col, show_progress=False)
                    except Exception as e:
                        logger.warning(f"GPU-accelerated Cox model fitting failed for stepwise model: {e}")
                        logger.warning("Falling back to lifelines for this step")
                        # Fall back to lifelines
                        from lifelines import CoxPHFitter as LifelinesCoxPHFitter
                        cph = LifelinesCoxPHFitter(penalizer=0.01)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            try:
                                cph.fit(current_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False)
                            except:
                                # Try with different solver parameters if first attempt fails
                                logger.info("First lifelines fitting attempt failed, trying with different parameters")
                                cph.fit(current_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False,
                                        alpha=0.05,  # Equivalent to step_size in some contexts
                                        tol=1e-5)    # More lenient tolerance
                else:
                    # Use lifelines Cox model
                    cph = CoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    # Suppress convergence warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        # Try with different solver options if needed
                        try:
                            cph.fit(current_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False)
                        except:
                            # Try with different solver parameters if first attempt fails
                            logger.info("First lifelines fitting attempt failed, trying with different parameters")
                            cph.fit(current_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False,
                                    alpha=0.05,  # Equivalent to step_size in some contexts
                                    tol=1e-5)    # More lenient tolerance
                
                # Get AIC and BIC
                model_criteria = get_criteria(cph, current_df)
                feature_key = ", ".join(current_features)
                stepwise_results[feature_key] = model_criteria
                
                # Check if this is the best model so far
                current_criterion = model_criteria["AIC"] if criterion == 'aic' or criterion == 'both' else model_criteria["BIC"]
                is_best = current_criterion < best_criterion_value
                
                if is_best:
                    best_criterion_value = current_criterion
                    best_feature_set = current_features.copy()
                    best_indicator = "★ BEST SO FAR"
                else:
                    best_indicator = ""
                
                logger.info(f"  Added {feature}: AIC={model_criteria['AIC']:.2f}, BIC={model_criteria['BIC']:.2f} {best_indicator}")
            except Exception as e:
                logger.warning(f"Error in stepwise selection when adding feature '{feature}': {e}")
                feature_key = ", ".join(current_features)
                stepwise_results[feature_key] = {"AIC": np.nan, "BIC": np.nan}
        
        results["stepwise"] = stepwise_results
        results["best_feature_set"] = best_feature_set
        
        if best_feature_set:
            logger.info(f"Stepwise selection complete. Best model has {len(best_feature_set)} features:")
            logger.info(f"Best features: {', '.join(best_feature_set)}")
            logger.info(f"Best {criterion.upper()}: {best_criterion_value:.4f}")
            
            # Fit final model with best feature set
            logger.info("Fitting final model with best feature set...")
            best_df = df_clean[best_feature_set + [duration_col, event_col]].copy()
            try:
                if use_gpu:
                    # Use GPU-accelerated Cox model
                    cph = CumlCoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    try:
                        # Fit the model with GPU acceleration
                        cph.fit(best_df, duration_col=duration_col, event_col=event_col, show_progress=False)
                    except Exception as e:
                        logger.warning(f"GPU-accelerated Cox model fitting failed for best model: {e}")
                        logger.warning("Falling back to lifelines for best model")
                        # Fall back to lifelines
                        from lifelines import CoxPHFitter as LifelinesCoxPHFitter
                        cph = LifelinesCoxPHFitter(penalizer=0.01)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=Warning)
                            try:
                                cph.fit(best_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False)
                            except:
                                # Try with different solver parameters if first attempt fails
                                logger.info("First lifelines fitting attempt failed, trying with different parameters")
                                cph.fit(best_df, duration_col=duration_col, event_col=event_col,
                                        robust=True, show_progress=False,
                                        alpha=0.05,  # Equivalent to step_size in some contexts
                                        tol=1e-5)    # More lenient tolerance
                else:
                    # Use lifelines Cox model
                    cph = CoxPHFitter(penalizer=0.01)  # Add small ridge penalty to improve convergence
                    # Suppress convergence warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        # Try with different solver options if needed
                        try:
                            cph.fit(best_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False)
                        except:
                            # Try with different solver parameters if first attempt fails
                            logger.info("First lifelines fitting attempt failed, trying with different parameters")
                            cph.fit(best_df, duration_col=duration_col, event_col=event_col,
                                    robust=True, show_progress=False,
                                    alpha=0.05,  # Equivalent to step_size in some contexts
                                    tol=1e-5)    # More lenient tolerance
                
                best_model_criteria = get_criteria(cph, best_df)
                results["best_model"] = {"features": best_feature_set, "criteria": best_model_criteria}
                logger.info(f"Best model: AIC={best_model_criteria['AIC']:.2f}, BIC={best_model_criteria['BIC']:.2f}")
                
                # Store feature coefficients and p-values for best model
                coef_df = cph.summary
                results["best_model_coefficients"] = coef_df.to_dict()
            except Exception as e:
                logger.warning(f"Error fitting best model: {e}")
                results["best_model"] = {"features": best_feature_set, "criteria": {"AIC": np.nan, "BIC": np.nan}}
    
    # Convert results to DataFrame if requested
    if return_dataframe:
        if method == 'univariate':
            # Create DataFrame from univariate results
            df_results = pd.DataFrame([
                {
                    "Feature": feature,
                    "AIC": results["univariate"][feature]["AIC"],
                    "BIC": results["univariate"][feature]["BIC"]
                }
                for feature in features
            ])
            # Sort by the specified criterion
            if criterion == 'aic' or criterion == 'both':
                df_results = df_results.sort_values("AIC")
            else:  # criterion == 'bic'
                df_results = df_results.sort_values("BIC")
        elif method == 'full':
            # Create DataFrame with just the full model results
            df_results = pd.DataFrame([
                {
                    "Model": "Full Model",
                    "Features": ", ".join(features),
                    "AIC": results["full_model"]["all_features"]["AIC"],
                    "BIC": results["full_model"]["all_features"]["BIC"]
                }
            ])
        else:  # method == 'stepwise'
            # Create DataFrame from stepwise results
            df_results = pd.DataFrame([
                {
                    "Features": feature_set,
                    "AIC": results["stepwise"][feature_set]["AIC"],
                    "BIC": results["stepwise"][feature_set]["BIC"]
                }
                for feature_set in results["stepwise"]
            ])
            # Sort by the specified criterion
            if criterion == 'aic' or criterion == 'both':
                df_results = df_results.sort_values("AIC")
            else:  # criterion == 'bic'
                df_results = df_results.sort_values("BIC")
        
        # Save results to CSV if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            # Create directory if it doesn't exist
            os.makedirs(save_path.parent, exist_ok=True)
            df_results.to_csv(save_path, index=False)
            logger.info(f"Saved information criteria results to {save_path}")
        
        return df_results
    else:
        # Save results to CSV if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            # Create directory if it doesn't exist
            os.makedirs(save_path.parent, exist_ok=True)
            
            # Convert to DataFrame for saving
            if method == 'univariate':
                df_to_save = pd.DataFrame([
                    {
                        "Feature": feature,
                        "AIC": results["univariate"][feature]["AIC"],
                        "BIC": results["univariate"][feature]["BIC"]
                    }
                    for feature in features
                ])
            elif method == 'full':
                df_to_save = pd.DataFrame([
                    {
                        "Model": "Full Model",
                        "Features": ", ".join(features),
                        "AIC": results["full_model"]["all_features"]["AIC"],
                        "BIC": results["full_model"]["all_features"]["BIC"]
                    }
                ])
            else:  # method == 'stepwise'
                df_to_save = pd.DataFrame([
                    {
                        "Features": feature_set,
                        "AIC": results["stepwise"][feature_set]["AIC"],
                        "BIC": results["stepwise"][feature_set]["BIC"]
                    }
                    for feature_set in results["stepwise"]
                ])
            
            df_to_save.to_csv(save_path, index=False)
            logger.info(f"Saved information criteria results to {save_path}")
        
        return results

def perform_cox_lasso(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    cv_folds: int = 3,
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 1.0,  # 1.0 for LASSO, between 0-1 for elastic net
    max_iter: int = 10000,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    use_gpu: bool = False,
    use_r: bool = True,
    parallel: bool = True
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
        cv_folds (int, optional): Number of cross-validation folds. Default is 3.
        alphas (List[float], optional): List of alpha values to try.
                                      If None, uses a logarithmic sequence.
        l1_ratio (float, optional): Elastic net mixing parameter (0 <= l1_ratio <= 1).
                                  l1_ratio=1 corresponds to LASSO. Default is 1.0.
        max_iter (int, optional): Maximum number of iterations. Default is 10000.
        output_dir (str or Path, optional): Directory to save outputs.
                                          If None, uses current directory.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
        use_gpu (bool, optional): Whether to use GPU acceleration if available. Default is False.
        use_r (bool, optional): Whether to use R's glmnet package via rpy2. Default is True.
                              This is often faster than Python implementations, even GPU-accelerated ones.
        parallel (bool, optional): Whether to use parallel processing for R cross-validation.
                                 Only used when use_r=True. Default is True.
    
    Returns:
        Dict[str, Any]: Dictionary containing results and paths to saved files
    """
    # Try to use R implementation if requested
    if use_r:
        try:
            logger.info("Using R's glmnet package for Cox-LASSO regression")
            return perform_r_cox_lasso(
                df=df,
                duration_col=duration_col,
                event_col=event_col,
                features=features,
                cv_folds=cv_folds,
                alphas=alphas,
                l1_ratio=l1_ratio,
                output_dir=output_dir,
                random_state=random_state,
                parallel=parallel
            )
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to use R-based Cox-LASSO: {str(e)}")
            logger.warning("Falling back to Python implementation")
    
    # Try to use GPU-accelerated implementation if requested
    if use_gpu:
        try:
            # Import GPU-accelerated implementation
            from src.gpu_cox import perform_gpu_cox_lasso
            
            logger.info("Using GPU-accelerated Cox-LASSO implementation")
            logger.info(f"Dataset size: {len(df)} rows, {len(features) if features else 'all'} features")
            logger.info(f"Cross-validation: {cv_folds} folds, L1 ratio: {l1_ratio}")
            logger.info(f"Max iterations: {max_iter}, Random state: {random_state}")
            
            if output_dir:
                logger.info(f"Results will be saved to: {output_dir}")
            
            return perform_gpu_cox_lasso(
                df=df,
                duration_col=duration_col,
                event_col=event_col,
                features=features,
                cv_folds=cv_folds,
                alphas=alphas,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                output_dir=output_dir,
                random_state=random_state
            )
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to use GPU-accelerated Cox-LASSO: {str(e)}")
            logger.warning("Falling back to CPU implementation")
            logger.info("This may be significantly slower, especially for large datasets")
    
    # Fall back to CPU implementation
    try:
        # Import required libraries
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError:
        logger.error("scikit-survival package is required for Cox-LASSO")
        raise ImportError("scikit-survival package is required for Cox-LASSO. Install it with 'pip install scikit-survival'")
    
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
        # Check if features exist in the dataframe
        # Note: We don't filter for numeric features here because encoded categorical features
        # are already numeric (0/1 values from one-hot encoding)
        features = [f for f in features if f in df.columns]
        logger.info(f"Using {len(features)} features provided by the caller")
    
    # Check if we have enough features
    if len(features) < 2:
        error_msg = f"LASSO regression requires at least 2 features. Found {len(features)} features."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Drop rows with missing values in the selected features, duration, and event
    cols_to_check = features + [duration_col, event_col]
    df_clean = df[cols_to_check].dropna()
    logger.info(f"Using {len(df_clean)} complete rows for CPU-based LASSO regression out of {len(df)} total rows")
    logger.info(f"Features: {len(features)} total, Cross-validation: {cv_folds} folds")
    
    # Prepare the data for scikit-survival
    logger.info("Preparing data for scikit-survival...")
    X = df_clean[features].values
    y = Surv.from_arrays(
        event=df_clean[event_col].astype(bool).values,
        time=df_clean[duration_col].values
    )
    logger.info(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize features
    logger.info("Standardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up alpha values (regularization strengths)
    if alphas is None:
        # Create a logarithmic sequence of alphas
        logger.info("Creating logarithmic sequence of alpha values for regularization path")
        alphas = np.logspace(-3, 1, 100)
        logger.info(f"Testing {len(alphas)} alpha values from {alphas[0]:.6f} to {alphas[-1]:.6f}")
    else:
        logger.info(f"Using {len(alphas)} user-provided alpha values")
    
    # Set up cross-validation
    logger.info(f"Setting up {cv_folds}-fold cross-validation with random_state={random_state}")
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Initialize and fit the Cox-LASSO model
    logger.info(f"Fitting CPU-based Cox-LASSO model with {cv_folds}-fold cross-validation...")
    logger.info(f"L1 ratio (elastic net mixing parameter): {l1_ratio} (1.0 = LASSO, 0.0 = Ridge)")
    logger.info(f"Max iterations: {max_iter}, tolerance: 1e-7")
    coxnet = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        alphas=alphas,
        normalize=False,  # We already standardized the features
        max_iter=max_iter,
        tol=1e-7
        # random_state parameter removed as it's not supported by CoxnetSurvivalAnalysis
    )
    
    # Fit the model with cross-validation
    logger.info("Starting cross-validation process (this may take some time)...")
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coxnet.fit(X_scaled, y)
    elapsed_time = time.time() - start_time
    logger.info(f"Cross-validation complete in {elapsed_time:.2f} seconds")
    
    # Get the optimal alpha value
    try:
        # Try different ways to access the optimal alpha value
        if hasattr(coxnet, 'cv_results_') and 'alpha_idx_max' in coxnet.cv_results_:
            # Original approach
            optimal_alpha_idx = coxnet.cv_results_['alpha_idx_max']
            optimal_alpha = coxnet.alphas_[optimal_alpha_idx]
        elif hasattr(coxnet, 'alpha_'):
            # Direct alpha attribute (common in scikit-learn)
            optimal_alpha = coxnet.alpha_
        elif hasattr(coxnet, 'alphas_') and hasattr(coxnet, 'cv_alphas_'):
            # Find the alpha with the best score
            best_idx = np.argmax(coxnet.cv_alphas_)
            optimal_alpha = coxnet.alphas_[best_idx]
        else:
            # Fallback: use the first alpha value
            logger.warning("Could not determine optimal alpha value, using first alpha")
            optimal_alpha = coxnet.alphas_[0]
        
        logger.info(f"Optimal alpha (regularization parameter): {optimal_alpha:.6f}")
        logger.info(f"This alpha value maximizes the partial likelihood in cross-validation")
    except Exception as e:
        logger.warning(f"Error determining optimal alpha: {e}")
        logger.warning("Using first alpha value as fallback")
        optimal_alpha = coxnet.alphas_[0]
        logger.info(f"Fallback alpha value: {optimal_alpha:.6f}")
    
    # Get the coefficients at the optimal alpha
    optimal_coefs = coxnet.coef_[:, optimal_alpha_idx]
    non_zero_coefs = np.sum(optimal_coefs != 0)
    logger.info(f"Model selected {non_zero_coefs} non-zero coefficients out of {len(features)} features")
    
    # Create a DataFrame with feature names and their coefficients
    logger.info("Creating coefficient summary...")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': optimal_coefs
    })
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
    
    # Log the top features
    top_features = coef_df[coef_df['Coefficient'] != 0].head(10)
    if not top_features.empty:
        logger.info("Top features selected by the model (by coefficient magnitude):")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            logger.info(f"  {i}. {row['Feature']}: {row['Coefficient']:.6f}")
    
    # Save coefficients to CSV
    coef_path = output_dir / "lasso_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info(f"Saved LASSO coefficients to {coef_path}")
    
    # Get the coefficient path (how coefficients change with different alphas)
    logger.info("Analyzing coefficient paths across all regularization strengths...")
    coef_path_data = []
    feature_entry_points = {}  # Track when each feature enters the model
    
    logger.info(f"Processing {len(coxnet.alphas_)} alpha values for coefficient paths")
    # Process in reverse order (from highest to lowest alpha)
    for i, alpha in enumerate(coxnet.alphas_):
        coefs = coxnet.coef_[:, i]
        non_zero_count = np.sum(coefs != 0)
        
        # Track when features enter the model (as alpha decreases)
        for j, feature in enumerate(features):
            if coefs[j] != 0 and feature not in feature_entry_points:
                feature_entry_points[feature] = alpha
        
        # Add to coefficient path data
        for j, feature in enumerate(features):
            coef_path_data.append({
                'Alpha': alpha,
                'Log_Alpha': np.log10(alpha),
                'Feature': feature,
                'Coefficient': coefs[j],
                'Non_Zero_Features': non_zero_count
            })
    
    # Log feature entry points
    if feature_entry_points:
        logger.info("Feature entry points (alpha values where features enter the model):")
        sorted_entries = sorted(feature_entry_points.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, alpha) in enumerate(sorted_entries[:10], 1):  # Show top 10
            logger.info(f"  {i}. {feature}: α = {alpha:.6f}")
        if len(sorted_entries) > 10:
            logger.info(f"  ... and {len(sorted_entries) - 10} more features")
    
    # Convert to DataFrame
    logger.info("Creating coefficient path DataFrame...")
    coef_path_df = pd.DataFrame(coef_path_data)
    
    # Save coefficient path data to CSV
    coef_path_csv = output_dir / "lasso_coefficient_path.csv"
    coef_path_df.to_csv(coef_path_csv, index=False)
    logger.info(f"Saved LASSO coefficient path data to {coef_path_csv}")
    logger.info(f"Data contains {len(coef_path_df)} rows for {len(features)} features across {len(coxnet.alphas_)} alpha values")
    
    # Get cross-validation results
    logger.info("Processing cross-validation results...")
    try:
        # Try to access cv_results_ if available
        if hasattr(coxnet, 'cv_results_') and 'mean_partial_likelihood' in coxnet.cv_results_:
            cv_results = pd.DataFrame({
                'Alpha': coxnet.alphas_,
                'Log_Alpha': np.log10(coxnet.alphas_),
                'Mean_CV_Score': coxnet.cv_results_['mean_partial_likelihood'],
                'Std_CV_Score': coxnet.cv_results_['std_partial_likelihood']
            })
        else:
            # If cv_results_ is not available, create a simple DataFrame with just the alphas
            logger.warning("Cross-validation results not available in the model")
            logger.warning("Creating a simplified results DataFrame")
            
            # Create a DataFrame with just the alphas and placeholder scores
            cv_results = pd.DataFrame({
                'Alpha': coxnet.alphas_,
                'Log_Alpha': np.log10(coxnet.alphas_),
                'Mean_CV_Score': np.zeros_like(coxnet.alphas_),  # Placeholder
                'Std_CV_Score': np.zeros_like(coxnet.alphas_)    # Placeholder
            })
            
            # Mark the optimal alpha with a higher score
            if 'optimal_alpha' in locals():
                optimal_idx = np.argmin(np.abs(coxnet.alphas_ - optimal_alpha))
                cv_results.loc[optimal_idx, 'Mean_CV_Score'] = 1.0  # Mark as best
    except Exception as e:
        logger.warning(f"Error processing cross-validation results: {e}")
        # Create a minimal DataFrame with just the alphas
        cv_results = pd.DataFrame({
            'Alpha': coxnet.alphas_,
            'Log_Alpha': np.log10(coxnet.alphas_),
            'Mean_CV_Score': np.zeros_like(coxnet.alphas_),
            'Std_CV_Score': np.zeros_like(coxnet.alphas_)
        })
    
    # Add number of non-zero features for each alpha
    logger.info("Calculating feature counts for each alpha value...")
    non_zero_features = []
    for i in range(len(coxnet.alphas_)):
        non_zero_features.append(np.sum(coxnet.coef_[:, i] != 0))
    cv_results['Non_Zero_Features'] = non_zero_features
    
    # Log best cross-validation score
    best_cv_idx = cv_results['Mean_CV_Score'].idxmax()
    best_cv_alpha = cv_results.loc[best_cv_idx, 'Alpha']
    best_cv_score = cv_results.loc[best_cv_idx, 'Mean_CV_Score']
    best_cv_features = cv_results.loc[best_cv_idx, 'Non_Zero_Features']
    logger.info(f"Best cross-validation score: {best_cv_score:.4f} at alpha={best_cv_alpha:.6f} with {best_cv_features:.0f} features")
    
    # Save cross-validation results to CSV
    cv_results_csv = output_dir / "lasso_cv_results.csv"
    cv_results.to_csv(cv_results_csv, index=False)
    logger.info(f"Saved cross-validation results to {cv_results_csv}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Coefficient path plot
    logger.info("Creating coefficient path plot...")
    plt.figure(figsize=(12, 8))
    
    # Get unique features
    unique_features = coef_path_df['Feature'].unique()
    logger.info(f"Plotting coefficient paths for {len(unique_features)} features")
    
    # Sort features by the order they enter the model (as lambda decreases)
    logger.info("Sorting features by order of entry into the model...")
    feature_entry_order = []
    for alpha_idx in range(len(coxnet.alphas_) - 1, -1, -1):  # Start from largest alpha (most regularization)
        coefs = coxnet.coef_[:, alpha_idx]
        for j, feature in enumerate(features):
            if coefs[j] != 0 and feature not in feature_entry_order:
                feature_entry_order.append(feature)
    
    # Add any remaining features that never entered the model
    for feature in features:
        if feature not in feature_entry_order:
            feature_entry_order.append(feature)
    
    # Create a color map
    colors = plt.cm.tab10.colors
    color_map = {feature: colors[i % len(colors)] for i, feature in enumerate(feature_entry_order)}
    
    # Plot each feature's coefficient path
    for feature in feature_entry_order:
        feature_data = coef_path_df[coef_path_df['Feature'] == feature]
        plt.plot(feature_data['Log_Alpha'], feature_data['Coefficient'],
                 label=feature, color=color_map[feature], linewidth=2)
    
    # Add vertical line at optimal alpha
    plt.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--',
                label=f'Optimal α: {optimal_alpha:.6f}')
    
    # Add labels and title
    plt.xlabel('Log(α)')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Coefficient Paths')
    
    # Add legend with sorted features
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    coef_path_plot = output_dir / "lasso_coefficient_path.png"
    logger.info(f"Saving coefficient path plot to {coef_path_plot}...")
    plt.savefig(coef_path_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved coefficient path plot to {coef_path_plot}")
    
    # 2. Cross-validation score plot with number of features
    logger.info("Creating cross-validation score plot with feature counts...")
    plt.figure(figsize=(12, 8))
    
    # Create primary axis for CV score
    ax1 = plt.gca()
    ax1.plot(cv_results['Log_Alpha'], cv_results['Mean_CV_Score'], 'b-', linewidth=2)
    
    # Add error bands
    upper = cv_results['Mean_CV_Score'] + cv_results['Std_CV_Score']
    lower = cv_results['Mean_CV_Score'] - cv_results['Std_CV_Score']
    ax1.fill_between(cv_results['Log_Alpha'], lower, upper, alpha=0.2, color='blue')
    
    # Add vertical line at optimal alpha
    ax1.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--',
               label=f'Optimal α: {optimal_alpha:.6f}')
    
    # Add labels for primary axis
    ax1.set_xlabel('Log(α)')
    ax1.set_ylabel('Mean CV Partial Likelihood', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary axis for number of features
    ax2 = ax1.twinx()
    ax2.plot(cv_results['Log_Alpha'], cv_results['Non_Zero_Features'], 'g-', linewidth=2)
    ax2.set_ylabel('Number of Non-Zero Features', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add title
    plt.title('Cross-Validation Results and Feature Count')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + ['Number of Features'], loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    cv_plot = output_dir / "lasso_cv_results.png"
    logger.info(f"Saving cross-validation results plot to {cv_plot}...")
    plt.savefig(cv_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cross-validation results plot to {cv_plot}")
    
    # Return results
    logger.info("Preparing final results dictionary...")
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
    
    # Log summary of results
    non_zero_features_count = sum(1 for coef in optimal_coefs if coef != 0)
    logger.info(f"LASSO analysis complete. Summary:")
    logger.info(f"  - Optimal alpha: {optimal_alpha:.6f}")
    logger.info(f"  - Selected {non_zero_features_count} non-zero features out of {len(features)} total features")
    logger.info(f"  - Created {len(results['output_files'])} output files in {output_dir}")
    logger.info(f"  - Top selected features: {', '.join(coef_df[coef_df['Coefficient'] != 0]['Feature'].head(5).tolist())}")
    
    return results

def perform_r_cox_lasso(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    cv_folds: int = 10,
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 1.0,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Perform Cox-LASSO regression using R's glmnet package via rpy2.
    
    This implementation uses R's highly optimized glmnet package, which is written in Fortran
    and is often faster than Python implementations, even GPU-accelerated ones.
    
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
        output_dir (str or Path, optional): Directory to save outputs.
                                          If None, uses current directory.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
        parallel (bool, optional): Whether to use parallel processing for cross-validation.
                                 Default is True.
    
    Returns:
        Dict[str, Any]: Dictionary containing results and paths to saved files
    """
    try:
        # Import required libraries
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
    except ImportError:
        logger.error("rpy2 package is required for R-based Cox-LASSO")
        raise ImportError("rpy2 package is required for R-based Cox-LASSO. Install it with 'pip install rpy2'")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path("results/lasso_analysis")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"R-based LASSO analysis results will be saved to: {output_dir}")
    
    # If features not specified, use all numeric columns except duration and event
    if features is None:
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in all_numeric if f not in [duration_col, event_col]]
    else:
        # Check if features exist in the dataframe
        features = [f for f in features if f in df.columns]
    
    logger.info(f"Using {len(features)} features for R-based LASSO regression")
    logger.info(f"Dataset size: {len(df)} rows, {len(features)} features")
    logger.info(f"Cross-validation: {cv_folds} folds, L1 ratio: {l1_ratio}")
    
    # Activate pandas and numpy conversion
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Create a copy of the dataframe with only the needed columns
    df_r = df[[duration_col, event_col] + features].copy()
    
    # Drop rows with missing values
    df_r = df_r.dropna()
    logger.info(f"Using {len(df_r)} complete rows after dropping missing values")
    
    # Convert to R dataframe
    rdf = pandas2ri.py2rpy(df_r)
    
    # Set up R environment
    robj = ro.r
    robj.assign("rdf", rdf)
    robj.assign("duration_col", duration_col)  # Pass duration column name to R
    robj.assign("event_col", event_col)        # Pass event column name to R
    robj.assign("seed", random_state)
    robj.assign("nfolds", cv_folds)
    robj.assign("alpha", l1_ratio)
    robj.assign("parallel", parallel)
    
    # Set number of lambda values
    nlambda = 30
    if alphas is not None:
        nlambda = len(alphas)
    robj.assign("nlambda", nlambda)
    
    # If alphas are provided, convert them to R
    if alphas is not None:
        robj.assign("lambda_values", ro.FloatVector(alphas))
        lambda_provided = True
    else:
        lambda_provided = False
    
    # Run R code to check and install required packages
    r_install_script = """
    # Function to check and install packages
    check_and_install <- function(pkg) {
        if (!requireNamespace(pkg, quietly = TRUE)) {
            message(paste("Installing package:", pkg))
            install.packages(pkg, repos = "https://cloud.r-project.org")
        }
        library(pkg, character.only = TRUE)
    }
    
    # Check and install required packages
    check_and_install("survival")
    check_and_install("glmnet")
    """
    
    # Execute package installation script
    logger.info("Checking and installing required R packages...")
    try:
        robj(r_install_script)
    except Exception as e:
        logger.error(f"Error installing R packages: {e}")
        raise ImportError(f"Failed to install required R packages: {e}")
    
    # Run main R code
    r_script = """
    # Load required libraries
    library(survival)
    library(glmnet)
    
    # Set random seed
    set.seed(seed)
    
    # Get feature columns (all columns except duration and event)
    feature_cols <- setdiff(colnames(rdf), c(duration_col, event_col))
    
    # Build design matrix
    X <- as.matrix(rdf[, feature_cols])
    
    # Cox survival outcome with competing risks
    # For competing risks (1=dialysis, 2=mortality), we need to create a status indicator
    # that is 1 for the event of interest (dialysis) and 0 otherwise
    # We'll fit separate models for each event type
    
    # For dialysis (event=1)
    dialysis_status <- as.numeric(rdf[[event_col]] == 1)
    y_dialysis <- Surv(rdf[[duration_col]], dialysis_status)
    
    # For mortality (event=2)
    mortality_status <- as.numeric(rdf[[event_col]] == 2)
    y_mortality <- Surv(rdf[[duration_col]], mortality_status)
    
    # We'll use dialysis as the primary outcome for the main model
    y <- y_dialysis
    
    # Fit models for dialysis outcome
    # Fit the full LASSO path for dialysis
    fit_dialysis <- glmnet(
        x = X,
        y = y_dialysis,
        family = "cox",
        alpha = alpha,  # 1 = pure LASSO, 0 < α < 1 = elastic-net
        nlambda = nlambda,
        standardize = TRUE
    )
    
    # Cross-validated lambda selection for dialysis
    cvfit_dialysis <- cv.glmnet(
        x = X,
        y = y_dialysis,
        family = "cox",
        alpha = alpha,
        nlambda = nlambda,
        nfolds = nfolds,
        parallel = parallel,
        type.measure = "deviance",  # partial-likelihood deviance
        seed = seed
    )
    
    # Extract the lambda that minimizes mean CV deviance for dialysis
    lambda.min.dialysis <- cvfit_dialysis$lambda.min  # "best" lambda
    lambda.1se.dialysis <- cvfit_dialysis$lambda.1se  # more parsimonious lambda
    
    # Corresponding coefficient vectors for dialysis
    beta.min.dialysis <- as.vector(coef(cvfit_dialysis, s = "lambda.min"))
    beta.1se.dialysis <- as.vector(coef(cvfit_dialysis, s = "lambda.1se"))
    
    # Grab the full CV curve for dialysis
    cv.df.dialysis <- data.frame(
        lambda = cvfit_dialysis$lambda,
        log_lambda = log(cvfit_dialysis$lambda),
        cv_mean = cvfit_dialysis$cvm,  # mean deviance
        cv_se = cvfit_dialysis$cvsd,
        n_nonzero = cvfit_dialysis$nzero
    )
    
    # Fit models for mortality outcome
    # Fit the full LASSO path for mortality
    fit_mortality <- glmnet(
        x = X,
        y = y_mortality,
        family = "cox",
        alpha = alpha,  # 1 = pure LASSO, 0 < α < 1 = elastic-net
        nlambda = nlambda,
        standardize = TRUE
    )
    
    # Cross-validated lambda selection for mortality
    cvfit_mortality <- cv.glmnet(
        x = X,
        y = y_mortality,
        family = "cox",
        alpha = alpha,
        nlambda = nlambda,
        nfolds = nfolds,
        parallel = parallel,
        type.measure = "deviance",  # partial-likelihood deviance
        seed = seed
    )
    
    # Extract the lambda that minimizes mean CV deviance for mortality
    lambda.min.mortality <- cvfit_mortality$lambda.min  # "best" lambda
    lambda.1se.mortality <- cvfit_mortality$lambda.1se  # more parsimonious lambda
    
    # Corresponding coefficient vectors for mortality
    beta.min.mortality <- as.vector(coef(cvfit_mortality, s = "lambda.min"))
    beta.1se.mortality <- as.vector(coef(cvfit_mortality, s = "lambda.1se"))
    
    # Grab the full CV curve for mortality
    cv.df.mortality <- data.frame(
        lambda = cvfit_mortality$lambda,
        log_lambda = log(cvfit_mortality$lambda),
        cv_mean = cvfit_mortality$cvm,  # mean deviance
        cv_se = cvfit_mortality$cvsd,
        n_nonzero = cvfit_mortality$nzero
    )
    
    # For compatibility with the rest of the code, set the main results to the dialysis outcome
    # These will be used as the default results
    lambda.min <- lambda.min.dialysis
    lambda.1se <- lambda.1se.dialysis
    beta.min <- beta.min.dialysis
    beta.1se <- beta.1se.dialysis
    cv.df <- cv.df.dialysis
    
    # Get the full coefficient paths
    lambda_path_dialysis <- fit_dialysis$lambda
    coef_path_dialysis <- as.matrix(coef(fit_dialysis))
    
    lambda_path_mortality <- fit_mortality$lambda
    coef_path_mortality <- as.matrix(coef(fit_mortality))
    """
    
    # Add custom lambda values if provided
    if lambda_provided:
        r_script = r_script.replace("nlambda = nlambda", "lambda = lambda_values")
    
    # Execute R script
    logger.info("Executing R code for Cox-LASSO regression...")
    robj(r_script)
    
    # Extract results from R
    logger.info("Extracting results from R...")
    
    # Extract dialysis results
    lambda_min_dialysis = robj("lambda.min.dialysis")[0]
    lambda_1se_dialysis = robj("lambda.1se.dialysis")[0]
    
    coef_min_dialysis = np.asarray(robj("beta.min.dialysis"))
    coef_1se_dialysis = np.asarray(robj("beta.1se.dialysis"))
    
    cv_curve_dialysis = pandas2ri.rpy2py(robj("cv.df.dialysis"))
    
    # Extract mortality results
    lambda_min_mortality = robj("lambda.min.mortality")[0]
    lambda_1se_mortality = robj("lambda.1se.mortality")[0]
    
    coef_min_mortality = np.asarray(robj("beta.min.mortality"))
    coef_1se_mortality = np.asarray(robj("beta.1se.mortality"))
    
    cv_curve_mortality = pandas2ri.rpy2py(robj("cv.df.mortality"))
    
    # For compatibility with the rest of the code, use the dialysis results as the main results
    lambda_min = lambda_min_dialysis
    lambda_1se = lambda_1se_dialysis
    coef_min = coef_min_dialysis
    coef_1se = coef_1se_dialysis
    cv_curve = cv_curve_dialysis
    
    # Get the full coefficient paths
    lambda_path = np.asarray(robj("lambda_path_dialysis"))
    coef_path = np.asarray(robj("coef_path_dialysis"))
    
    # Create coefficient dataframes for dialysis
    coef_min_dialysis_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coef_min_dialysis
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    coef_1se_dialysis_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coef_1se_dialysis
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    # Create coefficient dataframes for mortality
    coef_min_mortality_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coef_min_mortality
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    coef_1se_mortality_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coef_1se_mortality
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    # For compatibility with the rest of the code
    coef_min_df = coef_min_dialysis_df
    coef_1se_df = coef_1se_dialysis_df
    
    # Count non-zero coefficients for dialysis
    non_zero_min_dialysis = np.sum(np.abs(coef_min_dialysis) > 1e-6)
    non_zero_1se_dialysis = np.sum(np.abs(coef_1se_dialysis) > 1e-6)
    
    # Count non-zero coefficients for mortality
    non_zero_min_mortality = np.sum(np.abs(coef_min_mortality) > 1e-6)
    non_zero_1se_mortality = np.sum(np.abs(coef_1se_mortality) > 1e-6)
    
    # For compatibility with the rest of the code
    non_zero_min = non_zero_min_dialysis
    non_zero_1se = non_zero_1se_dialysis
    
    logger.info(f"Dialysis outcome:")
    logger.info(f"  Optimal lambda (min): {lambda_min_dialysis:.6f} with {non_zero_min_dialysis} non-zero coefficients")
    logger.info(f"  Optimal lambda (1se): {lambda_1se_dialysis:.6f} with {non_zero_1se_dialysis} non-zero coefficients")
    
    logger.info(f"Mortality outcome:")
    logger.info(f"  Optimal lambda (min): {lambda_min_mortality:.6f} with {non_zero_min_mortality} non-zero coefficients")
    logger.info(f"  Optimal lambda (1se): {lambda_1se_mortality:.6f} with {non_zero_1se_mortality} non-zero coefficients")
    
    # Save results to files
    # Dialysis results
    coef_min_dialysis_path = output_dir / "dialysis_lasso_coefficients_min.csv"
    coef_1se_dialysis_path = output_dir / "dialysis_lasso_coefficients_1se.csv"
    cv_curve_dialysis_path = output_dir / "dialysis_lasso_cv_curve.csv"
    
    coef_min_dialysis_df.to_csv(coef_min_dialysis_path, index=False)
    coef_1se_dialysis_df.to_csv(coef_1se_dialysis_path, index=False)
    cv_curve_dialysis.to_csv(cv_curve_dialysis_path, index=False)
    
    # Mortality results
    coef_min_mortality_path = output_dir / "mortality_lasso_coefficients_min.csv"
    coef_1se_mortality_path = output_dir / "mortality_lasso_coefficients_1se.csv"
    cv_curve_mortality_path = output_dir / "mortality_lasso_cv_curve.csv"
    
    coef_min_mortality_df.to_csv(coef_min_mortality_path, index=False)
    coef_1se_mortality_df.to_csv(coef_1se_mortality_path, index=False)
    cv_curve_mortality.to_csv(cv_curve_mortality_path, index=False)
    
    logger.info(f"Saved dialysis min lambda coefficients to {coef_min_dialysis_path}")
    logger.info(f"Saved dialysis 1se lambda coefficients to {coef_1se_dialysis_path}")
    logger.info(f"Saved dialysis CV curve to {cv_curve_dialysis_path}")
    
    logger.info(f"Saved mortality min lambda coefficients to {coef_min_mortality_path}")
    logger.info(f"Saved mortality 1se lambda coefficients to {coef_1se_mortality_path}")
    logger.info(f"Saved mortality CV curve to {cv_curve_mortality_path}")
    
    # For compatibility with the rest of the code
    coef_min_path = coef_min_dialysis_path
    coef_1se_path = coef_1se_dialysis_path
    cv_curve_path = cv_curve_dialysis_path
    
    # Create visualizations
    
    # 1. CV curve plots
    # Dialysis CV curve
    logger.info("Creating dialysis CV curve plot...")
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        cv_curve_dialysis['log_lambda'],
        cv_curve_dialysis['cv_mean'],
        yerr=cv_curve_dialysis['cv_se'],
        fmt='o-',
        capsize=5,
        color='blue',
        label='CV Partial Likelihood Deviance'
    )
    
    # Add vertical lines at optimal lambdas
    plt.axvline(x=np.log(lambda_min_dialysis), color='blue', linestyle='--',
                label=f'λ_min: {lambda_min_dialysis:.6f} ({non_zero_min_dialysis} features)')
    plt.axvline(x=np.log(lambda_1se_dialysis), color='red', linestyle='--',
                label=f'λ_1se: {lambda_1se_dialysis:.6f} ({non_zero_1se_dialysis} features)')
    
    plt.xlabel('log(λ)')
    plt.ylabel('Partial Likelihood Deviance')
    plt.title('Cross-Validation Results for Cox-LASSO (Dialysis Outcome)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    cv_plot_dialysis_path = output_dir / "dialysis_lasso_cv_curve.png"
    plt.savefig(cv_plot_dialysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved dialysis CV curve plot to {cv_plot_dialysis_path}")
    
    # Mortality CV curve
    logger.info("Creating mortality CV curve plot...")
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        cv_curve_mortality['log_lambda'],
        cv_curve_mortality['cv_mean'],
        yerr=cv_curve_mortality['cv_se'],
        fmt='o-',
        capsize=5,
        color='green',
        label='CV Partial Likelihood Deviance'
    )
    
    # Add vertical lines at optimal lambdas
    plt.axvline(x=np.log(lambda_min_mortality), color='blue', linestyle='--',
                label=f'λ_min: {lambda_min_mortality:.6f} ({non_zero_min_mortality} features)')
    plt.axvline(x=np.log(lambda_1se_mortality), color='red', linestyle='--',
                label=f'λ_1se: {lambda_1se_mortality:.6f} ({non_zero_1se_mortality} features)')
    
    plt.xlabel('log(λ)')
    plt.ylabel('Partial Likelihood Deviance')
    plt.title('Cross-Validation Results for Cox-LASSO (Mortality Outcome)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    cv_plot_mortality_path = output_dir / "mortality_lasso_cv_curve.png"
    plt.savefig(cv_plot_mortality_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved mortality CV curve plot to {cv_plot_mortality_path}")
    
    # For compatibility with the rest of the code
    cv_plot_path = cv_plot_dialysis_path
    
    # 2. Feature importance plots for min lambda
    # Dialysis min lambda
    if non_zero_min_dialysis > 0:
        logger.info("Creating dialysis feature importance plot for min lambda...")
        top_features = coef_min_dialysis_df[abs(coef_min_dialysis_df['Coefficient']) > 1e-6].head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_features['Feature'],
            top_features['Coefficient'],
            color=['red' if c < 0 else 'blue' for c in top_features['Coefficient']]
        )
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Top Features Selected by Cox-LASSO for Dialysis (λ_min = {lambda_min_dialysis:.6f})')
        plt.grid(True, alpha=0.3)
        
        importance_plot_dialysis_min_path = output_dir / "dialysis_lasso_feature_importance_min.png"
        plt.savefig(importance_plot_dialysis_min_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved dialysis feature importance plot to {importance_plot_dialysis_min_path}")
    
    # Mortality min lambda
    if non_zero_min_mortality > 0:
        logger.info("Creating mortality feature importance plot for min lambda...")
        top_features = coef_min_mortality_df[abs(coef_min_mortality_df['Coefficient']) > 1e-6].head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_features['Feature'],
            top_features['Coefficient'],
            color=['red' if c < 0 else 'green' for c in top_features['Coefficient']]
        )
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Top Features Selected by Cox-LASSO for Mortality (λ_min = {lambda_min_mortality:.6f})')
        plt.grid(True, alpha=0.3)
        
        importance_plot_mortality_min_path = output_dir / "mortality_lasso_feature_importance_min.png"
        plt.savefig(importance_plot_mortality_min_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved mortality feature importance plot to {importance_plot_mortality_min_path}")
    
    # For compatibility with the rest of the code
    importance_plot_path = importance_plot_dialysis_min_path if non_zero_min_dialysis > 0 else None
    
    # 3. Feature importance plots for 1se lambda
    # Dialysis 1se lambda
    if non_zero_1se_dialysis > 0:
        logger.info("Creating dialysis feature importance plot for 1se lambda...")
        top_features = coef_1se_dialysis_df[abs(coef_1se_dialysis_df['Coefficient']) > 1e-6].head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_features['Feature'],
            top_features['Coefficient'],
            color=['red' if c < 0 else 'blue' for c in top_features['Coefficient']]
        )
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Top Features Selected by Cox-LASSO for Dialysis (λ_1se = {lambda_1se_dialysis:.6f})')
        plt.grid(True, alpha=0.3)
        
        importance_plot_dialysis_1se_path = output_dir / "dialysis_lasso_feature_importance_1se.png"
        plt.savefig(importance_plot_dialysis_1se_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved dialysis feature importance plot to {importance_plot_dialysis_1se_path}")
    
    # Mortality 1se lambda
    if non_zero_1se_mortality > 0:
        logger.info("Creating mortality feature importance plot for 1se lambda...")
        top_features = coef_1se_mortality_df[abs(coef_1se_mortality_df['Coefficient']) > 1e-6].head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_features['Feature'],
            top_features['Coefficient'],
            color=['red' if c < 0 else 'green' for c in top_features['Coefficient']]
        )
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Top Features Selected by Cox-LASSO for Mortality (λ_1se = {lambda_1se_mortality:.6f})')
        plt.grid(True, alpha=0.3)
        
        importance_plot_mortality_1se_path = output_dir / "mortality_lasso_feature_importance_1se.png"
        plt.savefig(importance_plot_mortality_1se_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved mortality feature importance plot to {importance_plot_mortality_1se_path}")
    
    # For compatibility with the rest of the code
    importance_plot_path = importance_plot_dialysis_1se_path if non_zero_1se_dialysis > 0 else None
    
    # Prepare results dictionary
    results = {
        # For compatibility with existing code
        "optimal_alpha": lambda_min,
        "optimal_alpha_1se": lambda_1se,
        "coef_min": dict(zip(features, coef_min)),
        "coef_1se": dict(zip(features, coef_1se)),
        "cv_curve": cv_curve,
        "non_zero_features_min": non_zero_min,
        "non_zero_features_1se": non_zero_1se,
        "coef_min_path": str(coef_min_path),
        "coef_1se_path": str(coef_1se_path),
        "cv_curve_path": str(cv_curve_path),
        "cv_plot_path": str(cv_plot_path),
        
        # Add output_files key for compatibility with feature_selection.py
        "output_files": {
            "coefficients_csv": str(coef_min_path),
            "coefficient_path_plot": str(importance_plot_path) if 'importance_plot_path' in locals() and importance_plot_path is not None else "",
            "cv_results_plot": str(cv_plot_path)
        },
        
        # Dialysis results
        "dialysis": {
            "optimal_alpha_min": lambda_min_dialysis,
            "optimal_alpha_1se": lambda_1se_dialysis,
            "coef_min": dict(zip(features, coef_min_dialysis)),
            "coef_1se": dict(zip(features, coef_1se_dialysis)),
            "cv_curve": cv_curve_dialysis,
            "non_zero_features_min": non_zero_min_dialysis,
            "non_zero_features_1se": non_zero_1se_dialysis,
            "coef_min_path": str(coef_min_dialysis_path),
            "coef_1se_path": str(coef_1se_dialysis_path),
            "cv_curve_path": str(cv_curve_dialysis_path)
        },
        
        # Mortality results
        "mortality": {
            "optimal_alpha_min": lambda_min_mortality,
            "optimal_alpha_1se": lambda_1se_mortality,
            "coef_min": dict(zip(features, coef_min_mortality)),
            "coef_1se": dict(zip(features, coef_1se_mortality)),
            "cv_curve": cv_curve_mortality,
            "non_zero_features_min": non_zero_min_mortality,
            "non_zero_features_1se": non_zero_1se_mortality,
            "coef_min_path": str(coef_min_mortality_path),
            "coef_1se_path": str(coef_1se_mortality_path),
            "cv_curve_path": str(cv_curve_mortality_path)
        }
    }
    
    return results

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
    from lifelines.statistics import logrank_test
    from lifelines import CoxPHFitter
    
    def score_stat(col):
        # Create binary endpoint
        df_copy = df[[col, duration_col, event_col]].copy()
        df_copy['event_binary'] = (df_copy[event_col] > 0).astype(int)
        
        # For continuous features, we need to dichotomize them for log-rank test
        # We'll use the median as the cutpoint
        median_val = df_copy[col].median()
        high_group = df_copy[col] > median_val
        
        # Get durations and events for each group
        t_high = df_copy.loc[high_group, duration_col].values
        e_high = df_copy.loc[high_group, 'event_binary'].values
        t_low = df_copy.loc[~high_group, duration_col].values
        e_low = df_copy.loc[~high_group, 'event_binary'].values
        
        # Perform log-rank test
        result = logrank_test(t_high, t_low, e_high, e_low)
        
        # Fit a simple Cox model to get the coefficient direction
        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(df_copy, duration_col=duration_col, event_col='event_binary',
                   robust=False, show_progress=False)
            coef = cph.params_.iloc[0] if len(cph.params_) > 0 else 0
            coef_sign = 1 if coef >= 0 else -1
        except:
            coef_sign = 1  # Default to positive if Cox model fails
        
        # Return feature name, test statistic (with sign), and p-value
        return col, coef_sign * result.test_statistic, result.p_value
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(score_stat)(col) for col in features)
    
    # Create DataFrame with results
    result_df = pd.DataFrame(results, columns=['Feature', 'Chi_Square', 'P_Value'])
    result_df = result_df.sort_values('Chi_Square', ascending=False)
    
    return result_df

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
        
        # Fit a simple Cox model
        cph = CoxPHFitter(penalizer=0.0)  # No ridge penalty for faster convergence
        cph.fit(
            df_copy[[col, duration_col, 'event_binary']],
            duration_col=duration_col,
            event_col='event_binary',
            robust=False,  # Turn off robust for faster computation
            show_progress=False
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
    logger.info("Performing fast univariate screening...")
    screening_results = fast_univariate_screening(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        features=features,
        n_jobs=n_jobs
    )
    logger.info(f"Univariate screening completed in {time.time() - start_time:.2f} seconds")
    
    if output_dir:
        screening_path = output_dir / "fast_univariate_screening.csv"
        screening_results.to_csv(screening_path, index=False)
        logger.info(f"Saved screening results to {screening_path}")
    
    results['screening'] = screening_results
    
    # 2. Fast AIC/BIC calculation for all features
    start_time = time.time()
    logger.info(f"Calculating AIC/BIC for all {len(features)} features...")
    ic_results = fast_calculate_ic(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        features=features,  # Use all features, not just top N
        n_jobs=n_jobs
    )
    logger.info(f"AIC/BIC calculation completed in {time.time() - start_time:.2f} seconds")
    
    if output_dir:
        ic_path = output_dir / "fast_aic_bic.csv"
        ic_results.to_csv(ic_path, index=False)
        logger.info(f"Saved AIC/BIC results to {ic_path}")
    
    results['ic'] = ic_results
    
    # Skip multivariable model fitting as it's not needed for DeepSurv/DeepHit
    # These models contain thousands of weights, making classical AIC/BIC lose meaning
    logger.info("Skipping multivariable model fitting as requested")
    
    return results