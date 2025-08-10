"""
Feature selection step for CKD Risk Prediction

This module contains the ZenML step for performing assessing collinearity, feature importance and feature selection.
Generating statistics only on training datasets.
"""

import os
import glob
import pandas as pd
import numpy as np
import re
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from pathlib import Path
import xgboost as xgb
import shap
from sksurv.util import Surv
from src.feature_eval import calculate_vif, plot_vif, calculate_information_criteria, perform_cox_lasso

# Function to update results dictionary with AIC/BIC and LASSO results
def update_results_with_new_outputs(results, aic_bic_path=None, aic_bic_stepwise_path=None,
                                   aic_bic_categorical_path=None, lasso_output_dir=None,
                                   lasso_results=None):
    """
    Update the results dictionary with AIC/BIC and LASSO results.
    
    Args:
        results (dict): The original results dictionary
        aic_bic_path (Path, optional): Path to the AIC/BIC univariate results
        aic_bic_stepwise_path (Path, optional): Path to the AIC/BIC stepwise results
        aic_bic_categorical_path (Path, optional): Path to the AIC/BIC univariate results for categorical features
        lasso_output_dir (Path, optional): Directory containing LASSO results
        lasso_results (dict, optional): Results from the LASSO regression
        
    Returns:
        dict: Updated results dictionary
    """
    # Add AIC/BIC results
    if aic_bic_path is not None:
        results["aic_bic_univariate_path"] = str(aic_bic_path)
    
    if aic_bic_stepwise_path is not None:
        results["aic_bic_stepwise_path"] = str(aic_bic_stepwise_path)
    
    if aic_bic_categorical_path is not None:
        results["aic_bic_univariate_categorical_path"] = str(aic_bic_categorical_path)
    
    # Add LASSO results
    if lasso_output_dir is not None:
        results["lasso_output_dir"] = str(lasso_output_dir)
    
    if lasso_results is not None:
        results["lasso_coefficients_path"] = str(lasso_results['output_files']['coefficients_csv'])
        results["lasso_coefficient_path_plot"] = str(lasso_results['output_files']['coefficient_path_plot'])
        results["lasso_cv_results_plot"] = str(lasso_results['output_files']['cv_results_plot'])
    
    return results

# Load environment variables
load_dotenv()

# Load data types from YAML file
def load_yaml_file(file_path):
    """Load a YAML file and return its contents."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}

# Default master dataframe mapping
master_df_mapping_path = Path("src/default_master_df_mapping.yml")
master_df_mapping_config = load_yaml_file(master_df_mapping_path)
DEFAULT_FEATURES = master_df_mapping_config.get('features')
DEFAULT_CLUSTER = master_df_mapping_config.get('cluster')
DEFAULT_DURATION = master_df_mapping_config.get('duration')
DEFAULT_EVENT = master_df_mapping_config.get('event')  # Fixed typo in variable name
DEFAULT_CAT_FEATURES = master_df_mapping_config.get('cat_features', [])
DEFAULT_CONT_FEATURES = master_df_mapping_config.get('cont_features', [])

@step()
def feature_selection(df: pd.DataFrame, output_path: Optional[str] = None, features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Process features from training dataframe and produce feature related statistics
    
    Args:
        df: training DataFrame
        output_path: Path to save the feature assessment results
        features: List of features to analyze. If None, uses the default features from the YAML config.
        
    Return:
        Dictionary containing feature assessment results and paths to save statistics
    """
    try:
        print("\n=== Performing Feature Assessment ===\n")
        
        # Get output path from environment variable if not provided
        if output_path is None:
            output_path = os.getenv("EDA_OUTPUT_PATH", "results")
        
        # Create absolute path if relative
        output_path = Path(output_path)
        if not output_path.is_absolute():
            # Assuming the current working directory is the project root
            
            output_path = Path(os.getcwd()) / output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Feature assessment results will be saved to: {output_path}")
        
        # 1. Extract features from the dataframe
        if features is None:
            # Combine categorical and continuous features from YAML for comprehensive analysis
            all_features = []
            
            # Add categorical features
            cat_features = [f for f in DEFAULT_CAT_FEATURES if f in df.columns]
            if cat_features:
                print(f"Including {len(cat_features)} categorical features from YAML config")
                all_features.extend(cat_features)
            
            # Add continuous features
            cont_features = [f for f in DEFAULT_CONT_FEATURES if f in df.columns]
            if cont_features:
                print(f"Including {len(cont_features)} continuous features from YAML config")
                all_features.extend(cont_features)
            
            # Remove duplicates while preserving order
            features = list(dict.fromkeys(all_features))
            print(f"Using {len(features)} total features from YAML config (cat + cont)")
        else:
            print(f"Using {len(features)} user-provided features")
        
        # Check if all features exist in the dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: The following features are not in the dataframe: {missing_features}")
            # Remove missing features from the list
            features = [f for f in features if f in df.columns]
            print(f"Proceeding with {len(features)} available features")
        
        # Filter the dataframe to include only the features
        feature_df = df[features].copy()
        
        # Drop rows with missing values for correlation analysis
        feature_df_clean = feature_df.dropna()
        print(f"Using {len(feature_df_clean)} complete rows for correlation analysis out of {len(feature_df)} total rows")
        
        # 2. Assess multicollinearity
        
        # Calculate correlation matrix
        correlation_matrix = feature_df_clean.corr(method='pearson')
        
        # Save correlation matrix to CSV
        corr_matrix_path = output_path / "correlation_matrix.csv"
        correlation_matrix.to_csv(corr_matrix_path)
        print(f"Saved correlation matrix to {corr_matrix_path}")
        
        # Identify highly correlated features (|r| > 0.7)
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_correlation_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        # Save high correlation pairs to CSV
        if high_correlation_pairs:
            high_corr_df = pd.DataFrame(high_correlation_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
            high_corr_path = output_path / "high_correlation_pairs.csv"
            high_corr_df.to_csv(high_corr_path, index=False)
            print(f"Saved {len(high_correlation_pairs)} high correlation pairs to {high_corr_path}")
            print("Top highly correlated pairs:")
            for i, row in high_corr_df.head(5).iterrows():
                print(f"  {row['feature1']} and {row['feature2']}: r = {row['correlation']:.3f}")
        else:
            print("No highly correlated feature pairs found (|r| > 0.7)")
            
        # 4. Produce heatmap among features
        try:
            # Set up the matplotlib figure
            plt.figure(figsize=(12, 10))
            
            # Create a heatmap
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                        linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
            
            # Adjust layout and save
            plt.tight_layout()
            heatmap_path = output_path / "correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved correlation heatmap to {heatmap_path}")
            
            # Create a clustered heatmap for better visualization
            plt.figure(figsize=(12, 10))
            clustered_heatmap = sns.clustermap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                                              linewidths=0.5, figsize=(12, 12),
                                              dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.05, 0.18))
            clustered_heatmap_path = output_path / "clustered_correlation_heatmap.png"
            plt.savefig(clustered_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved clustered correlation heatmap to {clustered_heatmap_path}")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
        
        # 5. Calculate AIC and BIC for Cox models using fast approach
        try:
            print("\n=== Calculating AIC and BIC for Cox Models (Fast Approach) ===\n")
            
            # Use column names from YAML config instead of hardcoding
            duration_col = DEFAULT_DURATION
            event_col = DEFAULT_EVENT
            
            # Ensure we have the required columns
            required_cols = [duration_col, event_col]
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                print(f"Error: Missing required columns for Cox models: {missing_required}")
                print("Skipping AIC/BIC calculation")
            else:
                # Use the fast feature selection pipeline
                from src.feature_eval import fast_feature_selection
                
                # Set up output directory for AIC/BIC results
                aic_bic_output_dir = output_path / "aic_bic_analysis"
                os.makedirs(aic_bic_output_dir, exist_ok=True)
                
                # Use all cat_feature and cont_feature from YAML for AIC/BIC calculation
                all_features = []
                
                # Add categorical features
                cat_features = [f for f in DEFAULT_CAT_FEATURES if f in df.columns]
                if cat_features:
                    print(f"Including {len(cat_features)} categorical features from YAML config for AIC/BIC")
                    all_features.extend(cat_features)
                
                # Add continuous features
                cont_features = [f for f in DEFAULT_CONT_FEATURES if f in df.columns]
                if cont_features:
                    print(f"Including {len(cont_features)} continuous features from YAML config for AIC/BIC")
                    all_features.extend(cont_features)
                
                # Remove duplicates while preserving order
                all_features = list(dict.fromkeys(all_features))
                print(f"Running fast feature selection on {len(all_features)} total features (cat + cont) for AIC/BIC...")
                
                # Run fast feature selection
                aic_bic_results = fast_feature_selection(
                    df=df,
                    duration_col=duration_col,
                    event_col=event_col,
                    features=all_features,  # Use all cat_feature and cont_feature from YAML
                    top_n=len(all_features),  # Calculate AIC/BIC for all features
                    output_dir=aic_bic_output_dir,
                    n_jobs=-1  # Use all available cores
                )
                
                # Update results dictionary
                results = update_results_with_new_outputs(
                    results,
                    aic_bic_path=aic_bic_output_dir / "fast_univariate_screening.csv",
                    aic_bic_stepwise_path=None  # No multivariable model results
                )
                
                print(f"Fast AIC/BIC analysis results saved to: {aic_bic_output_dir}")
                
        except ValueError as e:
            print(f"AIC/BIC calculation error: {e}")
        except ImportError as e:
            print(f"AIC/BIC calculation error: {e}")
        except Exception as e:
            print(f"Unexpected error in AIC/BIC calculation: {e}")
            import traceback
            traceback.print_exc()
        
        # Store the selected features for VIF calculation
        selected_features = []
        if 'aic_bic_results' in locals() and 'ic' in aic_bic_results:
            # Sort features by AIC (lower is better) and select top 10
            top_features_by_aic = aic_bic_results['ic'].sort_values('AIC').head(10)
            selected_features = top_features_by_aic['Feature'].tolist()
            print(f"Selected top 10 features by AIC for VIF calculation: {selected_features}")
        
        # 6. Calculate Variance Inflation Factor (VIF) for features from YAML
        # Use only the 'features' list from the YAML file as requested
        yaml_features = DEFAULT_FEATURES
        yaml_features_in_df = [f for f in yaml_features if f in feature_df_clean.columns]
        
        if yaml_features_in_df:
            print("\n=== Calculating VIF for Features from YAML ===\n")
            print(f"Using {len(yaml_features_in_df)} features from YAML config for VIF calculation")
            
            try:
                # Calculate VIF using the modular function from feature_eval
                vif_path = output_path / "yaml_features_vif.csv"
                vif_data = calculate_vif(
                    df=feature_df_clean,
                    features=yaml_features_in_df,  # Use features from YAML
                    vif_threshold=5.0,
                    save_path=vif_path,
                    return_dataframe=True,
                    print_high_vif=True
                )
                
                # Create a VIF plot
                vif_plot_path = output_path / "yaml_features_vif_plot.png"
                plot_vif(
                    vif_data=vif_data,
                    save_path=vif_plot_path,
                    threshold=5.0,
                    title="VIF Analysis for Features from YAML"
                )
                print(f"Saved VIF plot for YAML features to {vif_plot_path}")
                
            except ValueError as e:
                print(f"VIF calculation error for YAML features: {e}")
            except ImportError as e:
                print(f"VIF calculation error for YAML features: {e}")
            except Exception as e:
                print(f"Unexpected error in VIF calculation for YAML features: {e}")
        else:
            print("No features from YAML were found in the dataframe for VIF calculation")
        
        # 7. Perform Cox-LASSO regression
        try:
            print("\n=== Performing Cox-LASSO Regression ===\n")
            
            # Ensure we have the required columns
            required_cols = [DEFAULT_DURATION, DEFAULT_EVENT]
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                print(f"Error: Missing required columns for Cox-LASSO: {missing_required}")
                print("Skipping Cox-LASSO regression")
            else:
                # Get random seed from environment variables
                RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
                
                # Set up output directory for LASSO results
                lasso_output_dir = output_path / "lasso_analysis"
                
                # Use only the 'features' list from the YAML file as requested
                selected_features = DEFAULT_FEATURES
                
                # Filter to only include features that exist in the dataframe
                selected_features = [f for f in selected_features if f in df.columns]
                print(f"Using {len(selected_features)} features from YAML 'features' list for LASSO: {selected_features}")
                
                # Identify which of the selected features are categorical
                cat_features = []
                if 'cat_features' in master_df_mapping_config:
                    all_cat_features = master_df_mapping_config.get('cat_features')
                    # Only include categorical features that are in our selected features list
                    cat_features = [f for f in all_cat_features if f in selected_features]
                    if cat_features:
                        print(f"Found {len(cat_features)} categorical features among features: {cat_features}")
                    else:
                        print("No categorical features found among features")
                
                # Create a copy of the dataframe for encoding categorical features
                df_encoded = df.copy()
                
                # Encode categorical features using one-hot encoding
                if cat_features:
                    print("Encoding categorical features for LASSO regression...")
                    # Use pandas get_dummies for one-hot encoding
                    df_encoded = pd.get_dummies(df_encoded, columns=cat_features, drop_first=True)
                    print(f"After encoding, dataframe has {len(df_encoded.columns)} columns")
                    
                    # Get the names of the encoded categorical features
                    encoded_cat_features = [col for col in df_encoded.columns
                                           if any(col.startswith(f"{cat}_") for cat in cat_features)]
                    print(f"Created {len(encoded_cat_features)} encoded features from {len(cat_features)} categorical features")
                    
                    # Get the non-categorical features from the selected features (YAML 'features' list)
                    non_cat_features = [f for f in selected_features if f not in cat_features]
                    
                    # Combine non-categorical features and encoded categorical features
                    lasso_features = non_cat_features + encoded_cat_features
                    print(f"Using {len(lasso_features)} total features for LASSO regression")
                else:
                    # If no categorical features, just use the selected features (YAML 'features' list) as is
                    lasso_features = selected_features
                    print(f"Using {len(lasso_features)} features from YAML 'features' list for LASSO regression (no categorical features)")
                
                # Perform Cox-LASSO regression
                lasso_results = perform_cox_lasso(
                    df=df_encoded,
                    duration_col=DEFAULT_DURATION,
                    event_col=DEFAULT_EVENT,
                    features=lasso_features,
                    cv_folds=10,  # Reduced from 10 to 3 for faster computation
                    output_dir=lasso_output_dir,
                    random_state=RANDOM_SEED,
                    use_gpu=False,  # Use R implementation instead of GPU
                    use_r=True,     # Use R's glmnet package via rpy2
                    parallel=True   # Use parallel processing for cross-validation
                )
                
                print(f"Optimal alpha (regularization parameter): {lasso_results['optimal_alpha']:.6f}")
                print(f"LASSO analysis results saved to: {lasso_output_dir}")
                
        except ValueError as e:
            print(f"Cox-LASSO regression error: {e}")
        except ImportError as e:
            print(f"Cox-LASSO regression error: {e}")
        except Exception as e:
            print(f"Unexpected error in Cox-LASSO regression: {e}")
            import traceback
            traceback.print_exc()
        
        # 7. Create 2 cause-specific survival models using XGBoost and examine SHAP values
        try:
            print("\n=== Creating Cause-Specific XGBoost Survival Models and SHAP Analysis ===\n")
            
            # Get random seed from environment variables
            RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
            
            # Use only the 'features' list from the YAML file as requested
            model_features = DEFAULT_FEATURES.copy()
            # Filter to only include features that exist in the dataframe
            model_features = [f for f in model_features if f in df.columns]
            print(f"Using {len(model_features)} features from YAML 'features' list for XGBoost and SHAP: {model_features}")
            
            # Identify categorical features among the selected features
            cat_features = [f for f in DEFAULT_CAT_FEATURES if f in model_features]
            
            # Create a copy of the dataframe for encoding categorical features
            df_encoded = df.copy()
            encoded_cat_features = []
            
            # Encode categorical features if any
            if cat_features:
                print(f"Encoding {len(cat_features)} categorical features for XGBoost and SHAP analysis...")
                df_encoded = pd.get_dummies(df_encoded, columns=cat_features, drop_first=True)
                
                # Get the names of the encoded categorical features
                encoded_cat_features = [col for col in df_encoded.columns
                                      if any(col.startswith(f"{cat}_") for cat in cat_features)]
                print(f"Created {len(encoded_cat_features)} encoded features from {len(cat_features)} categorical features")
                
                # Get non-categorical features
                non_cat_features = [f for f in model_features if f not in cat_features]
                
                # Update model_features with non-categorical and encoded features
                model_features = non_cat_features + encoded_cat_features
                
                # Replace the original dataframe with the encoded one
                df = df_encoded
            
            print(f"Using {len(model_features)} features for SHAP analysis (no VIF filtering)")
            
            # Ensure we have the required columns
            required_cols = [DEFAULT_DURATION, DEFAULT_EVENT]
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                print(f"Error: Missing required columns for survival analysis: {missing_required}")
                print("Skipping survival model and SHAP analysis")
            else:
                # Filter rows with missing values in features or target variables
                model_df = df[model_features + required_cols].dropna()
                print(f"Using {len(model_df)} complete rows for survival analysis")
                
                if len(model_df) < 50:
                    print("Warning: Sample size too small for reliable survival analysis")
                    print("Skipping survival model and SHAP analysis")
                else:
                    # Extract features and target variables
                    X = model_df[model_features]
                    
                    # Common XGBoost parameters
                    xgb_params = {
                        "objective": "survival:cox",
                        "tree_method": "hist",      # Use histogram-based algorithm
                        "device": "cuda",           # Use GPU acceleration
                        "eval_metric": "cox-nloglik",
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "seed": RANDOM_SEED
                    }
                    
                    # ----- Dialysis -------------------------------------------------------
                    print("Building XGBoost Survival model for Dialysis outcome...")
                    
                    # Prepare data for dialysis outcome
                    dialysis_events = (model_df["endpoint"] == 1).astype(int).values
                    dialysis_durations = model_df["duration"].values
                    
                    # Create DMatrix for XGBoost
                    dtrain_dial = xgb.DMatrix(X, label=dialysis_durations, weight=dialysis_events)
                    
                    # Train XGBoost model
                    bst_dial = xgb.train(xgb_params, dtrain_dial, num_boost_round=500)
                    
                    # Get feature importance
                    dial_importance = pd.DataFrame({
                        'Feature': model_features,
                        'Importance': [bst_dial.get_score(importance_type='gain').get(f, 0) for f in model_features]
                    }).sort_values('Importance', ascending=False)
                    
                    # Save feature importance to CSV
                    dial_importance_path = output_path / "dialysis_feature_importance.csv"
                    dial_importance.to_csv(dial_importance_path, index=False)
                    print(f"Saved dialysis feature importance to {dial_importance_path}")
                    
                    # Calculate SHAP values for dialysis model
                    print("Calculating SHAP values for Dialysis model...")
                    explainer_dial = shap.TreeExplainer(bst_dial)
                    shap_values_dial = explainer_dial.shap_values(X)
                    
                    # Plot SHAP summary for dialysis
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values_dial, X, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance for Dialysis Outcome")
                    dial_shap_path = output_path / "dialysis_shap_summary.png"
                    plt.savefig(dial_shap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved dialysis SHAP summary to {dial_shap_path}")
                    
                    # Plot SHAP beeswarm plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values_dial, X, show=False)
                    plt.title("SHAP Values for Dialysis Outcome")
                    dial_shap_beeswarm_path = output_path / "dialysis_shap_beeswarm.png"
                    plt.savefig(dial_shap_beeswarm_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved dialysis SHAP beeswarm plot to {dial_shap_beeswarm_path}")
                    
                    # ----- Mortality ------------------------------------------------------
                    print("Building XGBoost Survival model for Mortality outcome...")
                    
                    # Prepare data for mortality outcome
                    mortality_events = (model_df["endpoint"] == 2).astype(int).values
                    mortality_durations = model_df["duration"].values
                    
                    # Create DMatrix for XGBoost
                    dtrain_mort = xgb.DMatrix(X, label=mortality_durations, weight=mortality_events)
                    
                    # Train XGBoost model
                    bst_mort = xgb.train(xgb_params, dtrain_mort, num_boost_round=500)
                    
                    # Get feature importance
                    mort_importance = pd.DataFrame({
                        'Feature': model_features,
                        'Importance': [bst_mort.get_score(importance_type='gain').get(f, 0) for f in model_features]
                    }).sort_values('Importance', ascending=False)
                    
                    # Save feature importance to CSV
                    mort_importance_path = output_path / "mortality_feature_importance.csv"
                    mort_importance.to_csv(mort_importance_path, index=False)
                    print(f"Saved mortality feature importance to {mort_importance_path}")
                    
                    # Calculate SHAP values for mortality model
                    print("Calculating SHAP values for Mortality model...")
                    explainer_mort = shap.TreeExplainer(bst_mort)
                    shap_values_mort = explainer_mort.shap_values(X)
                    
                    # Plot SHAP summary for mortality
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values_mort, X, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance for Mortality Outcome")
                    mort_shap_path = output_path / "mortality_shap_summary.png"
                    plt.savefig(mort_shap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved mortality SHAP summary to {mort_shap_path}")
                    
                    # Plot SHAP beeswarm plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values_mort, X, show=False)
                    plt.title("SHAP Values for Mortality Outcome")
                    mort_shap_beeswarm_path = output_path / "mortality_shap_beeswarm.png"
                    plt.savefig(mort_shap_beeswarm_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved mortality SHAP beeswarm plot to {mort_shap_beeswarm_path}")
                    
                    # Create a combined feature importance plot
                    plt.figure(figsize=(12, 10))
                    
                    # Merge the two importance dataframes
                    combined_importance = pd.merge(
                        dial_importance, mort_importance,
                        on='Feature', suffixes=('_dialysis', '_mortality')
                    )
                    
                    # Sort by the sum of importances
                    combined_importance['Total'] = combined_importance['Importance_dialysis'] + combined_importance['Importance_mortality']
                    combined_importance = combined_importance.sort_values('Total', ascending=True)
                    
                    # Plot horizontal bar chart
                    features = combined_importance['Feature'].tolist()
                    y_pos = np.arange(len(features))
                    
                    plt.barh(y_pos - 0.2, combined_importance['Importance_dialysis'], 0.4, label='Dialysis', color='#1f77b4')
                    plt.barh(y_pos + 0.2, combined_importance['Importance_mortality'], 0.4, label='Mortality', color='#ff7f0e')
                    
                    plt.yticks(y_pos, features)
                    plt.xlabel('Feature Importance (Gain)')
                    plt.title('Feature Importance Comparison: Dialysis vs Mortality')
                    plt.legend()
                    plt.tight_layout()
                    
                    combined_importance_path = output_path / "combined_feature_importance.png"
                    plt.savefig(combined_importance_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved combined feature importance plot to {combined_importance_path}")
                    
                    # Save the combined importance data
                    combined_importance_csv_path = output_path / "combined_feature_importance.csv"
                    combined_importance.to_csv(combined_importance_csv_path, index=False)
                    print(f"Saved combined feature importance data to {combined_importance_csv_path}")
                    
        except Exception as e:
            print(f"Error in survival analysis and SHAP calculation: {e}")
            import traceback
            traceback.print_exc()
        
        # Return results
        results = {
            "correlation_matrix_path": str(corr_matrix_path),
            "high_correlation_pairs": high_correlation_pairs,
            "heatmap_path": str(heatmap_path) if 'heatmap_path' in locals() else None,
            "clustered_heatmap_path": str(clustered_heatmap_path) if 'clustered_heatmap_path' in locals() else None,
            "vif_path": str(vif_path) if 'vif_path' in locals() else None,
            "vif_plot_path": str(vif_plot_path) if 'vif_plot_path' in locals() else None,
            # Add SHAP analysis results
            "dialysis_feature_importance_path": str(dial_importance_path) if 'dial_importance_path' in locals() else None,
            "dialysis_shap_path": str(dial_shap_path) if 'dial_shap_path' in locals() else None,
            "dialysis_shap_beeswarm_path": str(dial_shap_beeswarm_path) if 'dial_shap_beeswarm_path' in locals() else None,
            "mortality_feature_importance_path": str(mort_importance_path) if 'mort_importance_path' in locals() else None,
            "mortality_shap_path": str(mort_shap_path) if 'mort_shap_path' in locals() else None,
            "mortality_shap_beeswarm_path": str(mort_shap_beeswarm_path) if 'mort_shap_beeswarm_path' in locals() else None,
            "combined_feature_importance_path": str(combined_importance_path) if 'combined_importance_path' in locals() else None,
            "mortality_shap_beeswarm_path": str(mort_shap_beeswarm_path) if 'mort_shap_beeswarm_path' in locals() else None,
            "combined_feature_importance_path": str(combined_importance_path) if 'combined_importance_path' in locals() else None,
            "combined_feature_importance_csv_path": str(combined_importance_csv_path) if 'combined_importance_csv_path' in locals() else None
        }
        
        # Update results with AIC/BIC and LASSO outputs
        results = update_results_with_new_outputs(
            results,
            aic_bic_path=aic_bic_path if 'aic_bic_path' in locals() else None,
            aic_bic_stepwise_path=aic_bic_stepwise_path if 'aic_bic_stepwise_path' in locals() else None,
            aic_bic_categorical_path=cat_aic_bic_path if 'cat_aic_bic_path' in locals() else None,
            lasso_output_dir=lasso_output_dir if 'lasso_output_dir' in locals() else None,
            lasso_results=lasso_results if 'lasso_results' in locals() else None
        )
        
        # 8. Create final correlation heatmap using only selected features
        try:
            print("\n=== Creating Final Correlation Heatmap with Selected Features ===\n")
            
            # Use only the 'features' list from YAML
            selected_features = DEFAULT_FEATURES
            selected_features = [f for f in selected_features if f in feature_df_clean.columns]
            
            if len(selected_features) > 1:
                # Calculate correlation matrix for selected features
                selected_correlation_matrix = feature_df_clean[selected_features].corr(method='pearson')
                
                # Create a heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(selected_correlation_matrix, annot=True, cmap='coolwarm', center=0,
                            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
                
                # Add labels and title
                plt.title("Correlation Heatmap for Selected Features")
                plt.tight_layout()
                
                # Save the heatmap
                selected_heatmap_path = output_path / "selected_features_correlation_heatmap.png"
                plt.savefig(selected_heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved correlation heatmap for selected features to {selected_heatmap_path}")
                
                # Create a clustered heatmap for better visualization
                plt.figure(figsize=(12, 10))
                clustered_selected_heatmap = sns.clustermap(selected_correlation_matrix, annot=True, cmap='coolwarm', center=0,
                                                  linewidths=0.5, figsize=(12, 12),
                                                  dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.05, 0.18))
                clustered_selected_heatmap_path = output_path / "clustered_selected_features_correlation_heatmap.png"
                plt.savefig(clustered_selected_heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved clustered correlation heatmap for selected features to {clustered_selected_heatmap_path}")
                
                # Add to results
                results["selected_heatmap_path"] = str(selected_heatmap_path)
                results["clustered_selected_heatmap_path"] = str(clustered_selected_heatmap_path)
            else:
                print("Not enough selected features for correlation heatmap (need at least 2)")
        except Exception as e:
            print(f"Error creating final correlation heatmap: {e}")
        
        print("\n=== Feature Assessment Completed ===\n")
        
        return results
        
    except Exception as e:
        print(f"Error in feature selection: {e}")
        import traceback
        traceback.print_exc()
        return {}