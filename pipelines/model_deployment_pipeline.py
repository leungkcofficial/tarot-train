"""
Model Deployment Pipeline for CKD Risk Prediction

This module defines the ZenML pipeline for training the best model with optimized hyperparameters,
evaluating it, and exporting it for deployment.
"""

import os
import glob
from zenml.pipelines import pipeline
from typing import Dict, Any, List

def get_latest_file(directory, prefix):
    """Get the most recently created file with the given prefix in the directory.
    
    Args:
        directory: Directory to search in
        prefix: Prefix of the filename to search for
        
    Returns:
        Path to the most recently created file, or None if no files found
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return None
    
    # Get all files with the given prefix
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No files found matching {pattern}")
        return None
    
    # Return the most recently created file
    latest_file = max(files, key=os.path.getctime)
    print(f"Found latest file: {latest_file}")
    return latest_file


@pipeline(enable_cache=True)
def model_deployment_pipeline(
    best_hyperparams: Dict[str, Any] = None,
    run_model_comparison: bool = True
):
    """
    Pipeline for training the best model with optimized hyperparameters and deploying it.
    
    This pipeline connects the following steps:
    1. Train the model with the best hyperparameters
    2. Evaluate the model
    3. Export the model
    4. Register the model
    5. Run model comparison if requested
    
    The model type is read from the hyperparameter configuration file.
    
    Args:
        best_hyperparams: Dictionary of best hyperparameters from optimization
        run_model_comparison: Whether to run model comparison
        
    Returns:
        A dictionary containing model information and evaluation results
    """
    # Import steps here to avoid circular imports
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.kfre_eval import kfre_eval
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    # Import deep learning survival modeling steps
    from steps.survival_dataset_builder import survival_dataset_builder
    from steps.model_train import train_model
    from steps.model_deploy import deploy_model
    from steps.model_eval import eval_model
    from steps.dl_model_comparison import dl_model_comparison
    
    # Define the pipeline steps
    # ingest_data now returns multiple DataFrames as separate outputs
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Pass all DataFrames to clean_data, which returns a tuple of cleaned dataframes
    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df, cci_score_df, hypertension_df, egfr_df = clean_data(
        cr_df=cr_df,
        hb_df=hb_df,
        a1c_df=a1c_df,
        alb_df=alb_df,
        po4_df=po4_df,
        ca_df=ca_df,
        ca_adjusted_df=ca_adjusted_df,
        hco3_df=hco3_df,
        upcr_df=upcr_df,
        uacr_df=uacr_df,
        icd10_df=icd10_df,
        operation_df=operation_df,
        death_df=death_df,
        demo_df=demo_df
    )
    
    # Merge the cleaned dataframes into a master dataframe
    final_df, prediction_df = merge_data(
        patient_df=patient_df,
        icd10_df=icd10_df_clean,
        cr_df=cr_df_clean,
        hb_df=hb_df_clean,
        a1c_df=a1c_df_clean,
        alb_df=alb_df_clean,
        po4_df=po4_df_clean,
        ca_df=ca_df_clean,
        ca_adjusted_df=ca_adjusted_df_clean,
        hco3_df=hco3_df_clean,
        upcr_df=upcr_df_clean,
        uacr_df=uacr_df_clean,
        operation_df=operation_df_clean,
        death_df=death_df_clean,
        cci_df=cci_df,
        cci_score_df=cci_score_df,
        hypertension_df=hypertension_df,
        egfr_df=egfr_df
    )
    
    # Split the merged data into training, temporal test, and spatial test sets
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df)
    
    # Calculate KFRE for all model types (as baseline)
    kfre_results = kfre_eval(
        train_df=train_df,
        spatial_test_df=spatial_test_df,
        temporal_test_df=temporal_test_df
    )
    
    # Impute missing data via MICE
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # Read configuration from YAML files
    from src.util import load_yaml_file
    
    # Load the default master dataframe mapping
    df_mapping = load_yaml_file("src/default_master_df_mapping.yml")
    
    # Get features from the mapping
    feature_cols = df_mapping.get("features", [])
    print(f"Using {len(feature_cols)} features from default_master_df_mapping.yml: {feature_cols}")
    
    # Load hyperparameter configuration from YAML file
    hyperparameter_config = load_yaml_file("src/hyperparameter_config.yml")
    
    # Get model type from configuration
    model_type = hyperparameter_config.get("model_type", "deepsurv")
    print(f"Using model type from configuration: {model_type}")
    
    # Validate model type
    if model_type.lower() not in ["deepsurv", "deephit"]:
        raise ValueError(f"Unsupported model type in configuration: {model_type}. Supported types are 'deepsurv' and 'deephit'.")
    
    # Get model configuration
    network_config = hyperparameter_config.get("network", {})
    default_config = network_config.get("default", {})
    model_specific_config = network_config.get(model_type.lower(), {})
    
    # Common parameters
    epochs = int(os.getenv("EPOCHS", default_config.get("epochs", 100)))
    
    # DeepHit specific parameters
    time_grid = model_specific_config.get("time_grid", [30, 60, 90, 180, 365, 730, 1095, 1460, 1825])
    
    # Time horizons for evaluation (using time_grid from configuration)
    time_horizons = time_grid
    
    # Output directory
    base_results_dir = "/mnt/dump/yard/projects/tarot2/results"
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(base_results_dir, "model_output"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model type: {model_type}")
    print(f"Output directory: {output_dir}")
    
    # If using deep learning survival models (DeepSurv or DeepHit)
    if model_type.lower() in ["deepsurv", "deephit"]:
        # Convert DataFrames to PyCox datasets
        train_ds, val_ds, test_ds, feature_metadata, labeller, num_features = survival_dataset_builder(
            train_df=train_df_preprocessed,
            temporal_test_df=temporal_test_df_preprocessed,
            spatial_test_df=spatial_test_df_preprocessed,
            features=feature_cols,
            duration_col="duration",
            event_col="endpoint"
        )
        
        # Use the number of features returned directly from the dataset builder
        input_dim = num_features
        print(f"Using {input_dim} features from dataset (as returned by dataset builder)")
        
        # Load hyperparameter configuration from YAML file
        from src.util import load_yaml_file
        hyperparameter_config = load_yaml_file("src/hyperparameter_config.yml")
        
        # Use default hyperparameters if none provided
        if best_hyperparams is None:
            # Get default model parameters
            network_config = hyperparameter_config.get("network", {})
            default_config = network_config.get("default", {})
            
            # Common parameters with fallbacks
            best_hyperparams = {
                "learning_rate": default_config.get("learning_rate", 0.001),
                "num_layers": default_config.get("num_layers", 3),
                "hidden_units": default_config.get("hidden_dims", [128, 64, 32])[0],
                "dropout": default_config.get("dropout", 0.2),
                "optimizer": "Adam",
                "batch_size": default_config.get("batch_size", 64)
            }
            
            # Add model-specific default hyperparameters
            if model_type.lower() == "deephit":
                model_specific_config = network_config.get(model_type.lower(), {})
                best_hyperparams["alpha"] = model_specific_config.get("alpha", 0.5)
                best_hyperparams["sigma"] = model_specific_config.get("sigma", 0.5)
            
            print(f"Using default hyperparameters from configuration: {best_hyperparams}")
        
        # Create a temporary dataframe with the necessary structure for model_train
        # This is a workaround to use the original model_train interface
        from steps.survival_dataset_builder import prepare_survival_dataset
        
        # Train model with best hyperparameters
        # We need to modify the hyperparameter_config to include the best hyperparameters
        import tempfile
        import yaml
        
        # Create a temporary hyperparameter config file with the best hyperparameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            # Load the original hyperparameter config
            hp_config = load_yaml_file("src/hyperparameter_config.yml")
            
            # Update with best hyperparameters if provided
            if best_hyperparams is not None:
                # Update network default parameters
                if 'network' not in hp_config:
                    hp_config['network'] = {}
                if 'default' not in hp_config['network']:
                    hp_config['network']['default'] = {}
                
                # Update common parameters
                hp_config['network']['default']['learning_rate'] = best_hyperparams.get('learning_rate', 0.001)
                hp_config['network']['default']['hidden_dims'] = [best_hyperparams.get('hidden_units', 128)]
                hp_config['network']['default']['dropout'] = best_hyperparams.get('dropout', 0.2)
                hp_config['network']['default']['batch_size'] = best_hyperparams.get('batch_size', 64)
                
                # Update model-specific parameters
                if model_type.lower() == 'deephit':
                    if 'deephit' not in hp_config['network']:
                        hp_config['network']['deephit'] = {}
                    hp_config['network']['deephit']['alpha'] = best_hyperparams.get('alpha', 0.5)
                    hp_config['network']['deephit']['sigma'] = best_hyperparams.get('sigma', 0.5)
            
            # Write the updated config to the temporary file
            yaml.dump(hp_config, temp_file)
            temp_config_path = temp_file.name
        
        # Train model with the temporary config file
        model_details, optimization_metrics = train_model(
            train_df_preprocessed=train_df_preprocessed,
            master_df_mapping_path="src/default_master_df_mapping.yml",
            hyperparameter_config_path=temp_config_path
        )
        
        # Clean up the temporary file
        import os
        os.unlink(temp_config_path)
        
        # Get the latest model details and optimization metrics files
        results_dir = "/mnt/dump/yard/projects/tarot2/results/model_details"
        model_details_path = get_latest_file(results_dir, "model_details")
        optimization_metrics_path = get_latest_file(results_dir, "optimization_metrics")
        
        # Deploy model using the file paths
        print(f"Deploying model using files:\n- Model details: {model_details_path}\n- Optimization metrics: {optimization_metrics_path}")
        deployed_model_details = deploy_model(
            model_metadata=model_details_path,
            optimization_metrics=optimization_metrics_path,
            model_name=f"CKD_{model_type.upper()}",
            model_stage="Staging",
            register_model=True,
            train_df_preprocessed=train_df_preprocessed,
            temporal_test_df_preprocessed=temporal_test_df_preprocessed,
            spatial_test_df_preprocessed=spatial_test_df_preprocessed
        )
        
        # Evaluate model separately using the deployed model details
        print("\n=== Evaluating model separately ===")
        evaluation_results = eval_model(
            deployed_model_details=deployed_model_details,
            train_df=train_df_preprocessed,
            temporal_test_df=temporal_test_df_preprocessed,
            spatial_test_df=spatial_test_df_preprocessed,
            n_bootstrap=100,
            visualize=True
        )
        
        # Run model comparison if requested
        if run_model_comparison:
            comparison_results = dl_model_comparison(
                test_df=spatial_test_df_preprocessed,
                test_ds=test_ds,
                model_type=model_type,
                deployed_model_details=deployed_model_details,
                kfre_predictions=None,  # We can't access kfre_results as a dictionary
                time_horizons=time_horizons,
                output_dir=output_dir,
                seed=42
            )
            
            return {
                "model_type": model_type,
                "model_details": model_details,
                "optimization_metrics": optimization_metrics,
                "deployed_model_details": deployed_model_details,
                "evaluation_results": evaluation_results,
                "comparison_results": comparison_results
            }
        
        return {
            "model_type": model_type,
            "model_details": model_details,
            "optimization_metrics": optimization_metrics,
            "deployed_model_details": deployed_model_details,
            "evaluation_results": evaluation_results
        }
    
    elif model_type.lower() == "kfre":
        # For KFRE, we don't need to train a model
        # We just use the KFRE evaluation results directly
        print(f"\n=== Using KFRE as baseline (no training required) ===\n")
        
        # KFRE results were already calculated earlier in the pipeline
        if kfre_results:
            print("Using previously calculated KFRE results")
            return kfre_results
        else:
            print("Warning: KFRE results not available")
            return {
                "model_type": "kfre",
                "error": "KFRE results not available",
                "message": "KFRE evaluation was performed but results are not available"
            }
    else:
        # Unsupported model type
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'deepsurv', 'deephit', and 'kfre'.")