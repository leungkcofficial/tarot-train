"""
Train and Deploy Single Model Step for Ensemble Deployment

This module contains the ZenML step for training and deploying a single model
based on its configuration.
"""

import os
import json
import tempfile
from datetime import datetime
from zenml.steps import step
from typing import Dict, Any, Tuple
import pandas as pd


@step(enable_cache=False)  # Disable cache since we're training models
def train_and_deploy_single_model(
    model_config: Dict[str, Any],
    train_df_preprocessed: pd.DataFrame,
    temporal_test_df_preprocessed: pd.DataFrame,
    spatial_test_df_preprocessed: pd.DataFrame,
    master_df_mapping_path: str = "src/default_master_df_mapping.yml"
) -> Tuple[Dict[str, Any], str, str]:
    """
    Train and deploy a single model based on its configuration.
    
    This step:
    1. Creates a model-specific hyperparameter configuration
    2. Temporarily saves it to a YAML file
    3. Calls the model_deploy step with the configuration
    4. Returns deployment details and prediction file paths
    
    Args:
        model_config: Model configuration from load_model_configurations
        train_df_preprocessed: Preprocessed training data
        temporal_test_df_preprocessed: Preprocessed temporal test data
        spatial_test_df_preprocessed: Preprocessed spatial test data
        master_df_mapping_path: Path to master dataframe mapping
        
    Returns:
        Tuple of (deployment_details, temporal_predictions_path, spatial_predictions_path)
    """
    from src.model_config_utils import create_model_specific_config, save_model_specific_config, get_model_identifier
    from src.util import load_yaml_file
    from steps.model_deploy import deploy_model
    
    model_identifier = get_model_identifier(model_config)
    print(f"\n=== Training and Deploying {model_identifier} ===")
    
    # Create model-specific configuration
    model_specific_config = create_model_specific_config(model_config)
    
    # Display configuration details
    print(f"Model No: {model_config['model_no']}")
    print(f"Algorithm: {model_config['algorithm']}")
    print(f"Structure: {model_config['structure']}")
    print(f"Target Endpoint: {model_config.get('target_endpoint', 'Both')}")
    print(f"Balancing Method: {model_config['balancing_method']}")
    print(f"Optimization Target: {model_config['optimization_target']}")
    
    # Create a temporary hyperparameter config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp_config:
        temp_config_path = tmp_config.name
        save_model_specific_config(model_specific_config, temp_config_path)
    
    try:
        # Temporarily replace the hyperparameter config
        original_config_path = "src/hyperparameter_config.yml"
        
        # Backup original config if it exists
        original_config = None
        if os.path.exists(original_config_path):
            original_config = load_yaml_file(original_config_path)
        
        # Save model-specific config to the expected location
        save_model_specific_config(model_specific_config, original_config_path)
        
        # Prepare model metadata and optimization metrics paths
        model_metadata_path = os.path.join(
            "results/final_deploy/model_config",
            model_config['details_json_file']
        )
        
        optimization_metrics_path = None
        if model_config.get('optim_json_file'):
            optimization_metrics_path = os.path.join(
                "results/final_deploy/model_config",
                model_config['optim_json_file']
            )
        
        # Call deploy_model step
        print(f"\nCalling deploy_model with:")
        print(f"  Model metadata: {model_metadata_path}")
        print(f"  Optimization metrics: {optimization_metrics_path}")
        print(f"  Target endpoint: {model_config.get('target_endpoint')}")
        
        deployment_details = deploy_model(
            model_metadata=model_metadata_path,
            optimization_metrics=optimization_metrics_path,
            model_name=f"Ensemble_{model_identifier}",
            model_stage="Development",
            register_model=False,  # Don't register individual models
            master_df_mapping_path=master_df_mapping_path,
            model_endpoint=model_config.get('target_endpoint'),
            train_df_preprocessed=train_df_preprocessed,
            temporal_test_df_preprocessed=temporal_test_df_preprocessed,
            spatial_test_df_preprocessed=spatial_test_df_preprocessed,
            batch_size=1000,
            cv_folds=10
        )
        
        # Extract prediction file paths from deployment details
        temporal_predictions_path = deployment_details.get('temporal_predictions_path', '')
        spatial_predictions_path = deployment_details.get('spatial_predictions_path', '')
        
        print(f"\nModel {model_identifier} deployed successfully")
        print(f"Temporal predictions: {temporal_predictions_path}")
        print(f"Spatial predictions: {spatial_predictions_path}")
        
        # Add model configuration to deployment details for tracking
        deployment_details['model_config'] = model_config
        deployment_details['model_identifier'] = model_identifier
        
        return deployment_details, temporal_predictions_path, spatial_predictions_path
        
    finally:
        # Restore original config if it existed
        if original_config is not None:
            save_model_specific_config(original_config, original_config_path)
        
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)