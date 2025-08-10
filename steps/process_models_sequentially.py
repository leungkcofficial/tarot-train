"""
Process Models Sequentially Step for Ensemble Deployment

This module contains the ZenML step for processing multiple models sequentially,
reading configurations from a JSON file to avoid ZenML artifact access issues.
"""

import json
import os
from zenml.steps import step
from typing import List, Dict, Any, Tuple
import pandas as pd


@step(enable_cache=True)
def process_models_sequentially(
    config_json_path: str,
    train_df_preprocessed: pd.DataFrame,
    temporal_test_df_preprocessed: pd.DataFrame,
    spatial_test_df_preprocessed: pd.DataFrame,
    master_df_mapping_path: str = "src/default_master_df_mapping.yml"
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Process all models sequentially, training and deploying each one.
    
    This step reads model configurations from a JSON file and processes
    each model one by one to save memory.
    
    Args:
        config_json_path: Path to the JSON file containing model configurations
        train_df_preprocessed: Preprocessed training data
        temporal_test_df_preprocessed: Preprocessed temporal test data
        spatial_test_df_preprocessed: Preprocessed spatial test data
        master_df_mapping_path: Path to master dataframe mapping
        
    Returns:
        Tuple of (deployment_details, temporal_predictions_paths, spatial_predictions_paths)
    """
    from steps.train_and_deploy_single_model import train_and_deploy_single_model
    
    print(f"\n=== Processing Models Sequentially ===")
    print(f"Loading configurations from: {config_json_path}")
    
    # Load model configurations from JSON file
    with open(config_json_path, 'r') as f:
        model_configs = json.load(f)
    
    print(f"Found {len(model_configs)} model configurations")
    
    # Store results
    all_deployment_details = []
    all_temporal_predictions = []
    all_spatial_predictions = []
    
    # Process each model
    for i, model_config in enumerate(model_configs):
        print(f"\n--- Processing Model {i+1}/{len(model_configs)} ---")
        print(f"Model {model_config['model_no']}: {model_config['algorithm']} - {model_config['structure']}")
        
        try:
            # Train and deploy single model
            deployment_details, temporal_pred_path, spatial_pred_path = train_and_deploy_single_model(
                model_config=model_config,
                train_df_preprocessed=train_df_preprocessed,
                temporal_test_df_preprocessed=temporal_test_df_preprocessed,
                spatial_test_df_preprocessed=spatial_test_df_preprocessed,
                master_df_mapping_path=master_df_mapping_path
            )
            
            # Store results
            all_deployment_details.append(deployment_details)
            all_temporal_predictions.append(temporal_pred_path)
            all_spatial_predictions.append(spatial_pred_path)
            
            print(f"Model {model_config['model_no']} processed successfully")
            
        except Exception as e:
            print(f"Error processing model {model_config['model_no']}: {str(e)}")
            print(f"Continuing with next model...")
            # Continue with next model even if one fails
            continue
    
    print(f"\n=== Successfully processed {len(all_deployment_details)} out of {len(model_configs)} models ===")
    
    return all_deployment_details, all_temporal_predictions, all_spatial_predictions