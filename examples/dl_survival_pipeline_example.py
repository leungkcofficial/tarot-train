"""
Example script for running the CKD risk prediction pipeline

This script demonstrates how to:
1. Run the training pipeline to find the best hyperparameters
2. Use those hyperparameters to train and deploy the best model
"""

import os
import mlflow
from zenml.client import Client
from pipelines.training_pipeline import train_pipeline
from pipelines.model_deployment_pipeline import model_deployment_pipeline


def main():
    """
    Run the CKD risk prediction pipeline
    """
    # Set output directory to be under the results directory
    base_results_dir = "/mnt/dump/yard/projects/tarot2/results"
    os.environ["OUTPUT_DIR"] = os.path.join(base_results_dir, "model_output")
    
    # Ensure results directory exists
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Note: Model configuration is now loaded from src/hyperparameter_config.yml
    # Note: Feature columns are read from src/default_master_df_mapping.yml

    # Initialize ZenML client
    client = Client()
    
    # Load model type from configuration
    from src.util import load_yaml_file
    hyperparameter_config = load_yaml_file("src/hyperparameter_config.yml")
    model_type = hyperparameter_config.get("model_type", "deepsurv")
    
    print(f"\n=== Step 1: Running hyperparameter optimization for {model_type.upper()} ===\n")
    
    # Run the training pipeline to find the best hyperparameters
    pipeline_instance = train_pipeline(
        run_hyperparameter_optimization=True
    )
    
    # Get the pipeline run
    pipeline_run = client.get_pipeline_run(pipeline_instance.id)
    
    # Get the output of the pipeline run
    pipeline_output = pipeline_run.get_output()
    
    # Extract the best hyperparameters
    best_hyperparams = pipeline_output.get("best_hyperparams", {})
    input_dim = pipeline_output.get("input_dim")
    time_grid = pipeline_output.get("time_grid")
    
    print(f"\nBest hyperparameters for {model_type.upper()}:")
    for param, value in best_hyperparams.items():
        print(f"  {param}: {value}")
    
    print(f"\n=== Step 2: Training and deploying the best {model_type.upper()} model ===\n")
    
    # Run the model deployment pipeline with the best hyperparameters
    deployment_pipeline_instance = model_deployment_pipeline(
        best_hyperparams=best_hyperparams,
        run_model_comparison=True
    )
    
    # Get the deployment pipeline run
    deployment_pipeline_run = client.get_pipeline_run(deployment_pipeline_instance.id)
    
    # Get the output of the deployment pipeline run
    deployment_output = deployment_pipeline_run.get_output()
    
    # Print the evaluation results
    evaluation_results = deployment_output.get("evaluation_results", {})
    print("\nModel evaluation results:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value}")
    
    # Print the export paths
    export_paths = deployment_output.get("export_paths", {})
    print("\nExported model paths:")
    for path_type, path in export_paths.items():
        print(f"  {path_type}: {path}")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()