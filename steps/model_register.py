"""
Model Registration Step for CKD Risk Prediction

This module contains the ZenML step for registering trained deep learning survival analysis models
(DeepSurv and DeepHit) in MLflow for model versioning and tracking.
"""

import os
import json
import mlflow
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


@step
def model_register(
    model_paths: Dict[str, str],
    model_params: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    model_name: Optional[str] = None,
    register_model: bool = True,
    model_stage: str = "Staging"
) -> Dict[str, Any]:
    """
    Register a trained survival model in MLflow.
    
    Args:
        model_paths: Dictionary containing paths to model files
        model_params: Model parameters and hyperparameters
        evaluation_results: Model evaluation results
        model_name: Name for the registered model (default: None, will use model_type)
        register_model: Whether to register the model in MLflow (default: True)
        model_stage: Stage for the registered model (default: "Staging")
        
    Returns:
        Dictionary containing registration information
    """
    try:
        # Extract model parameters
        model_type = model_params["model_type"]
        
        # Set model name if not provided
        if model_name is None:
            model_name = f"CKD_{model_type.upper()}"
        
        print(f"\n=== Registering {model_name} Model ===\n")
        
        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get model directory
        model_dir = model_paths["model_dir"]
        
        # Log model files as artifacts
        for key, path in model_paths.items():
            if key != "model_dir" and os.path.exists(path):
                mlflow.log_artifact(path)
        
        # Log model parameters
        for key, value in model_params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
        
        # Log evaluation metrics
        if "metrics" in evaluation_results:
            for key, value in evaluation_results["metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        
        # Log evaluation artifacts
        if "artifacts" in evaluation_results:
            for key, path in evaluation_results["artifacts"].items():
                if isinstance(path, str) and os.path.exists(path):
                    mlflow.log_artifact(path)
                elif isinstance(path, list):
                    for p in path:
                        if os.path.exists(p):
                            mlflow.log_artifact(p)
        
        # Register model in MLflow
        registered_model = None
        if register_model:
            # Get PyTorch model path
            torch_path = model_paths["torch_path"]
            
            # Register model
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/{os.path.basename(torch_path)}",
                name=model_name
            )
            
            # Set model stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage=model_stage
            )
            
            print(f"Registered model {model_name} version {registered_model.version} as {model_stage}")
        
        # Create registration info
        registration_info = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": timestamp,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        
        if registered_model is not None:
            registration_info["registered_model_version"] = registered_model.version
            registration_info["model_stage"] = model_stage
        
        # Create registration summary
        summary = {
            "registration_info": registration_info,
            "model_params": model_params,
            "evaluation_metrics": evaluation_results.get("metrics", {}),
            "model_paths": {k: v for k, v in model_paths.items() if k != "model_dir"}
        }
        
        # Save registration summary
        summary_path = os.path.join(model_dir, "registration_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Saved registration summary to {summary_path}")
        
        # Log registration summary
        mlflow.log_artifact(summary_path)
        
        return summary
        
    except Exception as e:
        print(f"Error registering model: {e}")
        import traceback
        traceback.print_exc()
        raise