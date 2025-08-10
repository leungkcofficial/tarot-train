"""
Model Export Step for CKD Risk Prediction

This module contains the ZenML step for exporting trained deep learning survival analysis models
(DeepSurv and DeepHit) to PyTorch and ONNX formats for deployment.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import mlflow
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import neural network architectures
from src.nn_architectures import create_network


@step
def model_export(
    model: Any,
    model_params: Dict[str, Any],
    output_dir: str = "model_output",
    export_onnx: bool = True,
    example_input: Optional[np.ndarray] = None,
    seed: int = 42
) -> Dict[str, str]:
    """
    Export a trained survival model to PyTorch and ONNX formats.
    
    Args:
        model: Trained PyCox model (DeepSurv or DeepHit)
        model_params: Model parameters and hyperparameters
        output_dir: Directory to save exported models (default: "model_output")
        export_onnx: Whether to export to ONNX format (default: True)
        example_input: Example input for ONNX export (default: None)
        seed: Random seed (default: 42)
        
    Returns:
        Dictionary containing paths to exported model files
    """
    try:
        print(f"\n=== Exporting {model_params['model_type'].upper()} Model ===\n")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Extract model parameters
        model_type = model_params["model_type"]
        input_dim = model_params["input_dim"]
        output_dim = model_params["output_dim"]
        hidden_dims = model_params["hidden_dims"]
        
        # For DeepSurv (CoxPH) models, compute and store baseline hazards
        if model_type.lower() == "deepsurv":
            try:
                # Ensure model is on the correct device
                model.net = model.net.to(device)
                
                # Compute baseline hazards if not already computed
                if not hasattr(model, 'baseline_hazards'):
                    print("Computing baseline hazards for DeepSurv model...")
                    # Get training data from model if available
                    if hasattr(model, 'training_data'):
                        x_train, y_train = model.training_data
                        x_train = x_train.to(device)
                        model.compute_baseline_hazards(x_train, y_train)
                    else:
                        print("Warning: No training data available for computing baseline hazards.")
                
                # Store baseline hazards in model parameters
                if hasattr(model, 'baseline_hazards'):
                    model_params['baseline_hazards'] = model.baseline_hazards
                    model_params['baseline_times'] = model.baseline_times
                    print("Baseline hazards computed and stored in model parameters.")
            except Exception as e:
                print(f"Warning: Failed to compute baseline hazards: {e}")
        
        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = os.path.join(output_dir, f"{model_type}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Export model parameters
        params_path = os.path.join(model_dir, "model_params.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_params = {}
        for key, value in model_params.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                serializable_params[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                serializable_params[key] = int(value)
            else:
                serializable_params[key] = value
        
        with open(params_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
        
        print(f"Saved model parameters to {params_path}")
        
        # Export PyTorch model
        torch_path = os.path.join(model_dir, f"{model_type}_model.pt")
        
        # Get the network from the model
        net = model.net
        
        # Save the network state dict
        torch.save(net.state_dict(), torch_path)
        
        print(f"Saved PyTorch model to {torch_path}")
        
        # Export ONNX model if requested
        onnx_path = None
        if export_onnx:
            onnx_path = os.path.join(model_dir, f"{model_type}_model.onnx")
            
            # Create example input if not provided and move to device
            if example_input is None:
                example_input = torch.randn(1, input_dim).to(device)
            elif isinstance(example_input, np.ndarray):
                example_input = torch.tensor(example_input, dtype=torch.float32).to(device)
            
            # Set the model to evaluation mode and ensure it's on the right device
            net = net.to(device)
            net.eval()
            
            # Export to ONNX
            torch.onnx.export(
                net,                                # model being run
                example_input,                      # model input (or a tuple for multiple inputs)
                onnx_path,                          # where to save the model
                export_params=True,                 # store the trained parameter weights inside the model file
                opset_version=12,                   # the ONNX version to export the model to
                do_constant_folding=True,           # whether to execute constant folding for optimization
                input_names=['input'],              # the model's input names
                output_names=['output'],            # the model's output names
                dynamic_axes={
                    'input': {0: 'batch_size'},     # variable length axes
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"Saved ONNX model to {onnx_path}")
            
            # Verify ONNX model
            try:
                import onnx
                
                # Load ONNX model
                onnx_model = onnx.load(onnx_path)
                
                # Check that the model is well formed
                onnx.checker.check_model(onnx_model)
                
                print("ONNX model is valid")
                
                # Try to run the model with ONNX Runtime
                try:
                    import onnxruntime as ort
                    
                    # Create ONNX Runtime session
                    ort_session = ort.InferenceSession(onnx_path)
                    
                    # Run the model
                    ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
                    ort_outputs = ort_session.run(None, ort_inputs)
                    
                    print("ONNX Runtime inference successful")
                    
                    # Compare PyTorch and ONNX Runtime outputs
                    torch_output = net(example_input).detach().numpy()
                    ort_output = ort_outputs[0]
                    
                    # Check if outputs are close
                    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-03, atol=1e-05)
                    
                    print("PyTorch and ONNX Runtime outputs match")
                    
                except ImportError:
                    print("ONNX Runtime not available, skipping inference test")
                except Exception as e:
                    print(f"ONNX Runtime inference failed: {e}")
            
            except ImportError:
                print("ONNX not available, skipping validation")
            except Exception as e:
                print(f"ONNX validation failed: {e}")
        
        # Create a simple inference script
        inference_script_path = os.path.join(model_dir, "inference.py")
        
        with open(inference_script_path, 'w') as f:
            f.write("""
import os
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Union, Optional

# Define inference function
def load_model(model_dir: str):
    \"\"\"
    Load a trained {model_type} model from the specified directory.
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        Loaded model and model parameters
    \"\"\"
    # Load model parameters
    with open(os.path.join(model_dir, "model_params.json"), 'r') as f:
        model_params = json.load(f)
    
    # Create network
    from src.nn_architectures import create_network
    
    net = create_network(
        model_type=model_params["model_type"],
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        output_dim=model_params["output_dim"],
        dropout=model_params.get("dropout", 0.0)
    )
    
    # Load state dict
    net.load_state_dict(torch.load(os.path.join(model_dir, "{model_type}_model.pt")))
    
    # Set to evaluation mode
    net.eval()
    
    return net, model_params

def predict_survival(model, model_params: Dict, x: np.ndarray, 
                    time_points: Optional[List[float]] = None) -> Dict:
    \"\"\"
    Predict survival probabilities for the given input.
    
    Args:
        model: Loaded PyTorch model
        model_params: Model parameters
        x: Input features (numpy array)
        time_points: Time points for prediction (default: None, uses model's time grid)
        
    Returns:
        Dictionary containing survival predictions
    \"\"\"
    # Convert input to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Get model type
    model_type = model_params["model_type"]
    
    # Get time grid
    if time_points is None and "time_grid" in model_params:
        time_points = model_params["time_grid"]
    elif time_points is None:
        time_points = [365, 730, 1825]  # Default: 1, 2, 5 years
    
    # Make prediction
    with torch.no_grad():
        if model_type.lower() == "deepsurv":
            # For DeepSurv, output is the risk score
            risk_score = model(x_tensor).numpy()
            
            # Higher risk score means higher risk (lower survival)
            # Convert to survival probabilities using baseline hazards
            survival_probs = dict()
            
            # Check if baseline hazards are available in model parameters
            if "baseline_hazards" in model_params and "baseline_times" in model_params:
                baseline_hazards = np.array(model_params["baseline_hazards"])
                baseline_times = np.array(model_params["baseline_times"])
                
                for t in time_points:
                    # Find the closest time point in baseline_times
                    idx = np.searchsorted(baseline_times, t)
                    
                    if idx >= len(baseline_times):
                        # If t is beyond the last time point, use the last baseline hazard
                        cumulative_hazard = baseline_hazards[-1]
                    elif idx == 0:
                        # If t is before the first time point, use the first baseline hazard
                        cumulative_hazard = baseline_hazards[0]
                    else:
                        # Interpolate between the two closest time points
                        t1, t2 = baseline_times[idx-1], baseline_times[idx]
                        h1, h2 = baseline_hazards[idx-1], baseline_hazards[idx]
                        
                        # Linear interpolation
                        cumulative_hazard = h1 + (h2 - h1) * (t - t1) / (t2 - t1)
                    
                    # Calculate survival probability: S(t) = exp(-H0(t) * exp(risk_score))
                    surv_prob = np.exp(-cumulative_hazard * np.exp(risk_score))
                    survival_probs[f"survival_prob_at_{t}_days"] = surv_prob.tolist()
            else:
                # Fallback to simplified approach if baseline hazards are not available
                print("Warning: Baseline hazards not available. Using simplified approach.")
                for t in time_points:
                    # Simple exponential decay: S(t) = exp(-h0(t) * exp(risk_score))
                    # where h0(t) is the baseline hazard at time t
                    # For simplicity, we use h0(t) = t/1000
                    baseline_hazard = t / 1000
                    surv_prob = np.exp(-baseline_hazard * np.exp(risk_score))
                    survival_probs[f"survival_prob_at_{t}_days"] = surv_prob.tolist()
            
            return {
                "risk_scores": risk_score.tolist(),
                "survival_probabilities": survival_probs
            }
        
        else:  # DeepHit
            # For DeepHit, output is the PMF
            pmf = model(x_tensor).numpy()
            
            # Convert PMF to survival function
            surv = 1 - np.cumsum(pmf, axis=1)
            
            # Get time grid from model parameters
            time_grid = np.array(model_params["time_grid"])
            
            # Interpolate survival function at requested time points
            survival_probs = {}
            
            for t in time_points:
                # Find the closest time point in the grid
                idx = np.searchsorted(time_grid, t)
                
                if idx >= len(time_grid):
                    # If t is beyond the last time point, use the last survival probability
                    surv_prob = surv[:, -1]
                elif idx == 0:
                    # If t is before the first time point, use the first survival probability
                    surv_prob = surv[:, 0]
                else:
                    # Interpolate between the two closest time points
                    t1, t2 = time_grid[idx-1], time_grid[idx]
                    s1, s2 = surv[:, idx-1], surv[:, idx]
                    
                    # Linear interpolation
                    surv_prob = s1 + (s2 - s1) * (t - t1) / (t2 - t1)
                
                survival_probs[f"survival_prob_at_{t}_days"] = surv_prob.tolist()
            
            return {
                "pmf": pmf.tolist(),
                "survival_probabilities": survival_probs
            }

# Example usage
if __name__ == "__main__":
    # Path to model directory
    model_dir = "."  # Current directory
    
    # Load model
    model, model_params = load_model(model_dir)
    
    # Create example input (random data)
    example_input = np.random.randn(5, model_params["input_dim"])
    
    # Make prediction
    predictions = predict_survival(model, model_params, example_input)
    
    # Print predictions
    print("Predictions:")
    print(json.dumps(predictions, indent=4))
""")
        
        print(f"Created inference script at {inference_script_path}")
        
        # Create a README file
        readme_path = os.path.join(model_dir, "README.md")
        
        with open(readme_path, 'w') as f:
            f.write(f"""# {model_type.upper()} Model

This directory contains a trained {model_type.upper()} model for CKD risk prediction.

## Model Information

- Model Type: {model_type}
- Input Dimension: {input_dim}
- Output Dimension: {output_dim}
- Hidden Dimensions: {hidden_dims}
- Timestamp: {timestamp}

## Files

- `model_params.json`: Model parameters and hyperparameters
- `{model_type}_model.pt`: PyTorch model weights
- `{model_type}_model.onnx`: ONNX model (if available)
- `inference.py`: Example inference script
- `README.md`: This file

## Usage

To use the model for inference, you can use the provided `inference.py` script:

```python
import numpy as np
from inference import load_model, predict_survival

# Load model
model, model_params = load_model(".")

# Create input data
x = np.array([[...]])  # Input features

# Make prediction
predictions = predict_survival(model, model_params, x)
print(predictions)
```

## ONNX Inference

If you want to use the ONNX model for inference, you can use ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("{model_type}_model.onnx")

# Create input data
x = np.array([[...]])  # Input features

# Make prediction
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: x.astype(np.float32)})
print(output)
```
""")
        
        print(f"Created README at {readme_path}")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(params_path)
        mlflow.log_artifact(torch_path)
        mlflow.log_artifact(inference_script_path)
        mlflow.log_artifact(readme_path)
        
        if onnx_path is not None:
            mlflow.log_artifact(onnx_path)
        
        # Return paths to exported files
        export_paths = {
            "model_dir": model_dir,
            "params_path": params_path,
            "torch_path": torch_path,
            "inference_script_path": inference_script_path,
            "readme_path": readme_path
        }
        
        if onnx_path is not None:
            export_paths["onnx_path"] = onnx_path
        
        return export_paths
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        raise