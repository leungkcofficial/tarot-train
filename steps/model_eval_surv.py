"""
Model Evaluation Step for CKD Risk Prediction

This module contains the ZenML step for evaluating deep learning survival analysis models
(DeepSurv and DeepHit) and generating comprehensive evaluation reports.
"""

import os
import numpy as np
import pandas as pd
import torch
import mlflow
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Import PyCox evaluation
from pycox.evaluation import EvalSurv


@step
def model_eval_surv(
    model: Any,
    test_ds: Any,
    model_params: Dict[str, Any],
    output_dir: str = "model_output",
    time_horizons: Optional[List[int]] = None,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a trained survival model and generate comprehensive evaluation reports.
    
    Args:
        model: Trained PyCox model (DeepSurv or DeepHit)
        test_ds: Test dataset (PyCox dataset)
        model_params: Model parameters and hyperparameters
        output_dir: Directory to save evaluation outputs (default: "model_output")
        time_horizons: List of time horizons for evaluation (default: None)
        n_bootstrap: Number of bootstrap samples for confidence intervals (default: 100)
        seed: Random seed (default: 42)
        
    Returns:
        Dictionary containing evaluation metrics and paths to generated reports
    """
    try:
        print(f"\n=== Evaluating {model_params['model_type'].upper()} Model ===\n")
        
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
        
        # Set default time horizons if not provided
        if time_horizons is None:
            # Get max duration from test data
            max_duration = test_ds[1].max()
            # Create time horizons at 1, 2, and 5 years (365, 730, 1825 days)
            time_horizons = [365, 730, 1825]
            # Filter out horizons beyond max duration
            time_horizons = [t for t in time_horizons if t <= max_duration]
            print(f"Using default time horizons: {time_horizons} days")
        
        # Get test data
        x_test, durations_test, events_test = test_ds
        
        # Convert test data to torch tensors with float32 dtype and move to device
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        
        # For CoxPH models, we need to compute baseline hazards before predicting
        if model_type.lower() == "deepsurv" and not hasattr(model, 'baseline_hazards_'):
            print("Computing baseline hazards for CoxPH model...")
            # We need training data to compute baseline hazards
            # Since we don't have direct access to training data here,
            # we'll use the model's training_data if available
            if hasattr(model, 'training_data'):
                print("Using model's stored training data for baseline hazards computation")
                _ = model.compute_baseline_hazards()
            else:
                print("WARNING: Model doesn't have training_data attribute. Survival predictions may fail.")
                # We'll try to use test data as a fallback, though this is not ideal
                durations_test_tensor = torch.tensor(durations_test, dtype=torch.float32).to(device)
                events_test_tensor = torch.tensor(events_test, dtype=torch.float32).to(device)
                model.training_data = (x_test_tensor, (durations_test_tensor, events_test_tensor))
                _ = model.compute_baseline_hazards()
        
        # Get survival function predictions
        print("Generating survival predictions...")
        surv_df = model.predict_surv_df(x_test_tensor)
        
        # Create EvalSurv object
        ev = EvalSurv(
            surv_df,
            durations_test,
            events_test,
            censor_surv='km'
        )
        
        # Calculate concordance index
        print("Calculating concordance index...")
        c_index = ev.concordance_td()
        print(f"Test concordance index: {c_index:.6f}")
        
        # Calculate time-dependent AUC
        print("Calculating time-dependent AUC...")
        auc_scores = {}
        for t in time_horizons:
            auc_t = ev.integrated_brier_score(times=np.array([t]))
            auc_scores[f"auc_at_{t}_days"] = auc_t
            print(f"AUC at {t} days: {auc_t:.6f}")
        
        # Calculate integrated Brier score
        print("Calculating integrated Brier score...")
        ibs = ev.integrated_brier_score(times=np.array(time_horizons))
        print(f"Integrated Brier score: {ibs:.6f}")
        
        # Calculate bootstrap confidence intervals for concordance index
        print(f"Calculating bootstrap confidence intervals (n={n_bootstrap})...")
        c_index_samples = []
        
        # Create bootstrap samples
        n_samples = len(durations_test)
        for i in range(n_bootstrap):
            if i % 20 == 0:
                print(f"Bootstrap sample {i+1}/{n_bootstrap}")
            
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Create bootstrap sample
            x_boot = x_test[indices]
            durations_boot = durations_test[indices]
            events_boot = events_test[indices]
            
            # Get predictions for bootstrap sample
            x_boot_tensor = torch.tensor(x_boot, dtype=torch.float32).to(device)
            surv_boot = model.predict_surv_df(x_boot_tensor)
            
            # Create EvalSurv object
            ev_boot = EvalSurv(
                surv_boot,
                durations_boot,
                events_boot,
                censor_surv='km'
            )
            
            # Calculate concordance index
            c_index_boot = ev_boot.concordance_td()
            c_index_samples.append(c_index_boot)
        
        # Calculate confidence intervals
        c_index_samples = np.array(c_index_samples)
        c_index_lower = np.percentile(c_index_samples, 2.5)
        c_index_upper = np.percentile(c_index_samples, 97.5)
        print(f"C-index 95% CI: [{c_index_lower:.6f}, {c_index_upper:.6f}]")
        
        # Create survival curves for selected patients
        print("Generating survival curve plots...")
        
        # Select a few patients with different risk profiles
        n_patients = min(5, len(x_test))
        
        # Get predicted survival curves
        surv_curves = surv_df.iloc[:, :n_patients]
        
        # Plot survival curves
        plt.figure(figsize=(10, 6))
        for i in range(n_patients):
            plt.step(surv_df.index, surv_curves.iloc[:, i], where="post", 
                    label=f"Patient {i+1} (Duration={durations_test[i]:.0f}, Event={events_test[i]})")
        
        plt.xlabel("Time (days)")
        plt.ylabel("Survival Probability")
        plt.title(f"{model_type.upper()} Predicted Survival Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        surv_curves_path = os.path.join(output_dir, f"{model_type}_survival_curves.png")
        plt.savefig(surv_curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved survival curves plot to {surv_curves_path}")
        
        # Plot time-dependent AUC
        print("Generating time-dependent AUC plot...")
        time_grid = np.linspace(np.min(time_horizons) * 0.8, np.max(time_horizons) * 1.1, 100)
        auc_grid = ev.time_dependent_auc(time_grid)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, auc_grid)
        plt.xlabel("Time (days)")
        plt.ylabel("Time-dependent AUC")
        plt.title(f"{model_type.upper()} Time-dependent AUC")
        plt.grid(True, alpha=0.3)
        
        # Add reference lines for time horizons
        for t in time_horizons:
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
            plt.text(t, 0.5, f"{t} days", rotation=90, verticalalignment='center')
        
        # Save plot
        auc_path = os.path.join(output_dir, f"{model_type}_time_dependent_auc.png")
        plt.savefig(auc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time-dependent AUC plot to {auc_path}")
        
        # Generate calibration plots
        print("Generating calibration plots...")
        
        for t in time_horizons:
            plt.figure(figsize=(8, 8))
            ev.plot_calibration(t, ax=plt.gca())
            plt.title(f"{model_type.upper()} Calibration at {t} days")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            calib_path = os.path.join(output_dir, f"{model_type}_calibration_{t}_days.png")
            plt.savefig(calib_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved calibration plot for {t} days to {calib_path}")
            
            # Log to MLflow
            mlflow.log_artifact(calib_path)
        
        # Create summary DataFrame
        summary_metrics = {
            "model_type": model_type,
            "c_index": c_index,
            "c_index_lower_ci": c_index_lower,
            "c_index_upper_ci": c_index_upper,
            "integrated_brier_score": ibs,
        }
        
        # Add time-dependent metrics
        for t in time_horizons:
            auc_t = ev.time_dependent_auc(np.array([t]))[0]
            brier_t = ev.brier_score(np.array([t]))[0]
            
            summary_metrics[f"auc_{t}_days"] = auc_t
            summary_metrics[f"brier_{t}_days"] = brier_t
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([summary_metrics])
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{model_type}_evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved evaluation summary to {summary_path}")
        
        # Log metrics to MLflow
        mlflow.log_metric("test_c_index", float(c_index))
        mlflow.log_metric("test_c_index_lower_ci", float(c_index_lower))
        mlflow.log_metric("test_c_index_upper_ci", float(c_index_upper))
        mlflow.log_metric("test_integrated_brier_score", float(ibs))
        
        for t in time_horizons:
            mlflow.log_metric(f"test_auc_{t}_days", float(summary_metrics[f"auc_{t}_days"]))
            mlflow.log_metric(f"test_brier_{t}_days", float(summary_metrics[f"brier_{t}_days"]))
        
        # Log artifacts to MLflow
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(surv_curves_path)
        mlflow.log_artifact(auc_path)
        
        # Return evaluation results
        evaluation_results = {
            "metrics": summary_metrics,
            "artifacts": {
                "summary_path": summary_path,
                "survival_curves_path": surv_curves_path,
                "auc_path": auc_path,
                "calibration_paths": [
                    os.path.join(output_dir, f"{model_type}_calibration_{t}_days.png")
                    for t in time_horizons
                ]
            }
        }
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        raise