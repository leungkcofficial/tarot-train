"""
Model Comparison Step for CKD Risk Prediction

This module contains the ZenML step for comparing the performance of different models:
- KFRE (Kidney Failure Risk Equation)
- DeepSurv (Deep Learning Survival Analysis)
- DeepHit (Deep Learning Survival Analysis with Competing Risks)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import torch
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import PyCox evaluation
from pycox.evaluation import EvalSurv


@step
def dl_model_comparison(
    test_df: pd.DataFrame,
    test_ds: Any,
    model_type: str,
    deployed_model_details: Dict[str, Any],
    kfre_predictions: Optional[pd.DataFrame] = None,
    time_horizons: Optional[List[int]] = None,
    output_dir: str = "model_output",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compare the performance of different survival models.
    
    Args:
        test_df: Test DataFrame with features and outcomes
        test_ds: Test dataset in PyCox format
        model_type: Type of model ('deepsurv' or 'deephit')
        deployed_model_details: Dictionary containing deployed model details
        kfre_predictions: KFRE predictions DataFrame (optional)
        time_horizons: List of time horizons for evaluation (default: None)
        output_dir: Directory to save comparison outputs (default: "model_output")
        seed: Random seed (default: 42)
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        print("\n=== Comparing Model Performance ===\n")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
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
        
        # Initialize results dictionary
        results = {
            "models": [],
            "c_index": [],
            "integrated_brier_score": []
        }
        
        # Add time-dependent metrics
        for t in time_horizons:
            results[f"auc_{t}_days"] = []
            results[f"brier_{t}_days"] = []
        
        # Evaluate KFRE if provided
        if kfre_predictions is not None:
            print("\nEvaluating KFRE model...")
            
            # Check if KFRE predictions contain survival probabilities
            if any(f"survival_prob_{t}_days" in kfre_predictions.columns for t in time_horizons):
                # Create survival function DataFrame
                kfre_surv = pd.DataFrame(index=np.sort(np.unique(durations_test)))
                
                # Fill with survival probabilities
                for i in range(len(x_test)):
                    # Get KFRE survival probabilities for this patient
                    surv_probs = {}
                    for t in time_horizons:
                        col_name = f"survival_prob_{t}_days"
                        if col_name in kfre_predictions.columns:
                            surv_probs[t] = kfre_predictions.iloc[i][col_name]
                    
                    # Interpolate survival curve
                    surv_curve = np.ones(len(kfre_surv))
                    for j, t in enumerate(kfre_surv.index):
                        # Find closest time horizons
                        lower_horizons = [h for h in time_horizons if h <= t]
                        upper_horizons = [h for h in time_horizons if h > t]
                        
                        if not lower_horizons and not upper_horizons:
                            # No time horizons, use 1.0
                            surv_curve[j] = 1.0
                        elif not lower_horizons:
                            # Only upper horizons, use the lowest
                            min_upper = min(upper_horizons)
                            surv_curve[j] = surv_probs[min_upper]
                        elif not upper_horizons:
                            # Only lower horizons, use the highest
                            max_lower = max(lower_horizons)
                            surv_curve[j] = surv_probs[max_lower]
                        else:
                            # Both lower and upper horizons, interpolate
                            max_lower = max(lower_horizons)
                            min_upper = min(upper_horizons)
                            
                            # Linear interpolation
                            alpha = (t - max_lower) / (min_upper - max_lower)
                            surv_curve[j] = surv_probs[max_lower] + alpha * (surv_probs[min_upper] - surv_probs[max_lower])
                    
                    kfre_surv[i] = surv_curve
                
                # Create EvalSurv object
                ev_kfre = EvalSurv(
                    kfre_surv,
                    durations_test,
                    events_test,
                    censor_surv='km'
                )
                
                # Calculate concordance index
                c_index_kfre = ev_kfre.concordance_td()
                print(f"KFRE concordance index: {c_index_kfre:.6f}")
                
                # Calculate integrated Brier score
                ibs_kfre = ev_kfre.integrated_brier_score(times=np.array(time_horizons))
                print(f"KFRE integrated Brier score: {ibs_kfre:.6f}")
                
                # Calculate time-dependent metrics
                metrics_kfre = {"c_index": c_index_kfre, "integrated_brier_score": ibs_kfre}
                
                for t in time_horizons:
                    auc_t = ev_kfre.time_dependent_auc(np.array([t]))[0]
                    brier_t = ev_kfre.brier_score(np.array([t]))[0]
                    
                    metrics_kfre[f"auc_{t}_days"] = auc_t
                    metrics_kfre[f"brier_{t}_days"] = brier_t
                    
                    print(f"KFRE AUC at {t} days: {auc_t:.6f}")
                    print(f"KFRE Brier score at {t} days: {brier_t:.6f}")
                
                # Add to results
                results["models"].append("KFRE")
                results["c_index"].append(c_index_kfre)
                results["integrated_brier_score"].append(ibs_kfre)
                
                for t in time_horizons:
                    results[f"auc_{t}_days"].append(metrics_kfre[f"auc_{t}_days"])
                    results[f"brier_{t}_days"].append(metrics_kfre[f"brier_{t}_days"])
            else:
                print("KFRE predictions do not contain survival probabilities, skipping evaluation")
        
        # Evaluate deployed model
        if deployed_model_details is not None:
            print(f"\nEvaluating deployed {model_type.upper()} model...")
            
            # Load model details
            model_weights_path = deployed_model_details.get('deployed_model_weights_path')
            input_dim = deployed_model_details.get('input_dim')
            output_dim = deployed_model_details.get('output_dim')
            hidden_dims = deployed_model_details.get('hidden_dims')
            hyperparameters = deployed_model_details.get('hyperparameters', {})
            
            # Create network
            from src.nn_architectures import create_network
            net = create_network(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=hyperparameters.get('dropout', 0.2)
            )
            
            # Load weights
            net.load_state_dict(torch.load(model_weights_path))
            
            # Create optimizer (not needed for evaluation, but required for model creation)
            optimizer_name = hyperparameters.get('optimizer', 'Adam')
            lr = hyperparameters.get('learning_rate', 0.001)
            
            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            else:
                optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
            
            # Create model
            from pycox.models import CoxPH, DeepHit
            if model_type.lower() == "deepsurv":
                model = CoxPH(net, optimizer=optimizer)
            else:
                alpha = hyperparameters.get('alpha', 0.2)
                sigma = hyperparameters.get('sigma', 0.1)
                time_grid = deployed_model_details.get('time_grid')
                model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
            
            # Move model to device
            model.net = model.net.to(device)
            
            # Convert input data to torch tensor with float32 dtype and move to device
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            
            # For CoxPH models, we need to compute baseline hazards before predicting
            if model_type.lower() == "deepsurv" and not hasattr(model, 'baseline_hazards_'):
                print("Computing baseline hazards for CoxPH model...")
                # We need training data to compute baseline hazards
                # Since we don't have direct access to training data here,
                # we'll use test data as a fallback
                durations_test_tensor = torch.tensor(durations_test, dtype=torch.float32).to(device)
                events_test_tensor = torch.tensor(events_test, dtype=torch.float32).to(device)
                model.training_data = (x_test_tensor, (durations_test_tensor, events_test_tensor))
                _ = model.compute_baseline_hazards()
            
            # Get survival function predictions
            model_surv = model.predict_surv_df(x_test_tensor)
            
            # Create EvalSurv object
            ev_model = EvalSurv(
                model_surv,
                durations_test,
                events_test,
                censor_surv='km'
            )
            
            # Calculate concordance index
            c_index_model = ev_model.concordance_td()
            print(f"{model_type.upper()} concordance index: {c_index_model:.6f}")
            
            # Calculate integrated Brier score
            ibs_model = ev_model.integrated_brier_score(times=np.array(time_horizons))
            print(f"{model_type.upper()} integrated Brier score: {ibs_model:.6f}")
            
            # Calculate time-dependent metrics
            metrics_model = {"c_index": c_index_model, "integrated_brier_score": ibs_model}
            
            for t in time_horizons:
                auc_t = ev_model.time_dependent_auc(np.array([t]))[0]
                brier_t = ev_model.brier_score(np.array([t]))[0]
                
                metrics_model[f"auc_{t}_days"] = auc_t
                metrics_model[f"brier_{t}_days"] = brier_t
                
                print(f"{model_type.upper()} AUC at {t} days: {auc_t:.6f}")
                print(f"{model_type.upper()} Brier score at {t} days: {brier_t:.6f}")
            
            # Add to results
            results["models"].append(model_type.upper())
            results["c_index"].append(c_index_model)
            results["integrated_brier_score"].append(ibs_model)
            
            for t in time_horizons:
                results[f"auc_{t}_days"].append(metrics_model[f"auc_{t}_days"])
                results[f"brier_{t}_days"].append(metrics_model[f"brier_{t}_days"])
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save comparison DataFrame
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved model comparison to {comparison_path}")
        
        # Create comparison plots
        print("\nGenerating comparison plots...")
        
        # C-index comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="models", y="c_index", data=comparison_df)
        plt.title("Concordance Index Comparison")
        plt.ylabel("C-index")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        c_index_path = os.path.join(output_dir, "c_index_comparison.png")
        plt.savefig(c_index_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved C-index comparison plot to {c_index_path}")
        
        # Integrated Brier score comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="models", y="integrated_brier_score", data=comparison_df)
        plt.title("Integrated Brier Score Comparison")
        plt.ylabel("Integrated Brier Score")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        ibs_path = os.path.join(output_dir, "ibs_comparison.png")
        plt.savefig(ibs_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Integrated Brier Score comparison plot to {ibs_path}")
        
        # Time-dependent AUC comparison
        for t in time_horizons:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="models", y=f"auc_{t}_days", data=comparison_df)
            plt.title(f"AUC at {t} Days Comparison")
            plt.ylabel(f"AUC at {t} Days")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            auc_path = os.path.join(output_dir, f"auc_{t}_days_comparison.png")
            plt.savefig(auc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved AUC at {t} days comparison plot to {auc_path}")
        
        # Time-dependent Brier score comparison
        for t in time_horizons:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="models", y=f"brier_{t}_days", data=comparison_df)
            plt.title(f"Brier Score at {t} Days Comparison")
            plt.ylabel(f"Brier Score at {t} Days")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            brier_path = os.path.join(output_dir, f"brier_{t}_days_comparison.png")
            plt.savefig(brier_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Brier Score at {t} days comparison plot to {brier_path}")
        
        # Create radar plot for overall comparison
        if len(results["models"]) > 1:
            print("\nGenerating radar plot for overall comparison...")
            
            # Prepare data for radar plot
            metrics = ["c_index"]
            for t in time_horizons:
                metrics.append(f"auc_{t}_days")
            
            # Normalize metrics
            radar_data = {}
            for metric in metrics:
                values = comparison_df[metric].values
                min_val = min(values)
                max_val = max(values)
                
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.ones_like(values)
                
                radar_data[metric] = normalized
            
            # Create radar plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Set number of angles based on number of metrics
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Set labels
            labels = [metric.replace("_", " ").title() for metric in metrics]
            plt.xticks(angles[:-1], labels, size=12)
            
            # Plot each model
            for i, model in enumerate(results["models"]):
                values = [radar_data[metric][i] for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Save plot
            radar_path = os.path.join(output_dir, "model_comparison_radar.png")
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved radar plot to {radar_path}")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(comparison_path)
        mlflow.log_artifact(c_index_path)
        mlflow.log_artifact(ibs_path)
        
        for t in time_horizons:
            mlflow.log_artifact(os.path.join(output_dir, f"auc_{t}_days_comparison.png"))
            mlflow.log_artifact(os.path.join(output_dir, f"brier_{t}_days_comparison.png"))
        
        if len(results["models"]) > 1 and "radar_path" in locals():
            mlflow.log_artifact(radar_path)
        
        # Return comparison results
        comparison_results = {
            "comparison_df": comparison_df,
            "artifacts": {
                "comparison_path": comparison_path,
                "c_index_path": c_index_path,
                "ibs_path": ibs_path,
                "auc_paths": [os.path.join(output_dir, f"auc_{t}_days_comparison.png") for t in time_horizons],
                "brier_paths": [os.path.join(output_dir, f"brier_{t}_days_comparison.png") for t in time_horizons]
            }
        }
        
        if len(results["models"]) > 1 and "radar_path" in locals():
            comparison_results["artifacts"]["radar_path"] = radar_path
        
        return comparison_results
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        import traceback
        traceback.print_exc()
        raise