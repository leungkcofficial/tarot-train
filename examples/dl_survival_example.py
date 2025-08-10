"""
Example Script for Deep Learning Survival Models

This script demonstrates how to use the deep learning survival models (DeepSurv and DeepHit)
outside of the ZenML pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import neural network architectures
from src.nn_architectures import create_network

# Import PyCox models
from pycox.models import CoxPH, DeepHit
from pycox.evaluation import EvalSurv
from pycox.datasets import metabric

# Import utility functions
from src.survival_utils import plot_survival_curves, calculate_c_index, calculate_brier_score


def load_example_data():
    """
    Load example survival data.
    
    Returns:
        Tuple containing:
        - X: Feature matrix
        - y: Target (time, event)
        - feature_names: List of feature names
    """
    # Load example data from PyCox (METABRIC dataset)
    print("Loading METABRIC dataset...")
    df = metabric.read_df()
    
    # Extract features and target
    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']
    
    # Standardize features
    standardize = StandardScaler().fit(df[cols_standardize])
    df_std = pd.DataFrame(standardize.transform(df[cols_standardize]), 
                         columns=cols_standardize, index=df.index)
    df_leave = df[cols_leave]
    
    # Combine standardized and non-standardized features
    X = pd.concat([df_std, df_leave], axis=1)
    
    # Get target
    y = df[['duration', 'event']]
    
    # Get feature names
    feature_names = list(X.columns)
    
    return X, y, feature_names


def prepare_data(X, y, val_size=0.2, test_size=0.2, seed=42):
    """
    Prepare data for training and evaluation.
    
    Args:
        X: Feature matrix
        y: Target (time, event)
        val_size: Validation set size (default: 0.2)
        test_size: Test set size (default: 0.2)
        seed: Random seed (default: 42)
        
    Returns:
        Tuple containing:
        - train_ds: Training dataset (PyCox format)
        - val_ds: Validation dataset (PyCox format)
        - test_ds: Test dataset (PyCox format)
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Split data into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=seed
    )
    
    # Convert to numpy arrays
    X_train = X_train.values.astype('float32')
    X_val = X_val.values.astype('float32')
    X_test = X_test.values.astype('float32')
    
    # Get durations and events
    durations_train = y_train['duration'].values.astype('float32')
    events_train = y_train['event'].values.astype('int32')
    
    durations_val = y_val['duration'].values.astype('float32')
    events_val = y_val['event'].values.astype('int32')
    
    durations_test = y_test['duration'].values.astype('float32')
    events_test = y_test['event'].values.astype('int32')
    
    # Create PyCox datasets
    train_ds = (X_train, durations_train, events_train)
    val_ds = (X_val, durations_val, events_val)
    test_ds = (X_test, durations_test, events_test)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return train_ds, val_ds, test_ds


def train_deepsurv(train_ds, val_ds, input_dim, hidden_dims=[32, 32], 
                  dropout=0.1, batch_size=64, epochs=100, patience=10, lr=0.01):
    """
    Train a DeepSurv model.
    
    Args:
        train_ds: Training dataset (PyCox format)
        val_ds: Validation dataset (PyCox format)
        input_dim: Input dimension
        hidden_dims: Hidden dimensions (default: [32, 32])
        dropout: Dropout rate (default: 0.1)
        batch_size: Batch size (default: 64)
        epochs: Number of epochs (default: 100)
        patience: Early stopping patience (default: 10)
        lr: Learning rate (default: 0.01)
        
    Returns:
        Trained DeepSurv model
    """
    print("\nTraining DeepSurv model...")
    
    # Create network
    net = create_network(
        model_type="deepsurv",
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout=dropout
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Create model
    model = CoxPH(net, optimizer=optimizer)
    
    # Create callbacks for early stopping
    callbacks = [
        {'event': 'es', 'patience': patience}
    ]
    
    # Train model
    log = model.fit(
        train_ds[0], train_ds[1], train_ds[2],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        val_data=val_ds,
        verbose=True
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(log.epoch, log.train_loss, label='Train Loss')
    plt.plot(log.epoch, log.val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DeepSurv Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('deepsurv_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model


def train_deephit(train_ds, val_ds, input_dim, hidden_dims=[32, 32], 
                 dropout=0.1, batch_size=64, epochs=100, patience=10, lr=0.01,
                 alpha=0.2, sigma=0.1, time_grid=None):
    """
    Train a DeepHit model.
    
    Args:
        train_ds: Training dataset (PyCox format)
        val_ds: Validation dataset (PyCox format)
        input_dim: Input dimension
        hidden_dims: Hidden dimensions (default: [32, 32])
        dropout: Dropout rate (default: 0.1)
        batch_size: Batch size (default: 64)
        epochs: Number of epochs (default: 100)
        patience: Early stopping patience (default: 10)
        lr: Learning rate (default: 0.01)
        alpha: Alpha parameter (default: 0.2)
        sigma: Sigma parameter (default: 0.1)
        time_grid: Time grid (default: None)
        
    Returns:
        Trained DeepHit model
    """
    print("\nTraining DeepHit model...")
    
    # Set default time grid if not provided
    if time_grid is None:
        # Get max duration from training data
        max_duration = train_ds[1].max()
        # Create time grid with 10 points
        time_grid = np.linspace(0, max_duration, 10)
        print(f"Using default time grid with 10 points up to {max_duration:.1f}")
    
    # Create network
    net = create_network(
        model_type="deephit",
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=len(time_grid),
        dropout=dropout
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Create model
    model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
    
    # Create callbacks for early stopping
    callbacks = [
        {'event': 'es', 'patience': patience}
    ]
    
    # Train model
    log = model.fit(
        train_ds[0], train_ds[1], train_ds[2],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        val_data=val_ds,
        verbose=True
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(log.epoch, log.train_loss, label='Train Loss')
    plt.plot(log.epoch, log.val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DeepHit Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('deephit_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model


def evaluate_model(model, test_ds, model_type, time_horizons=None):
    """
    Evaluate a trained survival model.
    
    Args:
        model: Trained PyCox model
        test_ds: Test dataset (PyCox format)
        model_type: Type of model ("deepsurv" or "deephit")
        time_horizons: Time horizons for evaluation (default: None)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating {model_type.upper()} model...")
    
    # Set default time horizons if not provided
    if time_horizons is None:
        # Get max duration from test data
        max_duration = test_ds[1].max()
        # Create time horizons at 25%, 50%, and 75% of max duration
        time_horizons = [
            max_duration * 0.25,
            max_duration * 0.5,
            max_duration * 0.75
        ]
        print(f"Using default time horizons: {[int(t) for t in time_horizons]}")
    
    # Get test data
    x_test, durations_test, events_test = test_ds
    
    # Get survival function predictions
    surv_df = model.predict_surv_df(x_test)
    
    # Create EvalSurv object
    ev = EvalSurv(
        surv_df,
        durations_test,
        events_test,
        censor_surv='km'
    )
    
    # Calculate concordance index
    c_index = ev.concordance_td()
    print(f"Concordance index: {c_index:.6f}")
    
    # Calculate integrated Brier score
    ibs = ev.integrated_brier_score(times=np.array(time_horizons))
    print(f"Integrated Brier score: {ibs:.6f}")
    
    # Calculate time-dependent metrics
    metrics = {"c_index": c_index, "integrated_brier_score": ibs}
    
    for t in time_horizons:
        auc_t = ev.time_dependent_auc(np.array([t]))[0]
        brier_t = ev.brier_score(np.array([t]))[0]
        
        metrics[f"auc_at_{int(t)}"] = auc_t
        metrics[f"brier_at_{int(t)}"] = brier_t
        
        print(f"AUC at {int(t)}: {auc_t:.6f}")
        print(f"Brier score at {int(t)}: {brier_t:.6f}")
    
    # Plot survival curves for a few patients
    n_patients = min(5, len(x_test))
    
    plt.figure(figsize=(10, 6))
    for i in range(n_patients):
        plt.step(surv_df.index, surv_df.iloc[:, i], where="post", 
                label=f"Patient {i+1} (Duration={durations_test[i]:.0f}, Event={events_test[i]})")
    
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title(f"{model_type.upper()} Predicted Survival Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{model_type}_survival_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot calibration at the middle time horizon
    middle_t = time_horizons[len(time_horizons) // 2]
    
    plt.figure(figsize=(8, 8))
    ev.plot_calibration(middle_t, ax=plt.gca())
    plt.title(f"{model_type.upper()} Calibration at {int(middle_t)}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_type}_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def compare_models(deepsurv_metrics, deephit_metrics):
    """
    Compare DeepSurv and DeepHit models.
    
    Args:
        deepsurv_metrics: DeepSurv evaluation metrics
        deephit_metrics: DeepHit evaluation metrics
    """
    print("\nComparing DeepSurv and DeepHit models...")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'DeepSurv': deepsurv_metrics,
        'DeepHit': deephit_metrics
    })
    
    # Print comparison
    print(comparison)
    
    # Plot comparison
    metrics_to_plot = ['c_index', 'integrated_brier_score']
    metrics_to_plot.extend([m for m in deepsurv_metrics.keys() if m.startswith('auc_at_')])
    
    plt.figure(figsize=(12, 8))
    comparison.loc[metrics_to_plot].plot(kind='bar')
    plt.title('Model Comparison')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plots saved to 'model_comparison.png'")


def main():
    """Main function."""
    # Load example data
    X, y, feature_names = load_example_data()
    
    # Prepare data
    train_ds, val_ds, test_ds = prepare_data(X, y)
    
    # Get input dimension
    input_dim = train_ds[0].shape[1]
    
    # Train DeepSurv model
    deepsurv_model = train_deepsurv(
        train_ds=train_ds,
        val_ds=val_ds,
        input_dim=input_dim,
        hidden_dims=[64, 32],
        dropout=0.1,
        batch_size=64,
        epochs=100,
        patience=10,
        lr=0.01
    )
    
    # Evaluate DeepSurv model
    deepsurv_metrics = evaluate_model(
        model=deepsurv_model,
        test_ds=test_ds,
        model_type="deepsurv"
    )
    
    # Train DeepHit model
    deephit_model = train_deephit(
        train_ds=train_ds,
        val_ds=val_ds,
        input_dim=input_dim,
        hidden_dims=[64, 32],
        dropout=0.1,
        batch_size=64,
        epochs=100,
        patience=10,
        lr=0.01,
        alpha=0.2,
        sigma=0.1
    )
    
    # Evaluate DeepHit model
    deephit_metrics = evaluate_model(
        model=deephit_model,
        test_ds=test_ds,
        model_type="deephit"
    )
    
    # Compare models
    compare_models(deepsurv_metrics, deephit_metrics)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()