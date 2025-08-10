"""
Model Development Module for CKD Risk Prediction

This module provides PyTorch-based model development utilities for the CKD risk prediction task.
It includes dataset classes, model wrappers, and training/evaluation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import json
import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
# Import pycox for survival analysis
try:
    from pycox.models import CoxPH, DeepHit
    PYCOX_AVAILABLE = True
except ImportError:
    print("Warning: pycox library not available. Survival analysis models will not be available.")
    PYCOX_AVAILABLE = False
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


from src.net_weaver import create_network, BaseNetwork, get_available_networks


class TabularDataset(Dataset):
    """PyTorch dataset for tabular data."""
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        cat_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
        normalize: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame or numpy array containing the data
            target_col: Name of the target column (if data is a DataFrame)
            cat_cols: List of categorical column names (if data is a DataFrame)
            cont_cols: List of continuous column names (if data is a DataFrame)
            normalize: Whether to normalize continuous features (default: True)
        """
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
        if isinstance(data, pd.DataFrame):
            # Extract target if provided
            if target_col is not None:
                self.y = data[target_col].values.astype(np.float32)
                self.X = data.drop(target_col, axis=1)
            else:
                self.y = np.zeros(len(data))  # Dummy target
                self.X = data.copy()
            
            # Get column names
            self.feature_names = self.X.columns.tolist()
            
            # Convert to numpy array
            self.X = self.X.values.astype(np.float32)
        else:
            # Assume data is a numpy array
            self.X = data.astype(np.float32)
            self.y = np.zeros(len(data))  # Dummy target
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        
        # Normalize continuous features if specified
        if normalize and self.X.shape[0] > 0:
            self.X = self.scaler.fit_transform(self.X)
        
        self.n_features = self.X.shape[1]
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, target)
        """
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
    def get_scaler(self) -> Optional[StandardScaler]:
        """Get the feature scaler."""
        return self.scaler
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names."""
        return self.feature_names


class CKDModel:
    """PyTorch model for CKD risk prediction."""
    
    def __init__(
        self,
        network_type: str = "mlp",
        input_dim: int = None,
        output_dim: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = None,
        **network_params
    ):
        """
        Initialize the model.
        
        Args:
            network_type: Type of network to use ("mlp", "resnet", "tabnet")
            input_dim: Number of input features
            output_dim: Number of output classes (default: 1 for binary classification)
            learning_rate: Learning rate for the optimizer (default: 0.001)
            weight_decay: Weight decay for the optimizer (default: 0.0001)
            device: Device to use for training ("cpu" or "cuda")
            **network_params: Additional parameters for the network
        """
        self.network_type = network_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.network_params = network_params
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create network if input_dim is provided
        self.network = None
        if input_dim is not None:
            self.network = create_network(
                network_type=network_type,
                input_dim=input_dim,
                output_dim=output_dim,
                **network_params
            ).to(self.device)
            
            # Create optimizer
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Create loss function
            self.criterion = nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    
    def fit(
        self,
        train_dataset: TabularDataset,
        val_dataset: Optional[TabularDataset] = None,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size (default: 64)
            epochs: Number of epochs (default: 100)
            patience: Patience for early stopping (default: 10)
            verbose: Whether to print progress (default: True)
            callbacks: List of callback functions to call after each epoch
            
        Returns:
            Dictionary containing training history
        """
        # Initialize network if not already initialized
        if self.network is None:
            self.input_dim = train_dataset.n_features
            self.network = create_network(
                network_type=self.network_type,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                **self.network_params
            ).to(self.device)
            
            # Create optimizer
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Create loss function
            self.criterion = nn.BCELoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # Initialize training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
        
        # Initialize early stopping variables
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            
            if verbose:
                train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Reshape target for binary classification
                if self.output_dim == 1:
                    target = target.view(-1, 1)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.network(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, batch_size)
                
                # Update history
                history["val_loss"].append(val_metrics["loss"])
                history["val_auc"].append(val_metrics["auc"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])
                
                # Check for early stopping
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_epoch = epoch
                    best_state_dict = self.network.state_dict().copy()
                
                # Print progress
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f} - "
                          f"Val AUC: {val_metrics['auc']:.4f}")
                
                # Check for early stopping
                if epoch - best_epoch >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # Print progress
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # Call callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, history)
        
        # Restore best model
        if val_dataset is not None and best_state_dict is not None:
            self.network.load_state_dict(best_state_dict)
        
        return history
    
    def evaluate(
        self,
        dataset: TabularDataset,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size (default: 64)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Set model to evaluation mode
        self.network.eval()
        
        # Initialize variables
        total_loss = 0.0
        all_targets = []
        all_outputs = []
        
        # Evaluation loop
        with torch.no_grad():
            for data, target in data_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Reshape target for binary classification
                if self.output_dim == 1:
                    target = target.view(-1, 1)
                
                # Forward pass
                output = self.network(data)
                loss = self.criterion(output, target)
                
                # Update statistics
                total_loss += loss.item()
                
                # Store targets and outputs for metric calculation
                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        
        # Concatenate targets and outputs
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        # Reshape for binary classification
        if self.output_dim == 1:
            all_targets = all_targets.ravel()
            all_outputs = all_outputs.ravel()
        
        # Calculate metrics
        metrics = {
            "loss": avg_loss,
            "auc": roc_auc_score(all_targets, all_outputs),
            "accuracy": accuracy_score(all_targets, all_outputs > 0.5),
            "precision": precision_score(all_targets, all_outputs > 0.5),
            "recall": recall_score(all_targets, all_outputs > 0.5),
            "f1": f1_score(all_targets, all_outputs > 0.5)
        }
        
        return metrics
    
    def predict(
        self,
        dataset: TabularDataset,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Make predictions on a dataset.
        
        Args:
            dataset: Dataset to predict on
            batch_size: Batch size (default: 64)
            
        Returns:
            Numpy array of predictions
        """
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Set model to evaluation mode
        self.network.eval()
        
        # Initialize variables
        all_outputs = []
        
        # Prediction loop
        with torch.no_grad():
            for data, _ in data_loader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                output = self.network(data)
                
                # Store outputs
                all_outputs.append(output.cpu().numpy())
        
        # Concatenate outputs
        all_outputs = np.concatenate(all_outputs)
        
        # Reshape for binary classification
        if self.output_dim == 1:
            all_outputs = all_outputs.ravel()
        
        return all_outputs
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            "network_type": self.network_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "network_params": self.network_params,
            "state_dict": self.network.state_dict() if self.network is not None else None
        }
        
        # Save to file
        torch.save(model_state, path)
    
    @classmethod
    def load(cls, path: str, device: str = None) -> "CKDModel":
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to ("cpu" or "cuda")
            
        Returns:
            Loaded model
        """
        # Load model state
        model_state = torch.load(path, map_location=torch.device("cpu"))
        
        # Create model
        model = cls(
            network_type=model_state["network_type"],
            input_dim=model_state["input_dim"],
            output_dim=model_state["output_dim"],
            learning_rate=model_state["learning_rate"],
            weight_decay=model_state["weight_decay"],
            device=device,
            **model_state["network_params"]
        )
        
        # Load state dict
        if model_state["state_dict"] is not None:
            model.network.load_state_dict(model_state["state_dict"])
        
        return model
    
    def log_to_mlflow(self) -> None:
        """Log model parameters and network architecture to MLflow."""
        # Log model parameters
        mlflow.log_param("network_type", self.network_type)
        mlflow.log_param("input_dim", self.input_dim)
        mlflow.log_param("output_dim", self.output_dim)
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("weight_decay", self.weight_decay)
        
        # Log network parameters
        for key, value in self.network_params.items():
            mlflow.log_param(f"network_{key}", value)
        
        # Log network architecture
        if self.network is not None:
            network_config = self.network.get_config()
            mlflow.log_param("network_config", json.dumps(network_config))


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot to (optional)
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    
    # Plot metrics
    if "val_auc" in history:
        axes[1].plot(history["val_auc"], label="AUC")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Accuracy")
    if "val_f1" in history:
        axes[1].plot(history["val_f1"], label="F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def train_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    target_col: str = "outcome",
    network_type: str = "mlp",
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    normalize: bool = True,
    verbose: bool = True,
    save_path: Optional[str] = None,
    **network_params
) -> Tuple[CKDModel, Dict[str, Any]]:
    """
    Train a model on the given data.
    
    Args:
        train_df: Training data
        val_df: Validation data (optional)
        target_col: Name of the target column (default: "outcome")
        network_type: Type of network to use (default: "mlp")
        batch_size: Batch size (default: 64)
        epochs: Number of epochs (default: 100)
        patience: Patience for early stopping (default: 10)
        learning_rate: Learning rate (default: 0.001)
        weight_decay: Weight decay (default: 0.0001)
        normalize: Whether to normalize features (default: True)
        verbose: Whether to print progress (default: True)
        save_path: Path to save the model to (optional)
        **network_params: Additional parameters for the network
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create datasets
    train_dataset = TabularDataset(
        data=train_df,
        target_col=target_col,
        normalize=normalize
    )
    
    if val_df is not None:
        val_dataset = TabularDataset(
            data=val_df,
            target_col=target_col,
            normalize=normalize
        )
    else:
        val_dataset = None
    
    # Create model
    model = CKDModel(
        network_type=network_type,
        input_dim=train_dataset.n_features,
        output_dim=1,  # Binary classification
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **network_params
    )
    
    # Train model
    history = model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        verbose=verbose
    )
    
    # Save model if path is provided
    if save_path is not None:
        model.save(save_path)
    
    # Log to MLflow
    model.log_to_mlflow()
    
    # Plot training history
    if verbose:
        plot_training_history(history)
    
    # Return model and history
    return model, history


def evaluate_model(
    model: CKDModel,
    test_df: pd.DataFrame,
    target_col: str = "outcome",
    normalize: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model on the given data.
    
    Args:
        model: Model to evaluate
        test_df: Test data
        target_col: Name of the target column (default: "outcome")
        normalize: Whether to normalize features (default: True)
        verbose: Whether to print progress (default: True)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create dataset
    test_dataset = TabularDataset(
        data=test_df,
        target_col=target_col,
        normalize=normalize
    )
    
    # Evaluate model
    metrics = model.evaluate(test_dataset)
    
    # Print metrics
    if verbose:
        print("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Return metrics
    return metrics