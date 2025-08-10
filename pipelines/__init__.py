"""
Pipelines package for CKD Risk Prediction

This package contains ZenML pipelines for the CKD Risk Prediction project.
"""

# Import pipelines for easier access
from pipelines.training_pipeline import train_pipeline

__all__ = [
    'train_pipeline'
]