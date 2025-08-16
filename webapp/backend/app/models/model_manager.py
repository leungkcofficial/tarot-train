"""
Model management system for TAROT CKD Risk Prediction
Handles loading, caching, and inference with ensemble models
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import structlog

from app.core.config import settings
from app.models.comprehensive_model_manager import ComprehensiveModelManager


logger = structlog.get_logger(__name__)


class ModelManager:
    """Manages ensemble model loading and inference - Updated to use ComprehensiveModelManager"""
    
    def __init__(self):
        self.comprehensive_manager = ComprehensiveModelManager()
        self.is_loaded = False
        
    async def load_models(self) -> None:
        """Load all 36 ensemble models using the comprehensive manager"""
        logger.info("Starting comprehensive model loading for all 36 models")
        start_time = time.time()
        
        try:
            await self.comprehensive_manager.load_all_models()
            self.is_loaded = True
            
            load_time = time.time() - start_time
            status = self.comprehensive_manager.get_status()
            
            logger.info(
                "Model loading completed successfully",
                loaded_models=status['loaded_models'],
                total_models=status['total_models'],
                load_percentage=status['load_percentage'],
                load_time_seconds=load_time
            )
            
        except Exception as e:
            logger.error("Failed to load models", error=str(e), exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")
    
    
    async def predict(self, patient_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate predictions using all 36 ensemble models
        
        Args:
            patient_data: Patient data as DataFrame or dictionary
            
        Returns:
            Dictionary containing ensemble predictions, SHAP values, and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert DataFrame to dict if needed
        if isinstance(patient_data, pd.DataFrame):
            patient_data = self._convert_dataframe_to_dict(patient_data)
        
        try:
            # Use comprehensive manager for prediction
            result = await self.comprehensive_manager.predict(patient_data)
            
            # Add SHAP values (placeholder for now)
            if 'shap_values' not in result:
                result['shap_values'] = self._generate_placeholder_shap(result.get('predictions', {}))
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _convert_dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to patient data dictionary format"""
        # This is a simplified conversion - may need adjustment based on actual data format
        if len(df) == 0:
            raise ValueError("Empty DataFrame provided")
        
        row = df.iloc[0]  # Take first row
        
        return {
            "demographics": {
                "age": row.get("age", 65),
                "gender": row.get("gender", "male")
            },
            "laboratory_values": [
                {"parameter": "creatinine", "value": row.get("creatinine", 150)},
                {"parameter": "hemoglobin", "value": row.get("hemoglobin", 11.0)},
                {"parameter": "phosphate", "value": row.get("phosphate", 1.2)},
                {"parameter": "bicarbonate", "value": row.get("bicarbonate", 24)},
                {"parameter": "albumin", "value": row.get("albumin", 35)},
                {"parameter": "uacr", "value": row.get("uacr", 100)},
            ],
            "medical_history": [
                {"condition": "hypertension", "diagnosed": bool(row.get("ht", False))},
                {"condition": "diabetes", "diagnosed": bool(row.get("dm", False))}
            ]
        }
    
    def _generate_placeholder_shap(self, predictions: Dict[str, List]) -> Dict[str, Dict[str, float]]:
        """Generate placeholder SHAP values"""
        feature_names = [
            'age_at_obs', 'gender', 'creatinine', 'hemoglobin', 'phosphate',
            'bicarbonate', 'albumin', 'uacr', 'cci_score_total', 'ht', 'observation_period'
        ]
        
        # Simple placeholder SHAP values
        dialysis_shap = {name: np.random.normal(0, 0.02) for name in feature_names}
        mortality_shap = {name: np.random.normal(0, 0.02) for name in feature_names}
        
        return {
            'dialysis': dialysis_shap,
            'mortality': mortality_shap
        }
    
    async def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess patient data using the loaded preprocessor"""
        def preprocess():
            try:
                # Apply the preprocessor
                processed = self.preprocessor.transform(data)
                
                # Ensure we have exactly 10 time points (pad with zeros if needed)
                if len(processed) < 10:
                    padding = np.zeros((10 - len(processed), processed.shape[1]))
                    processed = np.vstack([processed, padding])
                elif len(processed) > 10:
                    processed = processed[-10:]  # Take last 10 time points
                
                return processed
                
            except Exception as e:
                logger.error("Preprocessing failed", error=str(e))
                raise
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, preprocess
        )
    
    async def _generate_ensemble_predictions(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Generate predictions from the ensemble models"""
        def predict_batch(model_batch):
            batch_predictions = {
                'dialysis': [],
                'mortality': []
            }
            
            for model_name, model_state in model_batch.items():
                try:
                    # Convert to tensor
                    tensor_data = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
                    
                    # For now, return dummy predictions
                    # TODO: Implement actual model inference based on model architecture
                    dialysis_pred = np.random.random(5) * 0.5  # 5 time horizons
                    mortality_pred = np.random.random(5) * 0.4
                    
                    batch_predictions['dialysis'].append(dialysis_pred)
                    batch_predictions['mortality'].append(mortality_pred)
                    
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {model_name}", error=str(e))
            
            return batch_predictions
        
        # Split models into batches for parallel processing
        model_items = list(self.models.items())
        batch_size = 6
        
        all_predictions = {'dialysis': [], 'mortality': []}
        
        for i in range(0, len(model_items), batch_size):
            batch = dict(model_items[i:i + batch_size])
            batch_preds = await asyncio.get_event_loop().run_in_executor(
                self.executor, predict_batch, batch
            )
            
            all_predictions['dialysis'].extend(batch_preds['dialysis'])
            all_predictions['mortality'].extend(batch_preds['mortality'])
        
        # Ensemble averaging
        ensemble_dialysis = np.mean(all_predictions['dialysis'], axis=0).tolist()
        ensemble_mortality = np.mean(all_predictions['mortality'], axis=0).tolist()
        
        return {
            'dialysis_risk': ensemble_dialysis,
            'mortality_risk': ensemble_mortality
        }
    
    async def _calculate_shap_values(self, data: np.ndarray, predictions: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate SHAP values for feature importance"""
        # TODO: Implement actual SHAP calculation
        # For now, return dummy SHAP values
        
        feature_names = [
            'age_at_obs', 'gender', 'creatinine', 'hemoglobin', 'phosphate',
            'bicarbonate', 'albumin', 'uacr', 'cci_score_total', 'ht', 'observation_period'
        ]
        
        # Generate random SHAP values that sum to approximately the prediction
        dialysis_shap = {name: np.random.normal(0, 0.1) for name in feature_names}
        mortality_shap = {name: np.random.normal(0, 0.1) for name in feature_names}
        
        return {
            'dialysis': dialysis_shap,
            'mortality': mortality_shap
        }
    
    def _get_model_type_counts(self) -> Dict[str, int]:
        """Get count of each model type"""
        type_counts = {}
        for metadata in self.model_metadata.values():
            model_type = metadata.get('model_type', 'unknown')
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        return type_counts
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up model manager resources")
        
        # Clear models from memory
        self.models.clear()
        self.baseline_hazards.clear()
        self.model_metadata.clear()
        self.preprocessor = None
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.is_loaded = False
        logger.info("Model manager cleanup completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all 36 models"""
        if self.is_loaded:
            return self.comprehensive_manager.get_status()
        else:
            return {
                'is_loaded': False,
                'total_models': 36,
                'loaded_models': 0,
                'load_percentage': 0,
                'preprocessor_loaded': False,
                'model_details': {}
            }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up model manager resources")
        self.is_loaded = False
        # The comprehensive manager handles its own cleanup
        logger.info("Model manager cleanup completed")