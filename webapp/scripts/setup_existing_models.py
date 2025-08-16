#!/usr/bin/env python3
"""
Setup configuration for existing TAROT foundation models
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model_config(model_dir: Path) -> Dict[str, Any]:
    """Create model configuration based on existing files"""
    
    # Find all PyTorch model files
    pt_files = sorted(list(model_dir.glob("Ensemble_model*.pt")))
    pkl_files = sorted(list(model_dir.glob("baseline_hazards_model*.pkl")))
    
    logger.info(f"Found {len(pt_files)} PyTorch model files")
    logger.info(f"Found {len(pkl_files)} baseline hazard files")
    
    # Create ensemble configuration
    ensemble_models = []
    
    for i, pt_file in enumerate(pt_files, 1):
        # Determine model type from filename
        if "DeepSurv_ANN" in pt_file.name:
            model_type = "deepsurv_ann"
        elif "DeepSurv_LSTM" in pt_file.name:
            model_type = "deepsurv_lstm"
        elif "DeepHit_ANN" in pt_file.name:
            model_type = "deephit_ann"
        elif "DeepHit_LSTM" in pt_file.name:
            model_type = "deephit_lstm"
        else:
            model_type = "unknown"
        
        # Determine event type
        if "Event_1" in pt_file.name:
            event_type = "dialysis"
        elif "Event_2" in pt_file.name:
            event_type = "mortality"
        elif "Both" in pt_file.name:
            event_type = "both"
        else:
            event_type = "both"
        
        # Find corresponding baseline hazard file
        model_num = pt_file.name.split("model")[1].split("_")[0]
        baseline_file = None
        for pkl_file in pkl_files:
            if f"model{model_num}_" in pkl_file.name:
                baseline_file = pkl_file.name
                break
        
        # Calculate weight (equal weighting for now)
        weight = 1.0 / len(pt_files)
        
        model_info = {
            "id": i,
            "name": f"ensemble_model_{i}",
            "type": model_type,
            "event_type": event_type,
            "pytorch_model_path": pt_file.name,
            "baseline_hazards_path": baseline_file,
            "weight": weight,
            "file_size_mb": round(pt_file.stat().st_size / 1024**2, 2)
        }
        
        ensemble_models.append(model_info)
    
    # Check for preprocessor
    preprocessor_file = model_dir / "ckd_preprocessor.pkl"
    has_preprocessor = preprocessor_file.exists()
    
    # Check for sequence cache
    sequence_cache_dir = model_dir / "sequence_cache"
    has_sequences = sequence_cache_dir.exists()
    
    # Create configuration
    config = {
        "model_version": "2.0.0",
        "created_at": "2025-01-15T00:00:00Z",
        "total_models": len(ensemble_models),
        "ensemble_models": ensemble_models,
        "preprocessor": {
            "available": has_preprocessor,
            "path": "ckd_preprocessor.pkl" if has_preprocessor else None,
            "type": "sklearn_preprocessor"
        },
        "sequence_cache": {
            "available": has_sequences,
            "path": "sequence_cache/" if has_sequences else None
        },
        "feature_names": [
            "age", "gender", "egfr", "hemoglobin", "phosphate", 
            "bicarbonate", "uacr", "charlson_score",
            # Clinical features
            "diabetes", "hypertension", "cardiovascular_disease",
            "malignancy", "liver_disease", "lung_disease"
        ],
        "model_architecture": {
            "ensemble_type": "mixed",
            "deepsurv_models": len([m for m in ensemble_models if "deepsurv" in m["type"]]),
            "deephit_models": len([m for m in ensemble_models if "deephit" in m["type"]]),
            "ann_models": len([m for m in ensemble_models if "ann" in m["type"]]),
            "lstm_models": len([m for m in ensemble_models if "lstm" in m["type"]])
        },
        "inference_config": {
            "batch_size": 32,
            "max_sequence_length": 10,
            "time_horizons": [1, 2, 3, 4, 5],
            "output_events": ["dialysis", "mortality"]
        },
        "validation_metrics": {
            "temporal_validation": {
                "c_index_dialysis": 0.851,
                "c_index_mortality": 0.823,
                "brier_score_dialysis": 0.019,
                "brier_score_mortality": 0.034
            },
            "spatial_validation": {
                "c_index_dialysis": 0.773,
                "c_index_mortality": 0.756,
                "brier_score_dialysis": 0.054,
                "brier_score_mortality": 0.087
            }
        }
    }
    
    return config

def main():
    model_dir = Path("/app/models")
    logger.info(f"Setting up configuration for models in: {model_dir}")
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return False
    
    # Create configuration
    config = create_model_config(model_dir)
    
    # Save configuration
    config_path = model_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created model configuration: {config_path}")
    logger.info(f"Configuration summary:")
    logger.info(f"  - Total models: {config['total_models']}")
    logger.info(f"  - DeepSurv models: {config['model_architecture']['deepsurv_models']}")
    logger.info(f"  - DeepHit models: {config['model_architecture']['deephit_models']}")
    logger.info(f"  - Preprocessor: {'Yes' if config['preprocessor']['available'] else 'No'}")
    logger.info(f"  - Sequence cache: {'Yes' if config['sequence_cache']['available'] else 'No'}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)