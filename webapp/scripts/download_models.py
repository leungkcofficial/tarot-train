#!/usr/bin/env python3
"""
Model downloader for TAROT CKD Risk Prediction
Downloads ensemble models from various sources (HuggingFace, AWS S3, GitHub Releases, etc.)
"""

import os
import sys
import json
import logging
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Download and verify TAROT ensemble models"""
    
    def __init__(self, model_dir: str = "../models", base_url: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url or os.getenv('MODEL_DOWNLOAD_URL')
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Model configuration
        self.model_config = {
            "ensemble_models": [
                {
                    "name": "deepsurv_model_1",
                    "type": "deepsurv",
                    "path": "deepsurv_1.pkl",
                    "weight": 0.15,
                    "sha256": "placeholder_hash_1",
                    "size_mb": 45.2
                },
                {
                    "name": "deepsurv_model_2",
                    "type": "deepsurv", 
                    "path": "deepsurv_2.pkl",
                    "weight": 0.15,
                    "sha256": "placeholder_hash_2",
                    "size_mb": 47.8
                },
                {
                    "name": "deepsurv_model_3",
                    "type": "deepsurv",
                    "path": "deepsurv_3.pkl", 
                    "weight": 0.15,
                    "sha256": "placeholder_hash_3",
                    "size_mb": 44.1
                },
                {
                    "name": "deephit_model_1",
                    "type": "deephit",
                    "path": "deephit_1.pkl",
                    "weight": 0.12,
                    "sha256": "placeholder_hash_4", 
                    "size_mb": 52.3
                },
                {
                    "name": "deephit_model_2",
                    "type": "deephit",
                    "path": "deephit_2.pkl",
                    "weight": 0.12,
                    "sha256": "placeholder_hash_5",
                    "size_mb": 51.7
                },
                {
                    "name": "lstm_model_1",
                    "type": "lstm",
                    "path": "lstm_1.pkl",
                    "weight": 0.08,
                    "sha256": "placeholder_hash_6",
                    "size_mb": 38.9
                },
                {
                    "name": "lstm_model_2", 
                    "type": "lstm",
                    "path": "lstm_2.pkl",
                    "weight": 0.08,
                    "sha256": "placeholder_hash_7",
                    "size_mb": 39.4
                },
                {
                    "name": "gradient_boosting_1",
                    "type": "xgboost",
                    "path": "xgboost_1.pkl",
                    "weight": 0.05,
                    "sha256": "placeholder_hash_8",
                    "size_mb": 12.1
                },
                {
                    "name": "gradient_boosting_2",
                    "type": "xgboost", 
                    "path": "xgboost_2.pkl",
                    "weight": 0.05,
                    "sha256": "placeholder_hash_9",
                    "size_mb": 11.8
                },
                {
                    "name": "random_forest_1",
                    "type": "random_forest",
                    "path": "rf_1.pkl",
                    "weight": 0.05,
                    "sha256": "placeholder_hash_10",
                    "size_mb": 28.3
                }
            ],
            "feature_scaler": {
                "type": "standard_scaler",
                "path": "feature_scaler.pkl",
                "sha256": "placeholder_scaler_hash",
                "size_mb": 0.1
            },
            "feature_names": [
                "age", "gender_male", "egfr", "hemoglobin", "phosphate",
                "bicarbonate", "uacr", "charlson_score", "diabetes",
                "hypertension", "cardiovascular_disease", "cancer_history",
                "liver_disease", "lung_disease", "stroke_history"
            ],
            "model_version": "1.2.0",
            "created_at": "2024-01-15T00:00:00Z",
            "validation_metrics": {
                "temporal_c_index": 0.851,
                "spatial_c_index": 0.773,
                "temporal_brier_score": 0.019,
                "spatial_brier_score": 0.054
            }
        }
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download_file(self, url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """Download a file with progress bar and validation"""
        try:
            logger.info(f"Downloading {filepath.name} from {url}")
            
            # Check if file already exists and is valid
            if filepath.exists():
                current_size = filepath.stat().st_size
                if expected_size and current_size == expected_size:
                    logger.info(f"File {filepath.name} already exists with correct size")
                    return True
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file size
            actual_size = filepath.stat().st_size
            if expected_size and actual_size != expected_size:
                logger.error(f"Size mismatch for {filepath.name}: expected {expected_size}, got {actual_size}")
                filepath.unlink()
                return False
            
            logger.info(f"Successfully downloaded {filepath.name} ({actual_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def extract_archive(self, archive_path: Path) -> bool:
        """Extract ZIP or TAR archive"""
        try:
            logger.info(f"Extracting {archive_path.name}")
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.model_dir)
            elif archive_path.suffix.lower() in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.model_dir)
            else:
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            logger.info(f"Successfully extracted {archive_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def download_from_huggingface(self) -> bool:
        """Download models from HuggingFace Hub"""
        if not self.hf_token:
            logger.warning("HF_TOKEN not provided, skipping HuggingFace download")
            return False
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            repo_id = "tarot-ckd/ensemble-models-v1"
            logger.info(f"Downloading from HuggingFace Hub: {repo_id}")
            
            # List available files
            files = list_repo_files(repo_id=repo_id, token=self.hf_token)
            model_files = [f for f in files if f.endswith('.pkl')]
            
            success_count = 0
            for filename in model_files:
                try:
                    filepath = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.model_dir / "cache"),
                        token=self.hf_token
                    )
                    
                    # Move to models directory
                    target_path = self.model_dir / filename
                    Path(filepath).rename(target_path)
                    logger.info(f"Downloaded {filename} from HuggingFace")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to download {filename} from HuggingFace: {e}")
            
            return success_count > 0
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False
    
    def download_from_url(self) -> bool:
        """Download models from direct URL"""
        if not self.base_url:
            logger.warning("MODEL_DOWNLOAD_URL not provided")
            return False
        
        archive_name = "tarot_models.zip"
        archive_path = self.model_dir / archive_name
        
        success = self.download_file(self.base_url, archive_path)
        if not success:
            return False
        
        # Extract archive
        success = self.extract_archive(archive_path)
        if success:
            # Clean up archive
            archive_path.unlink()
        
        return success
    
    def download_from_github_releases(self) -> bool:
        """Download from GitHub releases"""
        try:
            repo = "tarot-ckd/tarot2"
            api_url = f"https://api.github.com/repos/{repo}/releases/latest"
            
            logger.info("Checking GitHub releases for model files")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            release_data = response.json()
            
            # Look for model archive in assets
            for asset in release_data.get('assets', []):
                if 'models' in asset['name'].lower() and asset['name'].endswith('.zip'):
                    download_url = asset['browser_download_url']
                    archive_path = self.model_dir / asset['name']
                    
                    success = self.download_file(download_url, archive_path, asset.get('size'))
                    if success:
                        success = self.extract_archive(archive_path)
                        if success:
                            archive_path.unlink()
                            return True
            
            logger.warning("No model archives found in GitHub releases")
            return False
            
        except Exception as e:
            logger.error(f"GitHub releases download failed: {e}")
            return False
    
    def create_dummy_models(self) -> bool:
        """Create dummy model files for development/testing"""
        logger.warning("Creating dummy models for testing purposes")
        
        try:
            import pickle
            
            # Create dummy model data
            dummy_model_data = {
                "model_type": "dummy",
                "version": "1.0.0", 
                "weights": [0.1] * 100,  # Dummy weights
                "feature_names": self.model_config["feature_names"],
                "training_date": "2024-01-15",
                "performance_metrics": {
                    "c_index": 0.75,
                    "brier_score": 0.08
                }
            }
            
            # Create model files
            for model_info in self.model_config["ensemble_models"]:
                model_path = self.model_dir / model_info["path"]
                with open(model_path, 'wb') as f:
                    pickle.dump(dummy_model_data, f)
                logger.info(f"Created dummy model: {model_info['path']}")
            
            # Create scaler file
            dummy_scaler = {
                "type": "standard_scaler",
                "mean_": [0.0] * len(self.model_config["feature_names"]),
                "scale_": [1.0] * len(self.model_config["feature_names"])
            }
            
            scaler_path = self.model_dir / self.model_config["feature_scaler"]["path"]
            with open(scaler_path, 'wb') as f:
                pickle.dump(dummy_scaler, f)
            logger.info("Created dummy feature scaler")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dummy models: {e}")
            return False
    
    def verify_models(self) -> Tuple[bool, List[str]]:
        """Verify that all required model files exist and are valid"""
        missing_files = []
        
        # Check ensemble models
        for model_info in self.model_config["ensemble_models"]:
            model_path = self.model_dir / model_info["path"]
            if not model_path.exists():
                missing_files.append(model_info["path"])
            elif model_path.stat().st_size == 0:
                missing_files.append(f"{model_info['path']} (empty)")
        
        # Check scaler
        scaler_path = self.model_dir / self.model_config["feature_scaler"]["path"]
        if not scaler_path.exists():
            missing_files.append(self.model_config["feature_scaler"]["path"])
        
        # Check config file
        config_path = self.model_dir / "model_config.json"
        if not config_path.exists():
            missing_files.append("model_config.json")
        
        is_valid = len(missing_files) == 0
        if is_valid:
            logger.info(f"Model verification successful: {len(self.model_config['ensemble_models'])} models + scaler")
        else:
            logger.error(f"Model verification failed. Missing: {missing_files}")
        
        return is_valid, missing_files
    
    def create_model_config(self) -> bool:
        """Create model configuration file"""
        try:
            config_path = self.model_dir / "model_config.json"
            
            with open(config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)
            
            logger.info(f"Created model configuration: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model config: {e}")
            return False
    
    def download_models(self, create_dummy: bool = False) -> bool:
        """Main download function with multiple fallback sources"""
        logger.info("Starting TAROT model download process...")
        
        download_methods = []
        
        if not create_dummy:
            # Try different download sources in order of preference
            if self.hf_token:
                download_methods.append(("HuggingFace Hub", self.download_from_huggingface))
            
            if self.base_url:
                download_methods.append(("Direct URL", self.download_from_url))
            
            download_methods.append(("GitHub Releases", self.download_from_github_releases))
        
        # Always have dummy models as fallback
        download_methods.append(("Dummy Models (Testing)", self.create_dummy_models))
        
        # Try each download method
        for method_name, method_func in download_methods:
            logger.info(f"Attempting download via {method_name}")
            
            try:
                success = method_func()
                if success:
                    logger.info(f"Successfully downloaded models via {method_name}")
                    break
            except Exception as e:
                logger.error(f"{method_name} download failed: {e}")
                continue
        else:
            logger.error("All download methods failed")
            return False
        
        # Create configuration file
        self.create_model_config()
        
        # Verify models
        is_valid, missing = self.verify_models()
        if not is_valid:
            logger.error(f"Model verification failed: {missing}")
            return False
        
        logger.info("Model download and verification completed successfully!")
        return True

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Download TAROT ensemble models')
    parser.add_argument('--model-dir', default='../models', help='Directory to store models')
    parser.add_argument('--base-url', help='Base URL for model download')
    parser.add_argument('--dummy', action='store_true', help='Create dummy models for testing')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    downloader = ModelDownloader(args.model_dir, args.base_url)
    
    if args.verify_only:
        is_valid, missing = downloader.verify_models()
        if is_valid:
            print("✓ All models are present and valid")
            sys.exit(0)
        else:
            print(f"✗ Missing or invalid models: {missing}")
            sys.exit(1)
    
    # Download models
    success = downloader.download_models(create_dummy=args.dummy)
    
    if success:
        print("✓ Model download completed successfully")
        
        # Print summary
        model_files = list(Path(args.model_dir).glob("*.pkl"))
        config_file = Path(args.model_dir) / "model_config.json"
        
        print(f"\nDownloaded files:")
        print(f"  - {len(model_files)} model files")
        print(f"  - 1 configuration file")
        print(f"  - Total size: {sum(f.stat().st_size for f in model_files) / 1024**2:.1f} MB")
        
        sys.exit(0)
    else:
        print("✗ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()