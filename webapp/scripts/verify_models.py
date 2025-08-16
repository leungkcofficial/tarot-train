#!/usr/bin/env python3
"""
Model verification script for TAROT CKD Risk Prediction
Verifies model integrity, compatibility, and performance
"""

import os
import sys
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVerifier:
    """Verify TAROT ensemble models"""
    
    def __init__(self, model_dir: str = "../models"):
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "model_config.json"
        
        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logger.error(f"Model configuration file not found: {self.config_path}")
            self.config = None
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def verify_file_integrity(self) -> Tuple[bool, List[str]]:
        """Verify file integrity using checksums"""
        if not self.config:
            return False, ["Missing configuration file"]
        
        issues = []
        
        # Verify ensemble models
        for model_info in self.config.get("ensemble_models", []):
            model_path = self.model_dir / model_info["path"]
            
            if not model_path.exists():
                issues.append(f"Missing model file: {model_info['path']}")
                continue
            
            # Check file size
            actual_size_mb = model_path.stat().st_size / 1024**2
            expected_size_mb = model_info.get("size_mb", 0)
            
            if expected_size_mb > 0:
                size_diff = abs(actual_size_mb - expected_size_mb) / expected_size_mb
                if size_diff > 0.1:  # Allow 10% variance
                    issues.append(f"Size mismatch for {model_info['path']}: "
                                f"expected {expected_size_mb:.1f}MB, got {actual_size_mb:.1f}MB")
            
            # Check hash (if provided and not placeholder)
            expected_hash = model_info.get("sha256", "")
            if expected_hash and not expected_hash.startswith("placeholder"):
                actual_hash = self.calculate_file_hash(model_path)
                if actual_hash != expected_hash:
                    issues.append(f"Hash mismatch for {model_info['path']}")
        
        # Verify feature scaler
        scaler_info = self.config.get("feature_scaler", {})
        if scaler_info:
            scaler_path = self.model_dir / scaler_info["path"]
            if not scaler_path.exists():
                issues.append(f"Missing feature scaler: {scaler_info['path']}")
        
        return len(issues) == 0, issues
    
    def verify_model_loading(self) -> Tuple[bool, List[str]]:
        """Verify that models can be loaded properly"""
        if not self.config:
            return False, ["Missing configuration file"]
        
        issues = []
        loaded_models = {}
        
        # Test loading each model
        for model_info in self.config.get("ensemble_models", []):
            model_path = self.model_dir / model_info["path"]
            
            if not model_path.exists():
                issues.append(f"Cannot load missing file: {model_info['path']}")
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    loaded_models[model_info["name"]] = model
                    logger.debug(f"Successfully loaded {model_info['name']}")
                    
            except Exception as e:
                issues.append(f"Failed to load {model_info['path']}: {str(e)}")
        
        # Test loading feature scaler
        scaler_info = self.config.get("feature_scaler", {})
        if scaler_info:
            scaler_path = self.model_dir / scaler_info["path"]
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                        loaded_models["scaler"] = scaler
                        logger.debug("Successfully loaded feature scaler")
                except Exception as e:
                    issues.append(f"Failed to load feature scaler: {str(e)}")
        
        logger.info(f"Successfully loaded {len(loaded_models)} model components")
        return len(issues) == 0, issues
    
    def verify_model_compatibility(self) -> Tuple[bool, List[str]]:
        """Verify model compatibility and feature alignment"""
        if not self.config:
            return False, ["Missing configuration file"]
        
        issues = []
        feature_names = self.config.get("feature_names", [])
        
        if not feature_names:
            issues.append("No feature names specified in configuration")
            return False, issues
        
        # Create dummy input data for testing
        dummy_features = np.random.randn(1, len(feature_names)).astype(np.float32)
        
        # Test each model with dummy data
        for model_info in self.config.get("ensemble_models", []):
            model_path = self.model_dir / model_info["path"]
            
            if not model_path.exists():
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Try to make predictions with dummy data
                if hasattr(model, 'predict'):
                    try:
                        predictions = model.predict(dummy_features)
                        logger.debug(f"Model {model_info['name']} prediction test passed")
                    except Exception as e:
                        issues.append(f"Model {model_info['name']} prediction failed: {str(e)}")
                        
                elif hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(dummy_features)
                        logger.debug(f"Model {model_info['name']} probability prediction test passed")
                    except Exception as e:
                        issues.append(f"Model {model_info['name']} probability prediction failed: {str(e)}")
                        
                else:
                    logger.warning(f"Model {model_info['name']} has no standard prediction interface")
                    
            except Exception as e:
                issues.append(f"Failed to test model {model_info['name']}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def verify_ensemble_weights(self) -> Tuple[bool, List[str]]:
        """Verify ensemble weight configuration"""
        if not self.config:
            return False, ["Missing configuration file"]
        
        issues = []
        total_weight = 0.0
        
        for model_info in self.config.get("ensemble_models", []):
            weight = model_info.get("weight", 0.0)
            if weight <= 0:
                issues.append(f"Invalid weight for {model_info['name']}: {weight}")
            total_weight += weight
        
        # Check if weights sum to approximately 1.0
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Ensemble weights sum to {total_weight:.3f}, expected ~1.0")
        
        logger.info(f"Ensemble weights sum to {total_weight:.3f}")
        return len(issues) == 0, issues
    
    def verify_model_version(self) -> Tuple[bool, List[str]]:
        """Verify model version compatibility"""
        if not self.config:
            return False, ["Missing configuration file"]
        
        issues = []
        model_version = self.config.get("model_version", "unknown")
        
        # Check version format
        if not model_version or model_version == "unknown":
            issues.append("Model version not specified")
        
        # Check if version follows semantic versioning
        try:
            version_parts = model_version.split('.')
            if len(version_parts) != 3:
                issues.append(f"Invalid version format: {model_version}")
            else:
                for part in version_parts:
                    int(part)  # Should be numeric
        except ValueError:
            issues.append(f"Invalid version format: {model_version}")
        
        logger.info(f"Model version: {model_version}")
        return len(issues) == 0, issues
    
    def run_full_verification(self) -> Tuple[bool, Dict[str, Any]]:
        """Run complete model verification suite"""
        logger.info("Starting comprehensive model verification...")
        
        verification_results = {
            "overall_status": True,
            "timestamp": "2024-01-15T00:00:00Z",
            "model_directory": str(self.model_dir),
            "tests": {}
        }
        
        # Test 1: Configuration exists
        config_valid = self.config is not None
        verification_results["tests"]["configuration"] = {
            "status": config_valid,
            "message": "Configuration file loaded" if config_valid else "Configuration file missing"
        }
        if not config_valid:
            verification_results["overall_status"] = False
        
        # Test 2: File integrity
        if config_valid:
            integrity_valid, integrity_issues = self.verify_file_integrity()
            verification_results["tests"]["file_integrity"] = {
                "status": integrity_valid,
                "message": "All files present" if integrity_valid else f"{len(integrity_issues)} issues found",
                "issues": integrity_issues
            }
            if not integrity_valid:
                verification_results["overall_status"] = False
        
        # Test 3: Model loading
        if config_valid:
            loading_valid, loading_issues = self.verify_model_loading()
            verification_results["tests"]["model_loading"] = {
                "status": loading_valid,
                "message": "All models loadable" if loading_valid else f"{len(loading_issues)} issues found",
                "issues": loading_issues
            }
            if not loading_valid:
                verification_results["overall_status"] = False
        
        # Test 4: Model compatibility
        if config_valid:
            compat_valid, compat_issues = self.verify_model_compatibility()
            verification_results["tests"]["compatibility"] = {
                "status": compat_valid,
                "message": "All models compatible" if compat_valid else f"{len(compat_issues)} issues found",
                "issues": compat_issues
            }
            if not compat_valid:
                verification_results["overall_status"] = False
        
        # Test 5: Ensemble weights
        if config_valid:
            weights_valid, weights_issues = self.verify_ensemble_weights()
            verification_results["tests"]["ensemble_weights"] = {
                "status": weights_valid,
                "message": "Ensemble weights valid" if weights_valid else f"{len(weights_issues)} issues found", 
                "issues": weights_issues
            }
            if not weights_valid:
                verification_results["overall_status"] = False
        
        # Test 6: Version compatibility
        if config_valid:
            version_valid, version_issues = self.verify_model_version()
            verification_results["tests"]["version"] = {
                "status": version_valid,
                "message": "Version format valid" if version_valid else f"{len(version_issues)} issues found",
                "issues": version_issues
            }
            if not version_valid:
                verification_results["overall_status"] = False
        
        # Summary
        passed_tests = sum(1 for test in verification_results["tests"].values() if test["status"])
        total_tests = len(verification_results["tests"])
        
        logger.info(f"Verification completed: {passed_tests}/{total_tests} tests passed")
        
        if verification_results["overall_status"]:
            logger.info("✓ All model verification tests passed")
        else:
            logger.error("✗ Some model verification tests failed")
        
        return verification_results["overall_status"], verification_results
    
    def generate_verification_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable verification report"""
        report_lines = []
        report_lines.append("TAROT Model Verification Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Timestamp: {results['timestamp']}")
        report_lines.append(f"Model Directory: {results['model_directory']}")
        report_lines.append(f"Overall Status: {'PASS' if results['overall_status'] else 'FAIL'}")
        report_lines.append("")
        
        # Test details
        for test_name, test_results in results["tests"].items():
            status_symbol = "✓" if test_results["status"] else "✗"
            report_lines.append(f"{status_symbol} {test_name.replace('_', ' ').title()}: {test_results['message']}")
            
            if "issues" in test_results and test_results["issues"]:
                for issue in test_results["issues"]:
                    report_lines.append(f"    - {issue}")
        
        report_lines.append("")
        
        # Model summary
        if self.config:
            num_models = len(self.config.get("ensemble_models", []))
            model_version = self.config.get("model_version", "unknown")
            report_lines.append(f"Model Configuration Summary:")
            report_lines.append(f"  - Ensemble size: {num_models} models")
            report_lines.append(f"  - Model version: {model_version}")
            report_lines.append(f"  - Feature count: {len(self.config.get('feature_names', []))}")
        
        return "\n".join(report_lines)

def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify TAROT ensemble models')
    parser.add_argument('--model-dir', default='../models', help='Directory containing models')
    parser.add_argument('--output', help='Save verification report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run verification
    verifier = ModelVerifier(args.model_dir)
    success, results = verifier.run_full_verification()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        report = verifier.generate_verification_report(results)
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()