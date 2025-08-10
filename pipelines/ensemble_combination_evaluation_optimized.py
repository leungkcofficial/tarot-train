"""
Optimized Ensemble Combination Evaluation Pipeline

This version includes:
- Parallel processing for bootstrap evaluation
- Smart sampling strategies for large-scale evaluation
- Memory-efficient processing
- Resume capability for interrupted runs
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations, islice
from typing import Dict, List, Tuple, Any, Generator
import warnings
from tqdm import tqdm
import pickle
from scipy import stats
from multiprocessing import Pool, cpu_count
import hashlib
from functools import partial

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation_metrics import (
    concordance_index_censored,
    integrated_brier_score,
    negative_log_likelihood
)


class OptimizedEnsembleEvaluator:
    """Optimized evaluator with parallel processing and smart sampling"""
    
    def __init__(self, 
                 predictions_dir: str,
                 output_dir: str,
                 n_bootstraps: int = 10,
                 random_seed: int = 42,
                 n_jobs: int = -1):
        """
        Initialize evaluator
        
        Args:
            predictions_dir: Directory containing individual model predictions
            output_dir: Directory to save evaluation results
            n_bootstraps: Number of bootstrap iterations
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        
        # Time points for evaluation
        self.time_points = np.array([365, 730, 1095, 1460, 1825])
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Resume capability
        self.checkpoint_file = os.path.join(output_dir, 'evaluation_checkpoint.pkl')
        self.completed_combinations = set()
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load checkpoint for resume capability"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                self.completed_combinations = checkpoint.get('completed_combinations', set())
                print(f"Resuming from checkpoint: {len(self.completed_combinations)} combinations already evaluated")
    
    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'completed_combinations': self.completed_combinations,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def get_combination_hash(self, model_keys: List[str]) -> str:
        """Get unique hash for a combination"""
        return hashlib.md5(','.join(sorted(model_keys)).encode()).hexdigest()
    
    def smart_sampling_strategy(self, all_keys: List[str], strategy: str = 'diverse') -> Generator:
        """
        Generate combinations using smart sampling strategies
        
        Args:
            all_keys: All model keys
            strategy: Sampling strategy ('diverse', 'size_based', 'random', 'all')
            
        Yields:
            Model combinations
        """
        n_models = len(all_keys)
        
        if strategy == 'all':
            # Generate all combinations
            for r in range(1, n_models + 1):
                for combo in combinations(all_keys, r):
                    yield combo
                    
        elif strategy == 'diverse':
            # Sample diverse combinations focusing on different sizes
            # Single models
            for key in all_keys:
                yield (key,)
            
            # Pairs
            for combo in combinations(all_keys, 2):
                yield combo
            
            # Sample from each size with decreasing density
            for r in range(3, n_models + 1):
                n_combos = len(list(combinations(range(n_models), r)))
                # Sample rate decreases as combination size increases
                sample_rate = min(1.0, 1000 / n_combos)
                
                if sample_rate < 1.0:
                    # Random sampling
                    n_samples = int(n_combos * sample_rate)
                    sampled_indices = np.random.choice(n_combos, n_samples, replace=False)
                    
                    for i, combo in enumerate(combinations(all_keys, r)):
                        if i in sampled_indices:
                            yield combo
                else:
                    # Take all
                    for combo in combinations(all_keys, r):
                        yield combo
                        
        elif strategy == 'size_based':
            # Focus on specific sizes that are likely to perform well
            preferred_sizes = [6, 8, 10, 12, 15, 18, 20, 24]  # Based on ensemble theory
            
            for r in preferred_sizes:
                if r <= n_models:
                    # Sample up to 100 combinations per size
                    combos = list(combinations(all_keys, r))
                    n_samples = min(100, len(combos))
                    sampled_combos = np.random.choice(len(combos), n_samples, replace=False)
                    
                    for idx in sampled_combos:
                        yield combos[idx]
                        
        elif strategy == 'random':
            # Pure random sampling across all sizes
            n_samples