"""
DataFrame Balancer Module for CKD Risk Prediction

This module provides a class for balancing imbalanced datasets in the context of
survival analysis and competing risks using various methods:
- SMOTEENN: SMOTE followed by Edited Nearest Neighbors cleaning
- SMOTETomek: SMOTE followed by Tomek links cleaning
- NearMiss: Undersampling using nearest neighbors

These methods can be used to address class imbalance issues in the CKD risk prediction
project, where the number of patients experiencing dialysis or death may be much smaller
than the number of censored patients.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Literal
from collections import Counter
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
import logging

# Import utility functions
from src.util import (
    load_yaml_file,
    df_event_focus,
    setup_cuda_environment,
    setup_imblearn,
    CUDA_AVAILABLE,
    IMBLEARN_AVAILABLE
)

# Try to import cuML for GPU-accelerated nearest neighbors
try:
    from cuml.neighbors import NearestNeighbors as cuNN
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("cuML library not found. GPU-accelerated nearest neighbors will not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize imblearn and import required classes if available
if setup_imblearn():
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.under_sampling import NearMiss


# Default master dataframe mapping
master_df_mapping_path = Path("src/default_master_df_mapping.yml")
master_df_mapping_config = load_yaml_file(master_df_mapping_path)
DEFAULT_FEATURES = master_df_mapping_config.get('features')
DEFAULT_CLUSTER = master_df_mapping_config.get('cluster')
DEFAULT_DURATION = master_df_mapping_config.get('duration')
DEFAULT_EVENT = master_df_mapping_config.get('event')

class DFBalancer:
    """
    DataFrame Balancer class for handling imbalanced datasets in survival analysis.
    
    This class provides methods for balancing datasets with competing risks,
    where there are multiple event types (e.g., dialysis and death) alongside
    censored observations.
    
    Attributes:
        dataframe (pd.DataFrame): The input DataFrame
        features (List[str]): List of feature column names
        durations (str): Name of the column containing time-to-event data
        events (str): Name of the column containing event indicators
        cluster (str): Name of the column containing cluster information
        balancing_method (str): Method to use for balancing
        model_type (str): Type of model to prepare data for ('deepsurv' or 'deephit')
        event_focus (int): Event value to focus on for DeepSurv (binary outcome)
        sampling_strategy (float or dict): Sampling strategy for balancing
        random_state (int): Random seed for reproducibility
        n_jobs (int): Number of parallel jobs to run
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: Optional[List[str]] = None,
        durations: str = DEFAULT_DURATION,
        events: str = DEFAULT_EVENT,
        cluster: str = DEFAULT_CLUSTER,
        balancing_method: str = 'ENN',
        model_type: Literal['deepsurv', 'deephit'] = 'deephit',
        event_focus: int = 1,
        sampling_strategy: Union[float, Dict, str] = 'auto',
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize the DFBalancer with the given parameters.
        
        Args:
            dataframe: DataFrame containing the data
            features: List of feature column names. If None, uses DEFAULT_FEATURES from config
            durations: Name of the column containing time-to-event data
            events: Name of the column containing event indicators
            cluster: Name of the column containing cluster information
            balancing_method: Method to use for balancing. Options:
                             - 'ENN': SMOTEENN (SMOTE followed by Edited Nearest Neighbors cleaning)
                             - 'Tomek': SMOTETomek (SMOTE followed by Tomek links cleaning)
                             - 'NearMiss': NearMiss undersampling
            model_type: Type of model to prepare data for. Options:
                       - 'deepsurv': Binary outcome model (0=censored, 1=event)
                       - 'deephit': Multi-class outcome model (0=censored, 1=event1, 2=event2, etc.)
            event_focus: Event value to focus on for DeepSurv (only used if model_type='deepsurv')
            sampling_strategy: Strategy for balancing classes. Can be:
                              - float: Ratio of majority to minority class
                              - dict: Specific count for each class
                              - 'auto': Automatic determination
            random_state: Random seed for reproducibility. If None, uses 42
            n_jobs: Number of parallel jobs to run. -1 means use all processors
        """
        self.dataframe = dataframe.copy()
        self.features = features if features is not None else DEFAULT_FEATURES
        self.durations = durations
        self.events = events
        self.cluster = cluster
        self.balancing_method = balancing_method
        self.model_type = model_type
        self.event_focus = event_focus
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state if random_state is not None else 42
        self.n_jobs = n_jobs
        
        # Validate inputs
        self._validate_inputs()
        
        # Log initialization
        logger.info(f"Initialized DFBalancer with {len(self.dataframe)} samples")
        logger.info(f"  Features: {len(self.features)} features")
        logger.info(f"  Duration column: {self.durations}")
        logger.info(f"  Event column: {self.events}")
        logger.info(f"  Cluster column: {self.cluster}")
        logger.info(f"  Balancing method: {self.balancing_method}")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Sampling strategy: {self.sampling_strategy}")
        if self.model_type == 'deepsurv':
            logger.info(f"  Event focus: {self.event_focus}")
        
        # Analyze initial class distribution
        self._analyze_class_distribution()
    
    def _validate_inputs(self):
        """Validate the input parameters."""
        # Check if required columns exist in the dataframe
        missing_features = [f for f in self.features if f not in self.dataframe.columns]
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in dataframe")
        
        if self.durations not in self.dataframe.columns:
            raise ValueError(f"Duration column '{self.durations}' not found in dataframe")
        
        if self.events not in self.dataframe.columns:
            raise ValueError(f"Event column '{self.events}' not found in dataframe")
        
        if self.cluster not in self.dataframe.columns:
            logger.warning(f"Cluster column '{self.cluster}' not found in dataframe. Adding dummy column.")
            self.dataframe[self.cluster] = 'dummy_cluster'
        
        # Check if balancing method is valid
        valid_methods = ['ENN', 'Tomek', 'NearMiss', 'Clustering']
        if self.balancing_method not in valid_methods:
            raise ValueError(f"Invalid balancing method: {self.balancing_method}. Choose from {valid_methods}")
        
        # Check if model type is valid
        valid_model_types = ['deepsurv', 'deephit']
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model type: {self.model_type}. Choose from {valid_model_types}")
        
        # Check if imblearn is available for methods that require it
        if not IMBLEARN_AVAILABLE and self.balancing_method in ['ENN', 'Tomek', 'NearMiss']:
            logger.warning(f"{self.balancing_method} requires imblearn package, which is not available.")
            logger.warning(f"Falling back to simple balancing method.")
    
    def _analyze_class_distribution(self):
        """Analyze the class distribution in the dataset."""
        # Count the number of samples in each class
        class_counts = self.dataframe[self.events].value_counts().sort_index()
        total_samples = len(self.dataframe)
        
        # Calculate class percentages
        class_percentages = (class_counts / total_samples * 100).round(2)
        
        # Log the results
        logger.info(f"Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count} samples ({class_percentages[cls]}%)")
        logger.info(f"Total samples: {total_samples}")
        
        # Calculate imbalance ratio
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            logger.info(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    
    def balance(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Balance the dataset using the specified method.
        
        Returns:
            Tuple containing:
                - numpy array of feature values
                - tuple of (durations, events) numpy arrays after balancing
        """
        logger.info(f"Balancing dataset using {self.balancing_method}...")
        
        if self.balancing_method == 'Clustering':
            balanced_df = self._clustering_balance()
        elif self.balancing_method == 'NearMiss' and IMBLEARN_AVAILABLE:
            balanced_df = self._nearmiss_undersample()
        elif self.balancing_method in ['ENN', 'Tomek'] and IMBLEARN_AVAILABLE:
            balanced_df = self._balance_with_imblearn()
        else:
            # Fallback to simple balancing if method is not available or imblearn is not available
            logger.warning(f"Method {self.balancing_method} not available. Falling back to simple balancing.")
            balanced_df = self._simple_balance()
        
        # For DeepSurv, convert multi-class events to binary
        if self.model_type == 'deepsurv':
            logger.info(f"Converting to binary outcome for DeepSurv (focusing on event {self.event_focus})")
            balanced_df = df_event_focus(balanced_df, self.events, self.event_focus)
        
        # Extract features, durations, and events as numpy arrays
        X = balanced_df[self.features].values
        durations = balanced_df[self.durations].values
        events = balanced_df[self.events].values
        
        # Log final balanced dataset info
        logger.info(f"Final balanced dataset size: {len(balanced_df)} samples")
        self._analyze_final_distribution(balanced_df)
        
        return X, (durations, events)
    
    def _analyze_final_distribution(self, balanced_df):
        """Analyze the class distribution after balancing."""
        # Count the number of samples in each class
        class_counts = balanced_df[self.events].value_counts().sort_index()
        total_samples = len(balanced_df)
        
        # Calculate class percentages
        class_percentages = (class_counts / total_samples * 100).round(2)
        
        # Log the results
        logger.info(f"Class distribution after balancing:")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count} samples ({class_percentages[cls]}%)")
    
    def _simple_balance(self) -> pd.DataFrame:
        """
        Perform simple balancing by resampling to match the majority class size.
        This is a fallback method when imblearn is not available.
        
        Returns:
            Balanced DataFrame
        """
        # Get class counts
        class_counts = self.dataframe[self.events].value_counts()
        max_class_size = class_counts.max()
        max_class = class_counts.idxmax()
        
        logger.info(f"Simple balancing:")
        logger.info(f"  Majority class: {max_class} with {max_class_size} samples")
        
        # Resample each class to match the majority class size
        sampled_dfs = []
        for cls in class_counts.index:
            cls_df = self.dataframe[self.dataframe[self.events] == cls]
            
            if len(cls_df) >= max_class_size:
                # Keep all samples if class size is already larger than or equal to max_class_size
                logger.info(f"  Class {cls}: keeping all {len(cls_df)} samples (no resampling needed)")
                sampled_dfs.append(cls_df)
            else:
                # Resample to max_class_size
                sampled_cls_df = resample(
                    cls_df, 
                    n_samples=max_class_size, 
                    replace=True, 
                    random_state=self.random_state
                )
                logger.info(f"  Class {cls}: resampled from {len(cls_df)} to {max_class_size} samples")
                sampled_dfs.append(sampled_cls_df)
        
        # Combine the sampled dataframes
        balanced_df = pd.concat(sampled_dfs)
        
        # Shuffle the dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return balanced_df
    
    def _nearmiss_undersample(self) -> pd.DataFrame:
        """
        Perform undersampling using the NearMiss algorithm.
        
        Returns:
            Balanced DataFrame
        """
        logger.info(f"NearMiss undersampling:")
        
        # Add original index to track samples
        df_with_index = self.dataframe.copy()
        df_with_index['_original_index'] = df_with_index.index
        
        # For DeepSurv, convert to binary first for balancing
        if self.model_type == 'deepsurv':
            df_temp = df_event_focus(df_with_index, self.events, self.event_focus)
        else:
            df_temp = df_with_index.copy()
        
        # Extract features and target
        X = df_temp.drop(columns=[self.events, self.durations, self.cluster])
        y = df_temp[self.events].astype(int)
        
        # Determine sampling strategy
        if isinstance(self.sampling_strategy, (float, int)) and y.nunique() > 2 and self.model_type == 'deephit':
            # Calculate the sampling strategy dict for multi-class undersampling
            non_major_class_count = y[y != 0].count()  # Sum of all non-major class rows
            majority_class_target = int((1 / self.sampling_strategy) * non_major_class_count)  # Target rows for majority class
            sampling_strategy_dict = {label: count for label, count in y.value_counts().items() if label != 0}
            sampling_strategy_dict[0] = majority_class_target  # Set the majority class target count
            logger.info(f"  Using calculated sampling strategy: {sampling_strategy_dict}")
            sampling_strategy = sampling_strategy_dict
        else:
            sampling_strategy = self.sampling_strategy
            logger.info(f"  Using provided sampling strategy: {sampling_strategy}")
        
        # Create and apply NearMiss
        # Note: NearMiss in newer versions doesn't accept random_state, only n_jobs and version
        nearmiss = NearMiss(
            sampling_strategy=sampling_strategy,
            version=3,  # Default to version 3 as it generally performs best
            n_jobs=self.n_jobs
        )
        
        try:
            X_resampled, y_resampled = nearmiss.fit_resample(X, y)
            
            # Get original indices to preserve all columns
            original_indices = X_resampled['_original_index'].values
            balanced_df = self.dataframe.loc[original_indices].copy()
            
            # Log results
            logger.info(f"  NearMiss undersampling completed successfully")
            logger.info(f"  Original dataset size: {len(self.dataframe)} samples")
            logger.info(f"  Balanced dataset size: {len(balanced_df)} samples")
            
            return balanced_df
            
        except Exception as e:
            logger.error(f"NearMiss undersampling failed: {e}")
            logger.info("Falling back to simple balancing.")
            return self._simple_balance()
    
    def _balance_with_imblearn(self) -> pd.DataFrame:
        """
        Balance the dataset using imblearn's combined methods.
        
        Returns:
            Balanced DataFrame
        """
        if not IMBLEARN_AVAILABLE:
            logger.warning("imblearn package is not available.")
            return self._simple_balance()
            
        logger.info(f"Balancing with {self.balancing_method}:")
        logger.info(f"  Using {len(self.features)} features")
        
        # Extract features and target
        X = self.dataframe[self.features]
        y = self.dataframe[self.events]
        
        # Set up sampling parameters
        params = {
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
        # Select the appropriate sampler
        if self.balancing_method == 'ENN':
            sampler = SMOTEENN(sampling_strategy=params['sampling_strategy'], 
                              random_state=params['random_state'], 
                              n_jobs=params['n_jobs'])
            logger.info(f"  Using SMOTEENN (SMOTE + Edited Nearest Neighbors)")
        elif self.balancing_method == 'Tomek':
            sampler = SMOTETomek(sampling_strategy=params['sampling_strategy'], 
                                random_state=params['random_state'], 
                                n_jobs=params['n_jobs'])
            logger.info(f"  Using SMOTETomek (SMOTE + Tomek links)")
        else:
            raise ValueError(f"Unsupported method: {self.balancing_method}. Use 'ENN', 'Tomek', or 'NearMiss'.")
        
        try:
            # Apply the balancing
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Create a new dataframe with the resampled data
            resampled_df = pd.DataFrame(X_resampled, columns=self.features)
            resampled_df[self.events] = y_resampled
            
            # Add duration column (this will be approximate for synthetic samples)
            if self.durations in self.dataframe.columns:
                # Create a mapping from original samples to their durations
                original_indices = {}
                for i, (_, row) in enumerate(self.dataframe.iterrows()):
                    key = tuple(row[self.features].values)
                    original_indices[key] = i
                
                # For each sample in the resampled data, check if it's original or synthetic
                durations = []
                for i, row in resampled_df.iterrows():
                    key = tuple(row[self.features].values)
                    if key in original_indices:
                        # Original sample, use its original duration
                        original_idx = original_indices[key]
                        durations.append(self.dataframe.iloc[original_idx][self.durations])
                    else:
                        # Synthetic sample, use the average duration of samples with the same class
                        same_class_durations = self.dataframe[self.dataframe[self.events] == row[self.events]][self.durations]
                        durations.append(same_class_durations.mean())
                
                resampled_df[self.durations] = durations
            
            # Add cluster column
            if self.cluster in self.dataframe.columns:
                resampled_df[self.cluster] = 'balanced_data'
            
            # Print class distribution after balancing
            class_counts_after = resampled_df[self.events].value_counts().sort_index()
            for cls, count in class_counts_after.items():
                logger.info(f"  Class {cls} after balancing: {count} samples")
            
            return resampled_df
            
        except Exception as e:
            logger.error(f"Balancing with {self.balancing_method} failed: {e}")
            logger.info("Falling back to simple balancing.")
            return self._simple_balance()
    
    def _define_medoid(self, df, feature_cols, event_col, event_focus=1, n_neighbors=50, model_type='deepsurv'):
        """
        Defines the medoid by calculating the total distance from each point to its neighbors.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            feature_cols (list): Feature columns for clustering.
            event_col (str): Event column name.
            event_focus (int, optional): Event value to focus on. Defaults to 1.
            n_neighbors (int, optional): Number of neighbors for distance calculation. Defaults to 50.
            model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.
            
        Returns:
            tuple: DataFrame containing medoids and DataFrame with remaining data.
        """
        # Separate data into majority and minority classes based on model type
        if model_type == 'deephit':
            df_minor = df[df[event_col] != 0].copy()
            df_major = df[df[event_col] == 0].copy()
        else:
            df2 = df_event_focus(df, event_col=event_col, event_focus=event_focus)
            df_minor = df2[df2[event_col] == 1].copy()
            df_major = df2[df2[event_col] != 1].copy()
        
        # Determine number of clusters (medoids) to extract
        n_clusters = min(len(df_major), len(df_minor))
        df_major_features = df_major[feature_cols].copy()
        
        # Use GPU acceleration if available
        if CUDA_AVAILABLE and setup_cuda_environment():
            try:
                logger.info(f"Using GPU-accelerated nearest neighbors for medoid calculation")
                nn = cuNN(n_neighbors=n_neighbors, algorithm='auto')
                nn.fit(df_major_features)
                distances, _ = nn.kneighbors(df_major_features)
                total_distance = distances.sum(axis=1)
                # Convert from GPU to CPU memory
                total_distance = cp.asnumpy(total_distance) if hasattr(total_distance, 'get') else total_distance
            except Exception as e:
                logger.error(f"GPU acceleration failed: {e}. Falling back to CPU.")
                # CPU fallback
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=n_neighbors)
                nn.fit(df_major_features)
                distances, _ = nn.kneighbors(df_major_features)
                total_distance = distances.sum(axis=1)
        else:
            # CPU implementation
            logger.info(f"Using CPU-based nearest neighbors for medoid calculation")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(df_major_features)
            distances, _ = nn.kneighbors(df_major_features)
            total_distance = distances.sum(axis=1)
        
        # Select points with smallest total distance as medoids
        cluster_center = np.argsort(total_distance)[:n_clusters]
        df_medoid = df_major.iloc[cluster_center].copy()
        
        # Combine medoids with minority class samples
        df_final = pd.concat([df_medoid, df_minor]).sample(frac=1.0, random_state=self.random_state)
        
        # Create remaining dataset (excluding selected medoids)
        df_remain = df.drop(df_major.index[cluster_center])
        
        logger.info(f"Defined {n_clusters} medoids for {model_type} model")
        return df_final, df_remain
    
    def _clustering_balance(self) -> pd.DataFrame:
        """
        Balance the dataset using medoid-based clustering.
        
        This method identifies representative samples (medoids) from the majority class
        and combines them with all samples from the minority classes. The process is
        repeated iteratively to achieve the desired sampling ratio.
        
        Returns:
            Balanced DataFrame
        """
        logger.info(f"Balancing with Clustering method:")
        
        # Make a copy of the dataframe to avoid modifying the original
        remaining_data = self.dataframe.copy()
        
        # For DeepSurv, convert to binary first for initial analysis
        if self.model_type == 'deepsurv':
            df_temp = df_event_focus(remaining_data, self.events, self.event_focus)
            logger.info(f"Converting to binary outcome for initial analysis (focusing on event {self.event_focus})")
        else:
            df_temp = remaining_data.copy()
        
        # Determine sampling strategy
        if isinstance(self.sampling_strategy, (float, int)):
            goal = round(1 / self.sampling_strategy)
            logger.info(f"Using sampling strategy {self.sampling_strategy}, targeting {goal} clustering iterations")
        elif self.sampling_strategy == 'auto':
            # Auto strategy: balance classes to be roughly equal
            class_counts = df_temp[self.events].value_counts()
            if self.model_type == 'deephit':
                # For multi-class, aim for roughly equal distribution
                majority_count = class_counts[0]
                minority_sum = sum(count for cls, count in class_counts.items() if cls != 0)
                goal = round(majority_count / minority_sum)
            else:
                # For binary, aim for 1:1 ratio
                goal = round(class_counts[0] / class_counts[1])
            logger.info(f"Auto sampling strategy: targeting {goal} clustering iterations")
        else:
            # Default to 2 iterations (majority:minority ratio of 1:2)
            goal = 2
            logger.info(f"Using default sampling strategy, targeting {goal} clustering iterations")
        
        # Perform iterative clustering
        all_clusters = []
        for repeat_count in range(goal):
            if len(remaining_data) == 0:
                logger.info(f"No more data remaining after {repeat_count} iterations")
                break
                
            logger.info(f"Performing clustering iteration {repeat_count + 1}/{goal}")
            
            # Extract medoids and update remaining data
            cluster_df, remaining_data = self._define_medoid(
                df=remaining_data,
                feature_cols=self.features,
                event_col=self.events,
                event_focus=self.event_focus,
                n_neighbors=min(50, len(remaining_data) // 10),  # Adaptive neighbor count
                model_type=self.model_type
            )
            
            all_clusters.append(cluster_df)
        
        # Combine all clusters and remove any duplicates
        final_clusters = pd.concat(all_clusters, ignore_index=True)
        final_clusters = final_clusters.drop_duplicates(subset=self.features, keep='first')
        
        logger.info(f"Clustering completed with {len(final_clusters)} samples")
        logger.info(f"  Original dataset size: {len(self.dataframe)} samples")
        logger.info(f"  Balanced dataset size: {len(final_clusters)} samples")
        
        return final_clusters

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create a sample imbalanced dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = np.random.randn(n_samples, 5)
    
    # Create imbalanced classes (0: 70%, 1: 20%, 2: 10%)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Create a dataframe
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['endpoint'] = y
    df['duration'] = np.random.uniform(100, 2000, n_samples)
    df['key'] = [f"patient_{i}" for i in range(n_samples)]
    
    # Test the DFBalancer with different methods and model types
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    # For DeepHit with NearMiss (multi-class)
    print("\nTesting with DeepHit and NearMiss (multi-class):")
    balancer_nearmiss = DFBalancer(
        df,
        features=features,
        balancing_method='NearMiss',
        model_type='deephit',
        sampling_strategy=0.5  # Majority class will be 2x the size of minority classes combined
    )
    X_nearmiss, (durations_nearmiss, events_nearmiss) = balancer_nearmiss.balance()
    
    # For DeepHit with Clustering (multi-class)
    print("\nTesting with DeepHit and Clustering (multi-class):")
    balancer_clustering = DFBalancer(
        df,
        features=features,
        balancing_method='Clustering',
        model_type='deephit',
        sampling_strategy=0.5  # Majority class will be 2x the size of minority classes combined
    )
    X_clustering, (durations_clustering, events_clustering) = balancer_clustering.balance()
    
    # For DeepHit (multi-class)
    print("\nTesting with DeepHit and SMOTEENN (multi-class):")
    balancer_deephit = DFBalancer(
        df, 
        features=features, 
        balancing_method='ENN', 
        model_type='deephit'
    )
    X_deephit, (durations_deephit, events_deephit) = balancer_deephit.balance()
    
    # For DeepSurv (binary, focusing on event 1)
    print("\nTesting with DeepSurv (binary, focusing on event 1):")
    balancer_deepsurv1 = DFBalancer(
        df, 
        features=features, 
        balancing_method='ENN', 
        model_type='deepsurv', 
        event_focus=1
    )
    X_deepsurv1, (durations_deepsurv1, events_deepsurv1) = balancer_deepsurv1.balance()
    
    print("Done!")
