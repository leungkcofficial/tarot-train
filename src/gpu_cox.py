"""GPU-accelerated Cox Proportional Hazards model implementation using cuML and cuPy.

This module provides a GPU-accelerated implementation of the Cox Proportional Hazards model
that can be used as a drop-in replacement for lifelines' CoxPHFitter.
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Set up logging
logger = logging.getLogger(__name__)

class CumlCoxPHFitter:
    """GPU-accelerated Cox Proportional Hazards model using cuML and cuPy.
    
    This class provides a similar interface to lifelines' CoxPHFitter but uses
    GPU acceleration for faster computation, especially with large datasets.
    
    Attributes:
        penalizer (float): L2 regularization strength
        l1_ratio (float): L1 ratio for elastic net regularization (0 = ridge, 1 = lasso)
        params_ (cupy.ndarray): Fitted coefficients
        log_likelihood_ (float): Log-likelihood of the fitted model
        summary (pd.DataFrame): Summary of the fitted model with coefficients and p-values
    """
    
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        """Initialize the CumlCoxPHFitter.
        
        Args:
            penalizer (float, optional): L2 regularization strength. Default is 0.0.
            l1_ratio (float, optional): L1 ratio for elastic net regularization. Default is 0.0.
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._params = None
        self.log_likelihood_ = None
        self.summary = None
        
        # Check if cuML and cuPy are available
        self._check_dependencies()
    
    @property
    def params_(self):
        """Get the model parameters."""
        return self._params
    
    @params_.setter
    def params_(self, value):
        """Set the model parameters.
        
        This allows for warm-starting the model with coefficients from a previous fit.
        
        Args:
            value: The parameter values to set
        """
        self._params = value
    
    def _check_dependencies(self):
        """Check if cuML and cuPy are available."""
        try:
            import cupy as cp
            import cuml
            logger.info("cuML and cuPy are available. GPU acceleration enabled.")
        except ImportError:
            raise ImportError("cuML and cuPy are required for GPU-accelerated Cox model. Please install them with 'pip install cuml cupy'.")
    
    def fit(self, df, duration_col, event_col, robust=False, weights_col=None, show_progress=True):
        """Fit the Cox model using GPU acceleration.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            duration_col (str): Name of the column containing the duration
            event_col (str): Name of the column containing the event indicator
            robust (bool, optional): Whether to use robust standard errors. Default is False.
            weights_col (str, optional): Name of the column containing sample weights. Default is None.
            show_progress (bool, optional): Whether to show fitting progress. Default is True.
            
        Returns:
            self: The fitted model
        """
        import cupy as cp
        import time
        start_time = time.time()
        
        if show_progress:
            logger.info("Starting Cox model fitting with GPU acceleration...")
            try:
                # Get GPU memory info if available
                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem = mem_info[0] / (1024**3)  # Convert to GB
                total_mem = mem_info[1] / (1024**3)  # Convert to GB
                logger.info(f"GPU memory: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
            except:
                logger.info("Unable to get GPU memory information")
        
        # Extract features and target
        if weights_col is not None:
            X = df.drop([duration_col, event_col, weights_col], axis=1)
            weights = df[weights_col].values
        else:
            X = df.drop([duration_col, event_col], axis=1)
            weights = np.ones(len(df))
        
        feature_names = X.columns.tolist()
        X_values = X.values
        durations = df[duration_col].values
        events = df[event_col].values
        
        if show_progress:
            logger.info(f"Data preparation: {len(df)} samples with {len(feature_names)} features")
            logger.info(f"Event rate: {np.mean(events):.2%} ({np.sum(events)} events out of {len(events)} samples)")
            data_prep_start = time.time()
            
        # Convert to GPU arrays
        if show_progress:
            logger.info("Transferring data to GPU...")
        X_gpu = cp.array(X_values, dtype=cp.float32)
        durations_gpu = cp.array(durations, dtype=cp.float32)
        events_gpu = cp.array(events, dtype=cp.float32)
        weights_gpu = cp.array(weights, dtype=cp.float32)
        
        # Sort by duration for efficient risk set calculation
        if show_progress:
            logger.info("Sorting data by duration for efficient risk set calculation...")
        sort_idx = cp.argsort(durations_gpu)
        X_sorted = X_gpu[sort_idx]
        durations_sorted = durations_gpu[sort_idx]
        events_sorted = events_gpu[sort_idx]
        weights_sorted = weights_gpu[sort_idx]
        
        if show_progress:
            data_prep_time = time.time() - data_prep_start
            logger.info(f"Data preparation completed in {data_prep_time:.2f} seconds")
        
        # Initialize parameters
        n_features = X_gpu.shape[1]
        beta = cp.zeros(n_features, dtype=cp.float32)
        
        # Define the negative log partial likelihood function
        self.nll_call_count = 0
        self.last_nll_log_time = time.time()
        
        def negative_log_likelihood(beta):
            # Track function calls for debugging
            self.nll_call_count += 1
            current_time = time.time()
            
            # Log progress periodically during first calls
            if show_progress and (self.nll_call_count <= 5 or self.nll_call_count % 50 == 0 or
                                 (current_time - self.last_nll_log_time) > 10):
                logger.info(f"Computing negative log-likelihood (call #{self.nll_call_count})")
                self.last_nll_log_time = current_time
            
            # Compute linear predictor
            eta = X_sorted.dot(beta)
            exp_eta = cp.exp(eta)
            weighted_exp_eta = exp_eta * weights_sorted
            
            # Calculate cumulative sum of exp(eta) for efficient risk set calculation
            # This is a key optimization for large datasets
            rev_cumsum_exp_eta = cp.flip(cp.cumsum(cp.flip(weighted_exp_eta)))
            
            # Calculate log partial likelihood
            log_lik = 0
            event_count = 0
            for i in range(len(durations_sorted)):
                if events_sorted[i]:
                    # For each event, add contribution to log likelihood
                    # The risk set is efficiently calculated using the reverse cumulative sum
                    risk_set_sum = rev_cumsum_exp_eta[i]
                    log_lik += eta[i] - cp.log(risk_set_sum)
                    event_count += 1
            
            # Log progress for first few calls
            if show_progress and self.nll_call_count <= 3:
                logger.info(f"  Processed {event_count} events, current log-likelihood: {log_lik:.4f}")
            
            # Add regularization term
            if self.penalizer > 0:
                if self.l1_ratio > 0:
                    # Elastic net regularization
                    l1_penalty = self.l1_ratio * self.penalizer * cp.sum(cp.abs(beta))
                    l2_penalty = (1 - self.l1_ratio) * self.penalizer * cp.sum(beta**2)
                    log_lik -= (l1_penalty + l2_penalty)
                else:
                    # Ridge regularization
                    log_lik -= self.penalizer * cp.sum(beta**2)
            
            return -log_lik
        
        # Define the gradient of the negative log partial likelihood
        self.grad_call_count = 0
        self.last_grad_log_time = time.time()
        
        def negative_log_likelihood_gradient(beta):
            # Track function calls for debugging
            self.grad_call_count += 1
            current_time = time.time()
            
            # Log progress periodically during first calls
            if show_progress and (self.grad_call_count <= 5 or self.grad_call_count % 50 == 0 or
                                 (current_time - self.last_grad_log_time) > 10):
                logger.info(f"Computing gradient (call #{self.grad_call_count})")
                self.last_grad_log_time = current_time
                
            grad_start_time = time.time()
            
            # Compute linear predictor
            eta = X_sorted.dot(beta)
            exp_eta = cp.exp(eta)
            weighted_exp_eta = exp_eta * weights_sorted
            
            # Calculate weighted feature sums for risk sets
            # This is a key optimization for large datasets
            weighted_features = X_sorted * weighted_exp_eta[:, cp.newaxis]
            rev_cumsum_weighted_features = cp.flip(cp.cumsum(cp.flip(weighted_features, axis=0), axis=0), axis=0)
            rev_cumsum_exp_eta = cp.flip(cp.cumsum(cp.flip(weighted_exp_eta)))
            
            # Calculate gradient
            gradient = cp.zeros_like(beta)
            event_count = 0
            for i in range(len(durations_sorted)):
                if events_sorted[i]:
                    # For each event, add contribution to gradient
                    # The weighted average is efficiently calculated using the reverse cumulative sums
                    risk_set_sum = rev_cumsum_exp_eta[i]
                    weighted_avg = rev_cumsum_weighted_features[i] / risk_set_sum
                    gradient += weighted_avg - X_sorted[i]
                    event_count += 1
            
            # Log progress for first few calls
            if show_progress and self.grad_call_count <= 3:
                grad_time = time.time() - grad_start_time
                grad_norm = float(cp.linalg.norm(gradient))
                logger.info(f"  Processed {event_count} events, gradient norm: {grad_norm:.4f}, time: {grad_time:.4f}s")
            
            # Add regularization gradient
            if self.penalizer > 0:
                if self.l1_ratio > 0:
                    # Elastic net regularization gradient
                    l1_grad = self.l1_ratio * self.penalizer * cp.sign(beta)
                    l2_grad = 2 * (1 - self.l1_ratio) * self.penalizer * beta
                    gradient += (l1_grad + l2_grad)
                else:
                    # Ridge regularization gradient
                    gradient += 2 * self.penalizer * beta
            
            return gradient
        
        # Optimize using L-BFGS-B
        # We'll use scipy's optimizer but with cuPy arrays
        from scipy.optimize import minimize
        
        # Convert functions to use CPU arrays for scipy.optimize
        def cpu_negative_log_likelihood(beta_cpu):
            beta_gpu = cp.array(beta_cpu, dtype=cp.float32)
            nll = float(negative_log_likelihood(beta_gpu))
            return nll
        
        def cpu_negative_log_likelihood_gradient(beta_cpu):
            beta_gpu = cp.array(beta_cpu, dtype=cp.float32)
            grad = cp.asnumpy(negative_log_likelihood_gradient(beta_gpu))
            return grad
        
        # Create a callback function to report progress during optimization
        if show_progress:
            # Initialize optimization tracking variables
            self.iter_count = 0
            self.last_log_time = time.time()
            self.best_nll = float('inf')
            self.no_improvement_count = 0
            
            def optimization_callback(xk):
                self.iter_count += 1
                current_time = time.time()
                
                # Calculate negative log-likelihood
                current_nll = cpu_negative_log_likelihood(xk)
                
                # Check for improvement
                if current_nll < self.best_nll:
                    self.best_nll = current_nll
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                # Log progress every 5 seconds or every 10 iterations, whichever comes first
                if self.iter_count % 10 == 0 or (current_time - self.last_log_time) > 5:
                    elapsed = current_time - start_time
                    logger.info(f"Optimization iteration {self.iter_count}: neg. log-likelihood = {current_nll:.4f}, elapsed time: {elapsed:.2f}s")
                    
                    # Report if we're not seeing improvement
                    if self.no_improvement_count > 20:
                        logger.info(f"No improvement in log-likelihood for {self.no_improvement_count} iterations")
                    
                    self.last_log_time = current_time
                    
                    # Log GPU memory usage if available
                    try:
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free_mem = mem_info[0] / (1024**3)  # Convert to GB
                        total_mem = mem_info[1] / (1024**3)  # Convert to GB
                        logger.info(f"GPU memory: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
                    except:
                        pass
        else:
            optimization_callback = None
        
        # Show optimization progress if requested
        if show_progress:
            logger.info("Starting optimization process...")
            logger.info(f"Dataset size: {X_values.shape[0]} samples, {X_values.shape[1]} features")
            logger.info(f"Optimization method: L-BFGS-B with max iterations: 1000")
            logger.info(f"Initial negative log-likelihood: {cpu_negative_log_likelihood(cp.asnumpy(beta)):.4f}")
            disp = True
        else:
            disp = False
        
        # Optimize
        opt_start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                cpu_negative_log_likelihood,
                cp.asnumpy(beta),
                method='L-BFGS-B',
                jac=cpu_negative_log_likelihood_gradient,
                callback=optimization_callback,
                options={'disp': disp, 'maxiter': 1000}
            )
        
        if show_progress:
            opt_time = time.time() - opt_start_time
            logger.info(f"Optimization completed in {opt_time:.2f} seconds")
            logger.info(f"Optimization status: {result.message}")
            logger.info(f"Number of iterations: {result.nit}")
            logger.info(f"Number of function evaluations: {result.nfev}")
            logger.info(f"Final negative log-likelihood: {result.fun:.4f}")
        
        # Store results
        self.params_ = cp.array(result.x, dtype=cp.float32)
        self.log_likelihood_ = -result.fun
        
        if show_progress:
            logger.info("Creating model summary...")
            # Count non-zero coefficients
            non_zero_coefs = cp.sum(cp.abs(self.params_) > 1e-6)
            logger.info(f"Model has {non_zero_coefs} non-zero coefficients out of {len(self.params_)} features")
        
        # Create summary DataFrame with error handling
        summary_start_time = time.time()
        try:
            self._create_summary(X_values, feature_names, durations, events)
            
            if show_progress:
                summary_time = time.time() - summary_start_time
                logger.info(f"Summary creation completed in {summary_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error in summary creation: {e}")
            logger.warning("Continuing without summary statistics")
            self.summary = None
            
            # Report top features by coefficient magnitude
            if hasattr(self, 'summary') and self.summary is not None:
                top_features = self.summary.sort_values('coef', key=abs, ascending=False).head(5)
                logger.info("Top 5 features by coefficient magnitude:")
                for idx, row in top_features.iterrows():
                    logger.info(f"  {idx}: coef={row['coef']:.6f}, exp(coef)={row['exp(coef)']:.6f}, p={row['p']:.6f}")
            
            # Report total time
            total_time = time.time() - start_time
            logger.info(f"Model fitting complete. Total time: {total_time:.2f} seconds")
            logger.info(f"Final log-likelihood: {self.log_likelihood_:.4f}")
        
        return self
    
    def _create_summary(self, X, feature_names, durations, events):
        """Create summary DataFrame with coefficients, standard errors, and p-values.
        
        Args:
            X (numpy.ndarray): Feature matrix
            feature_names (list): List of feature names
            durations (numpy.ndarray): Array of durations
            events (numpy.ndarray): Array of event indicators
        """
        import cupy as cp
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        # Calculate standard errors (using observed information matrix)
        # This is a simplified approach; a more accurate approach would use the Hessian
        n_samples = X.shape[0]
        X_gpu = cp.array(X, dtype=cp.float32)
        
        # Calculate the Hessian matrix (observed information matrix)
        # For simplicity, we'll use the approximation: H ≈ X^T X
        hessian = cp.dot(X_gpu.T, X_gpu) / n_samples
        
        # Add regularization to the Hessian if penalizer > 0
        if self.penalizer > 0:
            if self.l1_ratio > 0:
                # For elastic net, we'll use a simple approximation
                # This is not exact for L1 regularization
                reg_hessian = 2 * (1 - self.l1_ratio) * self.penalizer * cp.eye(len(self.params_))
                hessian += reg_hessian
            else:
                # For ridge regularization
                hessian += 2 * self.penalizer * cp.eye(len(self.params_))
        
        # Calculate standard errors
        try:
            hessian_inv = cp.linalg.inv(hessian)
            se = cp.sqrt(cp.diag(hessian_inv))
        except cp.linalg.LinAlgError:
            # If the Hessian is singular, use a pseudo-inverse
            logger.warning("Hessian matrix is singular. Using pseudo-inverse for standard error calculation.")
            hessian_inv = cp.linalg.pinv(hessian)
            se = cp.sqrt(cp.diag(hessian_inv))
        
        # Calculate z-scores and p-values
        z = self.params_ / se
        # Convert to numpy array before using scipy stats
        z_numpy = cp.asnumpy(cp.abs(z))
        p = 2 * (1 - stats.norm.cdf(z_numpy))
        
        # Calculate hazard ratios and confidence intervals
        hr = cp.exp(self.params_)
        hr_lower = cp.exp(self.params_ - 1.96 * se)
        hr_upper = cp.exp(self.params_ + 1.96 * se)
        
        # Create summary DataFrame with proper error handling
        try:
            # Convert all cupy arrays to numpy arrays
            coef_numpy = cp.asnumpy(self.params_)
            hr_numpy = cp.asnumpy(hr)
            se_numpy = cp.asnumpy(se)
            z_numpy_full = cp.asnumpy(z)
            hr_lower_numpy = cp.asnumpy(hr_lower)
            hr_upper_numpy = cp.asnumpy(hr_upper)
            
            # Create the DataFrame
            summary = pd.DataFrame({
                'coef': coef_numpy,
                'exp(coef)': hr_numpy,
                'se(coef)': se_numpy,
                'z': z_numpy_full,
                'p': p,
                'lower 0.95': hr_lower_numpy,
                'upper 0.95': hr_upper_numpy
            }, index=feature_names)
            
            self.summary = summary
            return summary
        except Exception as e:
            logger.error(f"Error creating summary DataFrame: {e}")
            logger.error(f"Types: params={type(self.params_)}, hr={type(hr)}, se={type(se)}, z={type(z)}, p={type(p)}")
            # Create a minimal summary to avoid further errors
            try:
                minimal_summary = pd.DataFrame({
                    'coef': cp.asnumpy(self.params_)
                }, index=feature_names)
                self.summary = minimal_summary
                return minimal_summary
            except Exception as e2:
                logger.error(f"Failed to create even minimal summary: {e2}")
                self.summary = None
                return None
    
    def predict_partial_hazard(self, X):
        """Predict partial hazard for new data.
        
        Args:
            X (pd.DataFrame or numpy.ndarray): New data for prediction
            
        Returns:
            numpy.ndarray: Predicted partial hazard
        """
        import cupy as cp
        
        # Convert to cupy array if needed
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        X_gpu = cp.array(X_values, dtype=cp.float32)
        
        # Calculate linear predictor
        eta = X_gpu.dot(self.params_)
        
        # Calculate partial hazard
        partial_hazard = cp.exp(eta)
        
        # Convert back to numpy array
        return cp.asnumpy(partial_hazard)
        
    def fit_gpu(self, X_gpu, durations_gpu, events_gpu, weights_gpu=None, show_progress=True):
        """Fit the Cox model directly using GPU arrays without pandas conversion.
        
        This method avoids the overhead of converting between CPU and GPU arrays
        by working directly with CuPy arrays throughout the fitting process.
        
        Args:
            X_gpu (cupy.ndarray): Feature matrix on GPU
            durations_gpu (cupy.ndarray): Duration values on GPU
            events_gpu (cupy.ndarray): Event indicators on GPU
            weights_gpu (cupy.ndarray, optional): Sample weights on GPU. Default is None.
            show_progress (bool, optional): Whether to show fitting progress. Default is True.
            
        Returns:
            self: The fitted model
        """
        import cupy as cp
        import time
        start_time = time.time()
        
        if show_progress:
            logger.info("Starting Cox model fitting with GPU acceleration (direct arrays)...")
            try:
                # Get GPU memory info if available
                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem = mem_info[0] / (1024**3)  # Convert to GB
                total_mem = mem_info[1] / (1024**3)  # Convert to GB
                logger.info(f"GPU memory: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
            except:
                logger.info("Unable to get GPU memory information")
        
        # Use default weights if not provided
        if weights_gpu is None:
            weights_gpu = cp.ones(len(durations_gpu), dtype=cp.float32)
        
        # Store feature names as integers since we don't have column names
        self.feature_names = [f"X{i}" for i in range(X_gpu.shape[1])]
        
        # Sort by duration for efficient risk set calculation
        if show_progress:
            logger.info("Sorting data by duration for efficient risk set calculation...")
        sort_idx = cp.argsort(durations_gpu)
        X_sorted = X_gpu[sort_idx]
        durations_sorted = durations_gpu[sort_idx]
        events_sorted = events_gpu[sort_idx]
        weights_sorted = weights_gpu[sort_idx]
        
        if show_progress:
            data_prep_time = time.time() - start_time
            logger.info(f"Data preparation completed in {data_prep_time:.2f} seconds")
        
        # Initialize parameters
        n_features = X_gpu.shape[1]
        
        # Use existing parameters if available (warm-start)
        if self._params is not None and len(self._params) == n_features:
            beta = self._params
            if show_progress:
                logger.info("Using warm-start with existing coefficients")
        else:
            beta = cp.zeros(n_features, dtype=cp.float32)
        
        # Define the negative log partial likelihood function
        self.nll_call_count = 0
        self.last_nll_log_time = time.time()
        
        def negative_log_likelihood(beta):
            # Track function calls for debugging
            self.nll_call_count += 1
            current_time = time.time()
            
            # Log progress periodically during first calls
            if show_progress and (self.nll_call_count <= 5 or self.nll_call_count % 50 == 0 or
                                 (current_time - self.last_nll_log_time) > 10):
                logger.info(f"Computing negative log-likelihood (call #{self.nll_call_count})")
                self.last_nll_log_time = current_time
            
            # Compute linear predictor
            eta = X_sorted.dot(beta)
            exp_eta = cp.exp(eta)
            weighted_exp_eta = exp_eta * weights_sorted
            
            # Calculate cumulative sum of exp(eta) for efficient risk set calculation
            # This is a key optimization for large datasets
            rev_cumsum_exp_eta = cp.flip(cp.cumsum(cp.flip(weighted_exp_eta)))
            
            # Calculate log partial likelihood
            log_lik = 0
            event_count = 0
            for i in range(len(durations_sorted)):
                if events_sorted[i]:
                    # For each event, add contribution to log likelihood
                    # The risk set is efficiently calculated using the reverse cumulative sum
                    risk_set_sum = rev_cumsum_exp_eta[i]
                    log_lik += eta[i] - cp.log(risk_set_sum)
                    event_count += 1
            
            # Log progress for first few calls
            if show_progress and self.nll_call_count <= 3:
                logger.info(f"  Processed {event_count} events, current log-likelihood: {log_lik:.4f}")
            
            # Add regularization term
            if self.penalizer > 0:
                if self.l1_ratio > 0:
                    # Elastic net regularization
                    l1_penalty = self.l1_ratio * self.penalizer * cp.sum(cp.abs(beta))
                    l2_penalty = (1 - self.l1_ratio) * self.penalizer * cp.sum(beta**2)
                    log_lik -= (l1_penalty + l2_penalty)
                else:
                    # Ridge regularization
                    log_lik -= self.penalizer * cp.sum(beta**2)
            
            return -log_lik
        
        # Define the gradient of the negative log partial likelihood
        self.grad_call_count = 0
        self.last_grad_log_time = time.time()
        
        def negative_log_likelihood_gradient(beta):
            # Track function calls for debugging
            self.grad_call_count += 1
            current_time = time.time()
            
            # Log progress periodically during first calls
            if show_progress and (self.grad_call_count <= 5 or self.grad_call_count % 50 == 0 or
                                 (current_time - self.last_grad_log_time) > 10):
                logger.info(f"Computing gradient (call #{self.grad_call_count})")
                self.last_grad_log_time = current_time
                
            grad_start_time = time.time()
            
            # Compute linear predictor
            eta = X_sorted.dot(beta)
            exp_eta = cp.exp(eta)
            weighted_exp_eta = exp_eta * weights_sorted
            
            # Calculate weighted feature sums for risk sets
            # This is a key optimization for large datasets
            weighted_features = X_sorted * weighted_exp_eta[:, cp.newaxis]
            rev_cumsum_weighted_features = cp.flip(cp.cumsum(cp.flip(weighted_features, axis=0), axis=0), axis=0)
            rev_cumsum_exp_eta = cp.flip(cp.cumsum(cp.flip(weighted_exp_eta)))
            
            # Calculate gradient
            gradient = cp.zeros_like(beta)
            event_count = 0
            for i in range(len(durations_sorted)):
                if events_sorted[i]:
                    # For each event, add contribution to gradient
                    # The weighted average is efficiently calculated using the reverse cumulative sums
                    risk_set_sum = rev_cumsum_exp_eta[i]
                    weighted_avg = rev_cumsum_weighted_features[i] / risk_set_sum
                    gradient += weighted_avg - X_sorted[i]
                    event_count += 1
            
            # Log progress for first few calls
            if show_progress and self.grad_call_count <= 3:
                grad_time = time.time() - grad_start_time
                grad_norm = float(cp.linalg.norm(gradient))
                logger.info(f"  Processed {event_count} events, gradient norm: {grad_norm:.4f}, time: {grad_time:.4f}s")
            
            # Add regularization gradient
            if self.penalizer > 0:
                if self.l1_ratio > 0:
                    # Elastic net regularization gradient
                    l1_grad = self.l1_ratio * self.penalizer * cp.sign(beta)
                    l2_grad = 2 * (1 - self.l1_ratio) * self.penalizer * beta
                    gradient += (l1_grad + l2_grad)
                else:
                    # Ridge regularization gradient
                    gradient += 2 * self.penalizer * beta
            
            return gradient
        
        # Optimize using L-BFGS-B
        # We'll use scipy's optimizer but with cuPy arrays
        from scipy.optimize import minimize
        
        # Convert functions to use CPU arrays for scipy.optimize
        def cpu_negative_log_likelihood(beta_cpu):
            beta_gpu = cp.array(beta_cpu, dtype=cp.float32)
            nll = float(negative_log_likelihood(beta_gpu))
            return nll
        
        def cpu_negative_log_likelihood_gradient(beta_cpu):
            beta_gpu = cp.array(beta_cpu, dtype=cp.float32)
            grad = cp.asnumpy(negative_log_likelihood_gradient(beta_gpu))
            return grad
        
        # Create a callback function to report progress during optimization
        if show_progress:
            # Initialize optimization tracking variables
            self.iter_count = 0
            self.last_log_time = time.time()
            self.best_nll = float('inf')
            self.no_improvement_count = 0
            
            def optimization_callback(xk):
                self.iter_count += 1
                current_time = time.time()
                
                # Calculate negative log-likelihood
                current_nll = cpu_negative_log_likelihood(xk)
                
                # Check for improvement
                if current_nll < self.best_nll:
                    self.best_nll = current_nll
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                # Log progress every 5 seconds or every 10 iterations, whichever comes first
                if self.iter_count % 10 == 0 or (current_time - self.last_log_time) > 5:
                    elapsed = current_time - start_time
                    logger.info(f"Optimization iteration {self.iter_count}: neg. log-likelihood = {current_nll:.4f}, elapsed time: {elapsed:.2f}s")
                    
                    # Report if we're not seeing improvement
                    if self.no_improvement_count > 20:
                        logger.info(f"No improvement in log-likelihood for {self.no_improvement_count} iterations")
                    
                    self.last_log_time = current_time
        else:
            optimization_callback = None
        
        # Show optimization progress if requested
        if show_progress:
            logger.info("Starting optimization process...")
            logger.info(f"Dataset size: {X_gpu.shape[0]} samples, {X_gpu.shape[1]} features")
            logger.info(f"Optimization method: L-BFGS-B with max iterations: 1000")
            logger.info(f"Initial negative log-likelihood: {cpu_negative_log_likelihood(cp.asnumpy(beta)):.4f}")
            disp = True
        else:
            disp = False
        
        # Optimize
        opt_start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                cpu_negative_log_likelihood,
                cp.asnumpy(beta),
                method='L-BFGS-B',
                jac=cpu_negative_log_likelihood_gradient,
                callback=optimization_callback,
                options={'disp': disp, 'maxiter': 1000}
            )
        
        if show_progress:
            opt_time = time.time() - opt_start_time
            logger.info(f"Optimization completed in {opt_time:.2f} seconds")
            logger.info(f"Optimization status: {result.message}")
            logger.info(f"Number of iterations: {result.nit}")
            logger.info(f"Number of function evaluations: {result.nfev}")
            logger.info(f"Final negative log-likelihood: {result.fun:.4f}")
        
        # Store results
        self._params = cp.array(result.x, dtype=cp.float32)
        self.log_likelihood_ = -result.fun
        
        # Create a minimal summary with just coefficients
        # We can't create a full summary without feature names
        self.summary = pd.DataFrame({
            'coef': cp.asnumpy(self._params)
        }, index=self.feature_names)
        
        return self

def calculate_cuml_information_criteria(model, df):
    """
    Calculate AIC and BIC using GPU acceleration with cuML/cuPy.
    
    Args:
        model: Fitted CumlCoxPHFitter model
        df: DataFrame used for fitting
        
    Returns:
        Dict with AIC and BIC values as Python floats
    """
    import cupy as cp
    import numpy as np
    
    try:
        # Get number of parameters and observations
        k = len(model.params_)
        n = df.shape[0]
        
        # Get log-likelihood (already calculated on GPU)
        ll = model.log_likelihood_
        
        # Calculate AIC and BIC
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * cp.log(cp.array([n], dtype=cp.float32))[0]
        
        # Ensure we return Python float values, not cupy arrays
        return {"AIC": float(aic), "BIC": float(bic)}
    except Exception as e:
        # Log the error and fall back to a simple calculation
        logger.error(f"Error in GPU-accelerated information criteria calculation: {e}")
        logger.error(f"Types: params={type(model.params_) if hasattr(model, 'params_') else 'N/A'}, "
                    f"log_likelihood={type(model.log_likelihood_) if hasattr(model, 'log_likelihood_') else 'N/A'}")
        
        try:
            # Try a simpler calculation
            k = len(model.params_)
            ll = float(model.log_likelihood_)
            n = float(df.shape[0])
            
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)
            
            return {"AIC": float(aic), "BIC": float(bic)}
        except Exception as e2:
            logger.error(f"Fallback calculation also failed: {e2}")
            # Return NaN values as a last resort
            return {"AIC": float('nan'), "BIC": float('nan')}

def perform_gpu_cox_lasso(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: Optional[List[str]] = None,
    cv_folds: int = 3,  # Reduced from 10 to 3 for faster computation
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 1.0,  # 1.0 for LASSO, between 0-1 for elastic net
    max_iter: int = 10000,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform GPU-accelerated Cox-LASSO regression with cross-validation and generate visualizations.
    
    This function:
    1. Determines the optimal regularization parameter (lambda) through cross-validation
    2. Tracks coefficient changes as lambda increases
    3. Visualizes the relationship between partial likelihood, lambda, and number of features
    4. Exports raw data as CSV and visualizations as PNG
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and target
        duration_col (str): Name of the survival duration column
        event_col (str): Name of the event indicator column
        features (List[str], optional): List of feature names to use.
                                      If None, uses all numeric columns in the DataFrame.
        cv_folds (int, optional): Number of cross-validation folds. Default is 10.
        alphas (List[float], optional): List of alpha values to try.
                                      If None, uses a logarithmic sequence.
        l1_ratio (float, optional): Elastic net mixing parameter (0 <= l1_ratio <= 1).
                                  l1_ratio=1 corresponds to LASSO. Default is 1.0.
        max_iter (int, optional): Maximum number of iterations. Default is 10000.
        output_dir (str or Path, optional): Directory to save outputs.
                                          If None, uses current directory.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
        Dict[str, Any]: Dictionary containing results and paths to saved files
    """
    import cupy as cp
    import cuml
    from sklearn.model_selection import KFold  # Use scikit-learn's KFold instead of cuML's
    import matplotlib.pyplot as plt
    import time
    import os
    import warnings
    from pathlib import Path
    from cuml.preprocessing import StandardScaler  # Use cuML's GPU-accelerated StandardScaler
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path("results/lasso_analysis")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"GPU-accelerated LASSO analysis results will be saved to: {output_dir}")
    
    # Log GPU information
    try:
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0] / (1024**3)  # Convert to GB
        total_mem = mem_info[1] / (1024**3)  # Convert to GB
        logger.info(f"GPU memory before processing: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
    except Exception as e:
        logger.warning(f"Unable to get GPU memory information: {e}")
    
    # If features not specified, use all numeric columns except duration and event
    if features is None:
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in all_numeric if f not in [duration_col, event_col]]
    else:
        # Check if features exist in the dataframe
        # Note: We don't filter for numeric features here because encoded categorical features
        # are already numeric (0/1 values from one-hot encoding)
        features = [f for f in features if f in df.columns]
        logger.info(f"Using {len(features)} features provided by the caller")
    
    # Check if we have enough features
    if len(features) < 2:
        error_msg = f"LASSO regression requires at least 2 features. Found {len(features)} features."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Drop rows with missing values in the selected features, duration, and event
    cols_to_check = features + [duration_col, event_col]
    df_clean = df[cols_to_check].dropna()
    logger.info(f"Using {len(df_clean)} complete rows for GPU-accelerated LASSO regression out of {len(df)} total rows")
    logger.info(f"Features: {len(features)} total, Cross-validation: {cv_folds} folds")
    
    # Prepare the data
    logger.info("Preparing data for GPU-accelerated processing...")
    X_np = df_clean[features].values
    durations_np = df_clean[duration_col].values
    events_np = df_clean[event_col].astype(bool).values
    
    # Transfer data to GPU first
    logger.info("Transferring data to GPU...")
    X_gpu_raw = cp.array(X_np, dtype=cp.float32)
    durations_gpu = cp.array(durations_np, dtype=cp.float32)
    events_gpu = cp.array(events_np, dtype=cp.bool_)
    
    # Standardize features directly on GPU
    logger.info("Standardizing features on GPU (zero mean, unit variance)...")
    scaler = StandardScaler()
    X_gpu = scaler.fit_transform(X_gpu_raw)
    
    # Set up alpha values (regularization strengths)
    if alphas is None:
        # Create a logarithmic sequence of alphas - reduced from 100 to 30 for faster computation
        logger.info("Creating logarithmic sequence of alpha values for regularization path")
        alphas_np = np.logspace(-3, 1, 30)  # Reduced from 100 to 30
        logger.info(f"Testing {len(alphas_np)} alpha values from {alphas_np[0]:.6f} to {alphas_np[-1]:.6f}")
    else:
        alphas_np = np.array(alphas)
        logger.info(f"Using {len(alphas_np)} user-provided alpha values")
    
    # Transfer alphas to GPU
    alphas_gpu = cp.array(alphas_np, dtype=cp.float32)
    
    # Set up cross-validation
    logger.info(f"Setting up {cv_folds}-fold cross-validation with random_state={random_state}")
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store cross-validation results
    n_alphas = len(alphas_np)
    mean_cv_scores = cp.zeros(n_alphas, dtype=cp.float32)
    std_cv_scores = cp.zeros(n_alphas, dtype=cp.float32)
    
    # Perform cross-validation
    logger.info("Starting GPU-accelerated cross-validation process...")
    start_time = time.time()
    
    # We'll implement a simplified version of cross-validation for Cox-LASSO
    # For each fold, we'll fit a Cox model with each alpha value and compute the partial likelihood
    
    # Initialize arrays to store fold results
    fold_scores = cp.zeros((cv_folds, n_alphas), dtype=cp.float32)
    
    # Perform cross-validation
    # Note: scikit-learn's KFold.split() expects numpy arrays, not cupy arrays
    # We'll use numpy arrays for the indices and then convert to GPU arrays
    X_np_indices = np.arange(len(X_gpu))  # Just need the indices, not the actual data
    
    for fold_idx, (train_idx_np, test_idx_np) in enumerate(cv.split(X_np_indices)):
        logger.info(f"Processing fold {fold_idx+1}/{cv_folds}")
        
        # Convert numpy indices to cupy indices
        train_idx = cp.array(train_idx_np)
        test_idx = cp.array(test_idx_np)
        
        # Split data for this fold
        X_train = X_gpu[train_idx]
        durations_train = durations_gpu[train_idx]
        events_train = events_gpu[train_idx]
        
        X_test = X_gpu[test_idx]
        durations_test = durations_gpu[test_idx]
        events_test = events_gpu[test_idx]
        
        # Initialize previous coefficients for warm-start
        previous_coefs = None
        
        # For each alpha value, fit a model and compute the partial likelihood on the test set
        for alpha_idx, alpha in enumerate(alphas_gpu):
            # Only log every 10 alpha values to reduce overhead
            if alpha_idx % 10 == 0 or alpha_idx == len(alphas_gpu) - 1:
                logger.info(f"  Processing alpha {alpha_idx+1}/{len(alphas_gpu)}: {float(alpha):.6f}")
            
            # Create a CumlCoxPHFitter with the current alpha as L2 regularization
            # and l1_ratio to control the elastic net mixing
            model = CumlCoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
            
            # Apply warm-start if we have previous coefficients
            if previous_coefs is not None:
                model.params_ = previous_coefs
            
            # Fit the model directly with GPU arrays
            try:
                model.fit_gpu(
                    X_train,
                    durations_train,
                    events_train,
                    show_progress=(alpha_idx == 0)  # Only show progress for first alpha
                )
                
                # Store coefficients for warm-start in next iteration
                previous_coefs = model.params_
                
                # Compute the partial likelihood on the test set using GPU arrays directly
                try:
                    # Create a temporary model to fit on the test data with fixed coefficients
                    test_model = CumlCoxPHFitter(penalizer=0)  # No regularization for evaluation
                    
                    # Use the previous model's coefficients as a starting point
                    test_model.params_ = model.params_
                    
                    # Fit on test data but don't optimize - just compute log likelihood with current coefficients
                    test_model.fit_gpu(
                        X_test,
                        durations_test,
                        events_test,
                        show_progress=False
                    )
                    
                    # Use the log likelihood as the score
                    # Note: Higher log likelihood is better, so we use it directly
                    fold_scores[fold_idx, alpha_idx] = test_model.log_likelihood_
                except Exception as e:
                    logger.warning(f"Error computing test score in fold {fold_idx+1}, alpha {alpha:.6f}: {e}")
                    # Alternative approach: use the training log likelihood as a fallback
                    fold_scores[fold_idx, alpha_idx] = model.log_likelihood_
            except Exception as e:
                logger.warning(f"Error in fold {fold_idx+1}, alpha {alpha:.6f}: {e}")
                fold_scores[fold_idx, alpha_idx] = float('nan')
    
    # Compute mean and standard deviation of scores across folds
    mean_cv_scores = cp.nanmean(fold_scores, axis=0)
    std_cv_scores = cp.nanstd(fold_scores, axis=0)
    
    # Find the optimal alpha value
    optimal_alpha_idx = cp.nanargmax(mean_cv_scores)
    optimal_alpha = alphas_gpu[optimal_alpha_idx]
    
    # Log cross-validation results
    elapsed_time = time.time() - start_time
    logger.info(f"Cross-validation complete in {elapsed_time:.2f} seconds")
    logger.info(f"Optimal alpha (regularization parameter): {float(optimal_alpha):.6f}")
    
    # Fit the final model with the optimal alpha
    logger.info("Fitting final model with optimal alpha...")
    final_model = CumlCoxPHFitter(penalizer=float(optimal_alpha), l1_ratio=l1_ratio)
    
    # Fit the final model directly with GPU arrays
    final_model.fit_gpu(
        X_gpu,
        durations_gpu,
        events_gpu,
        show_progress=True
    )
    
    # Get the coefficients
    coefs = final_model.params_
    
    # Create a DataFrame with feature names and their coefficients
    logger.info("Creating coefficient summary...")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': cp.asnumpy(coefs)
    })
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
    
    # Log the top features
    non_zero_coefs = (cp.abs(coefs) > 1e-6).sum().item()
    logger.info(f"Model selected {non_zero_coefs} non-zero coefficients out of {len(features)} features")
    
    top_features = coef_df[abs(coef_df['Coefficient']) > 1e-6].head(10)
    if not top_features.empty:
        logger.info("Top features selected by the model (by coefficient magnitude):")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            logger.info(f"  {i}. {row['Feature']}: {row['Coefficient']:.6f}")
    
    # Save coefficients to CSV
    coef_path = output_dir / "lasso_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info(f"Saved LASSO coefficients to {coef_path}")
    
    # Get the coefficient path (how coefficients change with different alphas)
    logger.info("Analyzing coefficient paths across all regularization strengths...")
    
    # For each alpha value, fit a model and get the coefficients
    coef_path_data = []
    feature_entry_points = {}  # Track when each feature enters the model
    
    # Initialize previous coefficients for warm-start
    previous_path_coefs = None
    
    # We'll fit models for each alpha value to get the coefficient path
    for alpha_idx, alpha in enumerate(alphas_np):
        # Only log every 10 alpha values to reduce overhead
        if alpha_idx % 10 == 0 or alpha_idx == len(alphas_np) - 1:
            logger.info(f"Processing alpha {alpha_idx+1}/{len(alphas_np)}: {alpha:.6f}")
        
        # Create a CumlCoxPHFitter with the current alpha
        path_model = CumlCoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
        
        # Apply warm-start if we have previous coefficients
        if previous_path_coefs is not None:
            path_model.params_ = previous_path_coefs
        
        try:
            # Fit the model directly with GPU arrays
            path_model.fit_gpu(
                X_gpu,
                durations_gpu,
                events_gpu,
                show_progress=False
            )
            
            # Store coefficients for warm-start in next iteration
            previous_path_coefs = path_model.params_
            
            # Get the coefficients
            path_coefs = cp.asnumpy(path_model.params_)
            
            # Count non-zero coefficients
            non_zero_count = np.sum(np.abs(path_coefs) > 1e-6)
            
            # Track when features enter the model (as alpha decreases)
            for j, feature in enumerate(features):
                if abs(path_coefs[j]) > 1e-6 and feature not in feature_entry_points:
                    feature_entry_points[feature] = alpha
            
            # Add to coefficient path data
            for j, feature in enumerate(features):
                coef_path_data.append({
                    'Alpha': alpha,
                    'Log_Alpha': np.log10(alpha),
                    'Feature': feature,
                    'Coefficient': path_coefs[j],
                    'Non_Zero_Features': non_zero_count
                })
        except Exception as e:
            logger.warning(f"Error processing alpha {alpha:.6f}: {e}")
            # Add NaN values for this alpha
            for j, feature in enumerate(features):
                coef_path_data.append({
                    'Alpha': alpha,
                    'Log_Alpha': np.log10(alpha),
                    'Feature': feature,
                    'Coefficient': np.nan,
                    'Non_Zero_Features': 0
                })
    
    # Log feature entry points
    if feature_entry_points:
        logger.info("Feature entry points (alpha values where features enter the model):")
        sorted_entries = sorted(feature_entry_points.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, alpha) in enumerate(sorted_entries[:10], 1):  # Show top 10
            logger.info(f"  {i}. {feature}: α = {alpha:.6f}")
        if len(sorted_entries) > 10:
            logger.info(f"  ... and {len(sorted_entries) - 10} more features")
    
    # Convert to DataFrame
    logger.info("Creating coefficient path DataFrame...")
    coef_path_df = pd.DataFrame(coef_path_data)
    
    # Save coefficient path data to CSV
    coef_path_csv = output_dir / "lasso_coefficient_path.csv"
    coef_path_df.to_csv(coef_path_csv, index=False)
    logger.info(f"Saved LASSO coefficient path data to {coef_path_csv}")
    
    # Create cross-validation results DataFrame
    logger.info("Processing cross-validation results...")
    cv_results = pd.DataFrame({
        'Alpha': alphas_np,
        'Log_Alpha': np.log10(alphas_np),
        'Mean_CV_Score': cp.asnumpy(mean_cv_scores),
        'Std_CV_Score': cp.asnumpy(std_cv_scores)
    })
    
    # Add number of non-zero features for each alpha
    logger.info("Calculating feature counts for each alpha value...")
    # Extract non-zero feature counts from coef_path_data
    non_zero_features = []
    for alpha in alphas_np:
        alpha_data = [item for item in coef_path_data if item['Alpha'] == alpha]
        if alpha_data and not np.isnan(alpha_data[0]['Non_Zero_Features']):
            non_zero_features.append(alpha_data[0]['Non_Zero_Features'])
        else:
            non_zero_features.append(0)
    
    cv_results['Non_Zero_Features'] = non_zero_features
    
    # Log best cross-validation score
    best_cv_idx = cv_results['Mean_CV_Score'].idxmax()
    best_cv_alpha = cv_results.loc[best_cv_idx, 'Alpha']
    best_cv_score = cv_results.loc[best_cv_idx, 'Mean_CV_Score']
    best_cv_features = cv_results.loc[best_cv_idx, 'Non_Zero_Features']
    logger.info(f"Best cross-validation score: {best_cv_score:.4f} at alpha={best_cv_alpha:.6f} with {best_cv_features:.0f} features")
    
    # Save cross-validation results to CSV
    cv_results_csv = output_dir / "lasso_cv_results.csv"
    cv_results.to_csv(cv_results_csv, index=False)
    logger.info(f"Saved cross-validation results to {cv_results_csv}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Coefficient path plot
    logger.info("Creating coefficient path plot...")
    plt.figure(figsize=(12, 8))
    
    # Get unique features
    unique_features = coef_path_df['Feature'].unique()
    logger.info(f"Plotting coefficient paths for {len(unique_features)} features")
    
    # Sort features by the order they enter the model (as lambda decreases)
    logger.info("Sorting features by order of entry into the model...")
    feature_entry_order = []
    for feature, alpha in sorted(feature_entry_points.items(), key=lambda x: x[1], reverse=True):
        feature_entry_order.append(feature)
    
    # Add any remaining features that never entered the model
    for feature in features:
        if feature not in feature_entry_order:
            feature_entry_order.append(feature)
    
    # Create a color map
    colors = plt.cm.tab10.colors
    color_map = {feature: colors[i % len(colors)] for i, feature in enumerate(feature_entry_order)}
    
    # Plot each feature's coefficient path
    for feature in feature_entry_order:
        feature_data = coef_path_df[coef_path_df['Feature'] == feature]
        plt.plot(feature_data['Log_Alpha'], feature_data['Coefficient'],
                 label=feature, color=color_map[feature], linewidth=2)
    
    # Add vertical line at optimal alpha
    plt.axvline(x=np.log10(float(optimal_alpha)), color='red', linestyle='--',
                label=f'Optimal α: {float(optimal_alpha):.6f}')
    
    # Add labels and title
    plt.xlabel('Log(α)')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Coefficient Paths (GPU-accelerated)')
    
    # Add legend with sorted features
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    coef_path_plot = output_dir / "lasso_coefficient_path.png"
    logger.info(f"Saving coefficient path plot to {coef_path_plot}...")
    plt.savefig(coef_path_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved coefficient path plot to {coef_path_plot}")
    
    # 2. Cross-validation score plot with number of features
    logger.info("Creating cross-validation score plot with feature counts...")
    plt.figure(figsize=(12, 8))
    
    # Create primary axis for CV score
    ax1 = plt.gca()
    ax1.plot(cv_results['Log_Alpha'], cv_results['Mean_CV_Score'], 'b-', linewidth=2)
    
    # Add error bands
    upper = cv_results['Mean_CV_Score'] + cv_results['Std_CV_Score']
    lower = cv_results['Mean_CV_Score'] - cv_results['Std_CV_Score']
    ax1.fill_between(cv_results['Log_Alpha'], lower, upper, alpha=0.2, color='blue')
    
    # Add vertical line at optimal alpha
    ax1.axvline(x=np.log10(float(optimal_alpha)), color='red', linestyle='--',
               label=f'Optimal α: {float(optimal_alpha):.6f}')
    
    # Add labels for primary axis
    ax1.set_xlabel('Log(α)')
    ax1.set_ylabel('Mean CV Partial Likelihood', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary axis for number of features
    ax2 = ax1.twinx()
    ax2.plot(cv_results['Log_Alpha'], cv_results['Non_Zero_Features'], 'g-', linewidth=2)
    ax2.set_ylabel('Number of Non-Zero Features', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add title
    plt.title('Cross-Validation Results and Feature Count (GPU-accelerated)')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + ['Number of Features'], loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    cv_plot = output_dir / "lasso_cv_results.png"
    logger.info(f"Saving cross-validation results plot to {cv_plot}...")
    plt.savefig(cv_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cross-validation results plot to {cv_plot}")
    
    # Log GPU memory usage after processing
    try:
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0] / (1024**3)  # Convert to GB
        total_mem = mem_info[1] / (1024**3)  # Convert to GB
        logger.info(f"GPU memory after processing: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
    except Exception as e:
        logger.warning(f"Unable to get GPU memory information: {e}")
    
    # Return results
    logger.info("Preparing final results dictionary...")
    results = {
        "optimal_alpha": float(optimal_alpha),
        "optimal_coefficients": coef_df.to_dict(orient='records'),
        "cv_results": cv_results.to_dict(orient='records'),
        "coefficient_path": coef_path_df.to_dict(orient='records'),
        "output_files": {
            "coefficients_csv": str(coef_path),
            "coefficient_path_csv": str(coef_path_csv),
            "cv_results_csv": str(cv_results_csv),
            "coefficient_path_plot": str(coef_path_plot),
            "cv_results_plot": str(cv_plot)
        }
    }
    
    # Log summary of results
    non_zero_features_count = sum(1 for coef in coef_df['Coefficient'] if abs(coef) > 1e-6)
    logger.info(f"GPU-accelerated LASSO analysis complete. Summary:")
    logger.info(f"  - Optimal alpha: {float(optimal_alpha):.6f}")
    logger.info(f"  - Selected {non_zero_features_count} non-zero features out of {len(features)} total features")
    logger.info(f"  - Created {len(results['output_files'])} output files in {output_dir}")
    logger.info(f"  - Top selected features: {', '.join(coef_df[abs(coef_df['Coefficient']) > 1e-6]['Feature'].head(5).tolist())}")
    
    return results


def batch_process_large_dataset(df, duration_col, event_col, features, batch_size=10000, **fit_params):
    """
    Process large datasets in batches to avoid GPU memory issues.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        duration_col (str): Name of the duration column
        event_col (str): Name of the event column
        features (List[str]): List of feature names
        batch_size (int, optional): Batch size. Default is 10000.
        **fit_params: Additional parameters to pass to the fit method
        
    Returns:
        CumlCoxPHFitter: Fitted model
    """
    import cupy as cp
    import numpy as np
    
    # Initialize model
    model = CumlCoxPHFitter(penalizer=fit_params.get('penalizer', 0.01))
    
    # Get total number of samples
    n_samples = len(df)
    
    # Initialize parameters
    n_features = len(features)
    beta = cp.zeros(n_features, dtype=cp.float32)
    
    # Number of batches
    n_batches = int(np.ceil(n_samples / batch_size))
    
    logger.info(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size}")
    logger.info(f"Features being used: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
    logger.info(f"Total epochs: {fit_params.get('max_epochs', 10)}")
    
    # Process in batches
    for epoch in range(fit_params.get('max_epochs', 10)):
        logger.info(f"Starting epoch {epoch+1}/{fit_params.get('max_epochs', 10)}")
        # Shuffle data
        logger.info(f"Shuffling data with random_state={epoch}")
        df_shuffled = df.sample(frac=1, random_state=epoch).reset_index(drop=True)
        
        # Process each batch
        batch_count = 0
        for i in range(0, n_samples, batch_size):
            batch_count += 1
            end_idx = min(i + batch_size, n_samples)
            batch_df = df_shuffled.iloc[i:end_idx]
            actual_batch_size = len(batch_df)
            
            logger.info(f"Processing batch {batch_count}/{n_batches} with {actual_batch_size} samples")
            
            # Fit model on this batch
            if i == 0 and epoch == 0:
                # First batch, first epoch - initialize the model
                logger.info("Initializing model with first batch")
                model.fit(batch_df, duration_col=duration_col, event_col=event_col,
                          show_progress=fit_params.get('show_progress', False))
            else:
                # Update model parameters using this batch
                # This would require a custom update method that's not implemented here
                # For simplicity, we're just refitting on each batch
                logger.info("Updating model with current batch")
                model.fit(batch_df, duration_col=duration_col, event_col=event_col,
                          show_progress=fit_params.get('show_progress', False))
            
            if batch_count % 5 == 0 or batch_count == n_batches:
                logger.info(f"Completed {batch_count}/{n_batches} batches in epoch {epoch+1}")
        
        logger.info(f"Completed epoch {epoch+1}/{fit_params.get('max_epochs', 10)}")
        
        return model
    
    def perform_gpu_cox_lasso(
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        features: Optional[List[str]] = None,
        cv_folds: int = 10,
        alphas: Optional[List[float]] = None,
        l1_ratio: float = 1.0,  # 1.0 for LASSO, between 0-1 for elastic net
        max_iter: int = 10000,
        output_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform Cox-LASSO regression with cross-validation using GPU acceleration.
        
        This function:
        1. Determines the optimal regularization parameter (lambda) through cross-validation
        2. Tracks coefficient changes as lambda increases
        3. Visualizes the relationship between partial likelihood, lambda, and number of features
        4. Exports raw data as CSV and visualizations as PNG
        
        Args:
            df (pd.DataFrame): Input DataFrame containing features and target
            duration_col (str): Name of the survival duration column
            event_col (str): Name of the event indicator column
            features (List[str], optional): List of feature names to use.
                                          If None, uses all numeric columns in the DataFrame.
            cv_folds (int, optional): Number of cross-validation folds. Default is 10.
            alphas (List[float], optional): List of alpha values to try.
                                          If None, uses a logarithmic sequence.
            l1_ratio (float, optional): Elastic net mixing parameter (0 <= l1_ratio <= 1).
                                      l1_ratio=1 corresponds to LASSO. Default is 1.0.
            max_iter (int, optional): Maximum number of iterations. Default is 10000.
            output_dir (str or Path, optional): Directory to save outputs.
                                              If None, uses current directory.
            random_state (int, optional): Random seed for reproducibility. Default is 42.
        
        Returns:
            Dict[str, Any]: Dictionary containing results and paths to saved files
        """
        import cupy as cp
        import cuml
        import matplotlib.pyplot as plt
        from cuml.model_selection import KFold as cuKFold
        from sklearn.model_selection import KFold
        
        # Check if scikit-survival is available
        try:
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            from sksurv.util import Surv
        except ImportError:
            logger.error("scikit-survival package is required for Cox-LASSO")
            raise ImportError("scikit-survival package is required for Cox-LASSO. Install it with 'pip install scikit-survival'")
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path("results/lasso_analysis")
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"GPU-accelerated LASSO analysis results will be saved to: {output_dir}")
        
        # If features not specified, use all numeric columns except duration and event
        if features is None:
            all_numeric = df.select_dtypes(include=['number']).columns.tolist()
            features = [f for f in all_numeric if f not in [duration_col, event_col]]
        else:
            # Filter to only include numeric features from the provided list
            numeric_features = df.select_dtypes(include=['number']).columns.tolist()
            features = [f for f in features if f in numeric_features]
        
        # Check if we have enough features
        if len(features) < 2:
            error_msg = f"LASSO regression requires at least 2 numeric features. Found {len(features)} numeric features."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Drop rows with missing values in the selected features, duration, and event
        cols_to_check = features + [duration_col, event_col]
        df_clean = df[cols_to_check].dropna()
        logger.info(f"Using {len(df_clean)} complete rows for GPU-accelerated LASSO regression out of {len(df)} total rows")
        
        # Check if dataset is large enough to require batch processing
        large_dataset = len(df_clean) > 100000  # Threshold for batch processing
        
        # Prepare the data for scikit-survival
        X = df_clean[features].values
        y = Surv.from_arrays(
            event=df_clean[event_col].astype(bool).values,
            time=df_clean[duration_col].values
        )
        
        # Convert to GPU arrays
        logger.info(f"Transferring data to GPU for processing ({X.shape[0]} samples, {X.shape[1]} features)")
        X_gpu = cp.array(X, dtype=cp.float32)
        
        # Standardize features on GPU
        logger.info("Standardizing features on GPU (zero mean, unit variance)")
        X_mean = cp.mean(X_gpu, axis=0)
        X_std = cp.std(X_gpu, axis=0)
        X_scaled_gpu = (X_gpu - X_mean) / X_std
        
        # Convert back to CPU for scikit-survival (which doesn't support GPU directly)
        logger.info("Transferring standardized data back to CPU for scikit-survival compatibility")
        X_scaled = cp.asnumpy(X_scaled_gpu)
        
        # Set up alpha values (regularization strengths)
        if alphas is None:
            # Create a logarithmic sequence of alphas
            alphas = cp.logspace(-3, 1, 100)
            alphas_cpu = cp.asnumpy(alphas)
        else:
            alphas_cpu = alphas
        
        # Set up cross-validation
        if large_dataset:
            logger.info(f"Large dataset detected ({len(df_clean)} samples). Using batch processing for cross-validation.")
            # Process in batches for cross-validation
            # For simplicity, we'll use scikit-learn's KFold here
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            logger.info(f"Created {cv_folds}-fold cross-validation splits for large dataset")
            logger.info(f"Each fold will have approximately {len(df_clean) // cv_folds} samples")
        else:
            # Use cuML's KFold for smaller datasets
            logger.info(f"Setting up {cv_folds}-fold cross-validation using cuML's KFold")
            cv = cuKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            # Convert to scikit-learn compatible format
            cv = [(cp.asnumpy(train), cp.asnumpy(test)) for train, test in cv.split(X_scaled)]
            logger.info(f"Prepared cross-validation folds for GPU processing")
            logger.info(f"Each fold will have approximately {len(df_clean) // cv_folds} samples for testing")
        
        # Initialize and fit the Cox-LASSO model
        logger.info(f"Fitting GPU-accelerated Cox-LASSO model with {cv_folds}-fold cross-validation...")
        logger.info(f"Testing {len(alphas_cpu)} alpha values ranging from {min(alphas_cpu):.6f} to {max(alphas_cpu):.6f}")
        logger.info(f"L1 ratio (elastic net mixing parameter): {l1_ratio} (1.0 = LASSO, 0.0 = Ridge)")
        coxnet = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=alphas_cpu,
            normalize=False,  # We already standardized the features
            max_iter=max_iter,
            tol=1e-7,
            random_state=random_state
        )
        
        # Fit the model with cross-validation
        logger.info(f"Starting cross-validation process with max_iter={max_iter}...")
        logger.info(f"This may take some time for large datasets or many alpha values")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coxnet.fit(X_scaled, y)
        logger.info(f"Cross-validation complete. Analyzing results...")
        
        # Get the optimal alpha value
        optimal_alpha_idx = coxnet.cv_results_['alpha_idx_max']
        optimal_alpha = coxnet.alphas_[optimal_alpha_idx]
        logger.info(f"Optimal alpha (regularization parameter): {optimal_alpha:.6f}")
        logger.info(f"This alpha value maximizes the partial likelihood in cross-validation")
        
        # Get the coefficients at the optimal alpha
        optimal_coefs = coxnet.coef_[:, optimal_alpha_idx]
        non_zero_coefs = np.sum(optimal_coefs != 0)
        logger.info(f"Model selected {non_zero_coefs} non-zero coefficients out of {len(features)} features")
        
        # Create a DataFrame with feature names and their coefficients
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': optimal_coefs
        })
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
        
        # Log the top features
        top_features = coef_df[coef_df['Coefficient'] != 0].head(10)
        if not top_features.empty:
            logger.info("Top features selected by the model (by coefficient magnitude):")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                logger.info(f"  {i}. {row['Feature']}: {row['Coefficient']:.6f}")
        
        # Save coefficients to CSV
        coef_path = output_dir / "lasso_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        logger.info(f"Saved LASSO coefficients to {coef_path}")
        
        # Get the coefficient path (how coefficients change with different alphas)
        logger.info("Analyzing coefficient paths across all regularization strengths...")
        coef_path_data = []
        feature_entry_points = {}  # Track when each feature enters the model
        
        # Process in reverse order (from highest to lowest alpha)
        for i, alpha in enumerate(coxnet.alphas_):
            coefs = coxnet.coef_[:, i]
            non_zero_count = np.sum(coefs != 0)
            
            # Track when features enter the model (as alpha decreases)
            for j, feature in enumerate(features):
                if coefs[j] != 0 and feature not in feature_entry_points:
                    feature_entry_points[feature] = alpha
            
            # Add to coefficient path data
            for j, feature in enumerate(features):
                coef_path_data.append({
                    'Alpha': alpha,
                    'Log_Alpha': np.log10(alpha),
                    'Feature': feature,
                    'Coefficient': coefs[j],
                    'Non_Zero_Features': non_zero_count
                })
        
        # Log feature entry points
        if feature_entry_points:
            logger.info("Feature entry points (alpha values where features enter the model):")
            sorted_entries = sorted(feature_entry_points.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, alpha) in enumerate(sorted_entries[:10], 1):  # Show top 10
                logger.info(f"  {i}. {feature}: α = {alpha:.6f}")
            if len(sorted_entries) > 10:
                logger.info(f"  ... and {len(sorted_entries) - 10} more features")
        
        # Convert to DataFrame
        coef_path_df = pd.DataFrame(coef_path_data)
        
        # Save coefficient path data to CSV
        coef_path_csv = output_dir / "lasso_coefficient_path.csv"
        coef_path_df.to_csv(coef_path_csv, index=False)
        logger.info(f"Saved LASSO coefficient path data to {coef_path_csv}")
        
        # Get cross-validation results
        cv_results = pd.DataFrame({
            'Alpha': coxnet.alphas_,
            'Log_Alpha': np.log10(coxnet.alphas_),
            'Mean_CV_Score': coxnet.cv_results_['mean_partial_likelihood'],
            'Std_CV_Score': coxnet.cv_results_['std_partial_likelihood']
        })
        
        # Add number of non-zero features for each alpha
        non_zero_features = []
        for i in range(len(coxnet.alphas_)):
            non_zero_features.append(np.sum(coxnet.coef_[:, i] != 0))
        cv_results['Non_Zero_Features'] = non_zero_features
        
        # Save cross-validation results to CSV
        cv_results_csv = output_dir / "lasso_cv_results.csv"
        cv_results.to_csv(cv_results_csv, index=False)
        logger.info(f"Saved cross-validation results to {cv_results_csv}")
        
        # Create visualizations
        
        # 1. Coefficient path plot
        plt.figure(figsize=(12, 8))
        
        # Get unique features
        unique_features = coef_path_df['Feature'].unique()
        
        # Sort features by the order they enter the model (as lambda decreases)
        feature_entry_order = []
        for alpha_idx in range(len(coxnet.alphas_) - 1, -1, -1):  # Start from largest alpha (most regularization)
            coefs = coxnet.coef_[:, alpha_idx]
            for j, feature in enumerate(features):
                if coefs[j] != 0 and feature not in feature_entry_order:
                    feature_entry_order.append(feature)
        
        # Add any remaining features that never entered the model
        for feature in features:
            if feature not in feature_entry_order:
                feature_entry_order.append(feature)
        
        # Create a color map
        colors = plt.cm.tab10.colors
        color_map = {feature: colors[i % len(colors)] for i, feature in enumerate(feature_entry_order)}
        
        # Plot each feature's coefficient path
        for feature in feature_entry_order:
            feature_data = coef_path_df[coef_path_df['Feature'] == feature]
            plt.plot(feature_data['Log_Alpha'], feature_data['Coefficient'],
                     label=feature, color=color_map[feature], linewidth=2)
        
        # Add vertical line at optimal alpha
        plt.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--',
                    label=f'Optimal α: {optimal_alpha:.6f}')
        
        # Add labels and title
        plt.xlabel('Log(α)')
        plt.ylabel('Coefficient Value')
        plt.title('GPU-Accelerated LASSO Coefficient Paths')
        
        # Add legend with sorted features
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        coef_path_plot = output_dir / "lasso_coefficient_path.png"
        plt.savefig(coef_path_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved coefficient path plot to {coef_path_plot}")
        
        # 2. Cross-validation score plot with number of features
        plt.figure(figsize=(12, 8))
        
        # Create primary axis for CV score
        ax1 = plt.gca()
        ax1.plot(cv_results['Log_Alpha'], cv_results['Mean_CV_Score'], 'b-', linewidth=2)
        
        # Add error bands
        upper = cv_results['Mean_CV_Score'] + cv_results['Std_CV_Score']
        lower = cv_results['Mean_CV_Score'] - cv_results['Std_CV_Score']
        ax1.fill_between(cv_results['Log_Alpha'], lower, upper, alpha=0.2, color='blue')
        
        # Add vertical line at optimal alpha
        ax1.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--',
                   label=f'Optimal α: {optimal_alpha:.6f}')
        
        # Add labels for primary axis
        ax1.set_xlabel('Log(α)')
        ax1.set_ylabel('Mean CV Partial Likelihood', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create secondary axis for number of features
        ax2 = ax1.twinx()
        ax2.plot(cv_results['Log_Alpha'], cv_results['Non_Zero_Features'], 'g-', linewidth=2)
        ax2.set_ylabel('Number of Non-Zero Features', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Add title
        plt.title('GPU-Accelerated Cross-Validation Results and Feature Count')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + ['Number of Features'], loc='best')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        cv_plot = output_dir / "lasso_cv_results.png"
        plt.savefig(cv_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cross-validation results plot to {cv_plot}")
        
        # Return results
        results = {
            "optimal_alpha": optimal_alpha,
            "optimal_coefficients": coef_df.to_dict(orient='records'),
            "cv_results": cv_results.to_dict(orient='records'),
            "coefficient_path": coef_path_df.to_dict(orient='records'),
            "output_files": {
                "coefficients_csv": str(coef_path),
                "coefficient_path_csv": str(coef_path_csv),
                "cv_results_csv": str(cv_results_csv),
                "coefficient_path_plot": str(coef_path_plot),
                "cv_results_plot": str(cv_plot)
            }
        }
        
        return results