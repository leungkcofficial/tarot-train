#!/usr/bin/env python3
"""
Baseline Competing-Risk Dashboard — CKD Dialysis & Mortality

This module implements Fine-Gray competing risk models for CKD patients,
providing risk estimates for dialysis and mortality at 1-5 year horizons.
It fits models, generates visualizations, and exports model objects for reuse.

The module uses rpy2 to interface with R's fastcmprsk package for Fine-Gray modeling.
"""

import os
import sys
import time
import json
import logging
import hashlib
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fine_gray_model.log')
    ]
)
logger = logging.getLogger(__name__)

# Import utility functions
try:
    from src.util import load_yaml_file
except ImportError:
    # Handle case where module is run directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.util import load_yaml_file

# Global variables
DEFAULT_MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "default_master_df_mapping.yml")
DEFAULT_TIME_HORIZONS = [365, 730, 1095, 1460, 1825]  # 1-5 years in days

# R interface setup and error handling
class RInterfaceError(Exception):
    """Exception raised for errors in the R interface."""
    pass

class ModelFittingError(Exception):
    """Exception raised for errors in model fitting."""
    pass

class PredictionError(Exception):
    """Exception raised for errors in prediction."""
    pass

class ExportError(Exception):
    """Exception raised for errors in exporting results."""
    pass

def setup_r_environment(seed: int = 42, n_threads: Optional[int] = None) -> Tuple[Any, bool]:
    """
    Set up the R environment for Fine-Gray modeling.
    
    Args:
        seed: Random seed for reproducibility
        n_threads: Number of threads to use (None = auto-detect)
        
    Returns:
        Tuple of (R interface object, success flag)
    """
    try:
        # Import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
        
        # Set R random seed
        ro.r(f'set.seed({seed})')
        
        # Set up thread control
        if n_threads is None:
            # Check environment variables
            env_threads = os.environ.get('OMP_NUM_THREADS') or os.environ.get('MKL_NUM_THREADS')
            if env_threads:
                n_threads = int(env_threads)
            else:
                # Auto-detect: use n_cores - 1, minimum 1
                n_threads = max(1, multiprocessing.cpu_count() - 1)
        
        # Set thread count in R
        ro.r(f'Sys.setenv(OMP_NUM_THREADS={n_threads})')
        ro.r(f'Sys.setenv(MKL_NUM_THREADS={n_threads})')
        
        # Try to load required packages
        required_packages = ['fastcmprsk', 'ggplot2', 'jsonlite', 'tools']
        loaded_packages = {}
        
        for pkg in required_packages:
            try:
                loaded_packages[pkg] = importr(pkg)
                logger.info(f"Successfully loaded R package: {pkg}")
            except Exception as e:
                logger.error(f"Failed to load R package {pkg}: {e}")
                # Try to install the package
                try:
                    logger.info(f"Attempting to install R package: {pkg}")
                    utils = importr('utils')
                    utils.install_packages(pkg)
                    loaded_packages[pkg] = importr(pkg)
                    logger.info(f"Successfully installed and loaded R package: {pkg}")
                except Exception as install_error:
                    logger.error(f"Failed to install R package {pkg}: {install_error}")
                    raise RInterfaceError(
                        f"Required R package '{pkg}' could not be loaded or installed. "
                        f"Please install it manually in R with: install.packages('{pkg}')"
                    )
        
        # Enable pandas-to-R conversion
        pandas2ri.activate()
        
        # Return R interface and success flag
        return {
            'ro': ro,
            'pandas2ri': pandas2ri,
            'importr': importr,
            'localconverter': localconverter,
            'packages': loaded_packages,
            'n_threads': n_threads
        }, True
        
    except ImportError as e:
        logger.error(f"Failed to import rpy2: {e}")
        logger.error("Please install rpy2 with: pip install rpy2")
        return None, False
    except Exception as e:
        logger.error(f"Error setting up R environment: {e}")
        return None, False

def convert_df_to_r(r_interface: Dict[str, Any], df: pd.DataFrame) -> Any:
    """
    Convert a pandas DataFrame to an R DataFrame.
    
    Args:
        r_interface: R interface object from setup_r_environment
        df: Pandas DataFrame to convert
        
    Returns:
        R DataFrame object
    """
    try:
        with r_interface['localconverter'](r_interface['ro'].default_converter + 
                                          r_interface['pandas2ri'].converter):
            r_df = r_interface['ro'].conversion.py2rpy(df)
        
        # Verify conversion
        r_nrow = r_interface['ro'].r('nrow')(r_df)[0]
        r_ncol = r_interface['ro'].r('ncol')(r_df)[0]
        
        if r_nrow != df.shape[0] or r_ncol != df.shape[1]:
            logger.warning(
                f"DataFrame conversion size mismatch: Python {df.shape} vs R {(r_nrow, r_ncol)}"
            )
        
        return r_df
    except Exception as e:
        logger.error(f"Error converting DataFrame to R: {e}")
        raise RInterfaceError(f"Failed to convert DataFrame to R format: {e}")

def validate_dataframe(df: pd.DataFrame, 
                      mapping_file: str = DEFAULT_MAPPING_FILE) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate the input dataframe structure and prepare it for modeling.
    
    Args:
        df: Input DataFrame
        mapping_file: Path to YAML mapping file
        
    Returns:
        Tuple of (validated DataFrame, mapping dictionary)
    """
    start_time = time.time()
    logger.info("Validating input dataframe...")
    
    # Load mapping
    mapping = load_yaml_file(mapping_file)
    if not mapping:
        raise ValueError(f"Failed to load mapping from {mapping_file}")
    
    # Check required columns
    required_cols = [mapping.get('duration'), mapping.get('event')]
    if None in required_cols:
        raise ValueError("Mapping file must specify 'duration' and 'event' columns")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from dataframe: {missing_cols}")
    
    # Check data types and convert if necessary
    duration_col = mapping.get('duration')
    event_col = mapping.get('event')
    
    # Ensure duration is numeric
    if not pd.api.types.is_numeric_dtype(df[duration_col]):
        logger.warning(f"Converting {duration_col} to numeric")
        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
    
    # Ensure event is numeric
    if not pd.api.types.is_numeric_dtype(df[event_col]):
        logger.warning(f"Converting {event_col} to numeric")
        df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    
    # Check for negative durations
    neg_durations = (df[duration_col] < 0).sum()
    if neg_durations > 0:
        logger.warning(f"Found {neg_durations} negative values in {duration_col}")
        logger.info("Removing rows with negative durations")
        df = df[df[duration_col] >= 0]
    
    # Check for invalid endpoint values
    valid_endpoints = mapping.get('endpoint', {})
    if valid_endpoints:
        valid_values = list(valid_endpoints.values())
        invalid_endpoints = df[~df[event_col].isin(valid_values)][event_col].count()
        if invalid_endpoints > 0:
            logger.warning(
                f"Found {invalid_endpoints} invalid values in {event_col}. "
                f"Valid values are {valid_values}"
            )
            logger.info("Removing rows with invalid endpoint values")
            df = df[df[event_col].isin(valid_values)]
    
    # Handle missing values
    original_row_count = df.shape[0]
    required_cols_for_check = required_cols.copy()
    
    # Add feature columns if specified
    feature_cols = mapping.get('features', [])
    if feature_cols:
        required_cols_for_check.extend(feature_cols)
    
    # Drop rows with missing values in required columns
    df_clean = df.dropna(subset=required_cols_for_check)
    dropped_row_count = original_row_count - df_clean.shape[0]
    
    if dropped_row_count > 0:
        logger.warning(
            f"Dropped {dropped_row_count} rows ({dropped_row_count/original_row_count:.2%}) "
            f"with missing values in required columns"
        )
    
    logger.info(f"Validation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Final dataframe shape: {df_clean.shape}")
    
    return df_clean, mapping

def fit_fine_gray_model(r_interface: Dict[str, Any], 
                       df: pd.DataFrame,
                       duration_col: str,
                       event_col: str,
                       event_value: int,
                       feature_cols: List[str],
                       bootstrap_samples: int = 200,
                       censoring_value: int = 0) -> Any:
    """
    Fit a Fine-Gray competing risk model.
    
    Args:
        r_interface: R interface object from setup_r_environment
        df: Input DataFrame
        duration_col: Column name for duration/time
        event_col: Column name for event indicator
        event_value: Value in event_col that indicates the event of interest
        feature_cols: List of feature column names
        bootstrap_samples: Number of bootstrap samples for variance estimation
        censoring_value: Value in event_col that indicates censoring
        
    Returns:
        Fitted Fine-Gray model object
    """
    start_time = time.time()
    logger.info(f"Fitting Fine-Gray model for event {event_value}...")
    
    try:
        # Convert DataFrame to R
        r_df = convert_df_to_r(r_interface, df)
        ro = r_interface['ro']
        
        # Create formula for model
        formula = f"Crisk({duration_col}, {event_col}, failcode={event_value}, cencode={censoring_value}) ~ " + " + ".join(feature_cols)
        logger.debug(f"Model formula: {formula}")
        
        # Set up fastCrr call
        r_code = f"""
        library(fastcmprsk)
        
        # Set global seed for reproducibility
        set.seed(seed)
        
        # Always sample data to a manageable size
        sample_data <- function(df, target_rows=5000) {{
            set.seed({seed})
            if (nrow(df) > target_rows) {{
                # Calculate sampling fraction
                fraction <- target_rows / nrow(df)
                # Sample with stratification by status
                status_table <- table(df$status)
                sampled_indices <- c()
                
                for (s in names(status_table)) {{
                    # Get indices for this status
                    status_indices <- which(df$status == s)
                    # Calculate how many to sample
                    n_sample <- min(length(status_indices), round(length(status_indices) * fraction))
                    # Sample indices
                    if (n_sample > 0) {{
                        sampled_indices <- c(sampled_indices,
                                           sample(status_indices, n_sample))
                    }}
                }}
                
                # Return sampled data
                return(df[sampled_indices,])
            }}
            return(df)
        }}
        
        # Extract columns from dataframe
        duration_vec <- rdf${duration_col}
        endpoint_vec <- rdf${event_col}
        
        # Create a combined dataframe for modeling
        model_df <- data.frame(
            time = duration_vec,
            status = endpoint_vec
        )
        
        # Add feature columns to the model dataframe
        for (f in feat) {{
            model_df[[f]] <- rdf[[f]]
        }}
        
        # Print summary of the model dataframe
        print(paste("Original model dataframe dimensions:", nrow(model_df), "x", ncol(model_df)))
        
        # Always sample data to a manageable size
        sampled_df <- sample_data(model_df)
        print(paste("Using sampled dataframe with dimensions:", nrow(sampled_df), "x", ncol(sampled_df)))
        
        # Print status distribution in sampled data
        status_counts <- table(sampled_df$status)
        print("Status distribution in sampled data:")
        print(status_counts)
        
        # Create formula for the model
        # The formula should be: Crisk(time, status, failcode, cencode) ~ feature1 + feature2 + ...
        formula_str <- paste0("Crisk(time, status, failcode=", {event_value}, ", cencode=", {censoring_value}, ") ~ ",
                             paste(feat, collapse=" + "))
        print(paste("Using formula:", formula_str))
        model_formula <- as.formula(formula_str)
        
        # Try to fit model with bootstrap variance
        # Check if fastCrr supports the B parameter
        has_bootstrap <- tryCatch({{
            # Try to get the formals (parameters) of the fastCrr function
            params <- names(formals(fastCrr))
            "B" %in% params
        }}, error = function(e) {{
            # If there's an error, assume B is not supported
            FALSE
        }})
        
        # Set a timeout for model fitting (10 minutes)
        timeout_seconds <- 10 * 60
        start_time <- Sys.time()
        
        # Wrap model fitting in tryCatch to handle errors
        model_result <- tryCatch({{
            # Set options to get more detailed error messages
            options(warn = 1)  # Print warnings as they occur
            options(error = function() {{
                traceback(3)  # Print traceback on error
            }})
            
            if (has_bootstrap) {{
                # If B parameter is supported, use it
                print("Using fastCrr with bootstrap variance (B parameter)")
                model <- fastCrr(model_formula, data=sampled_df, B={bootstrap_samples})
            }} else {{
                # If B parameter is not supported, use default parameters
                print("Using fastCrr without bootstrap variance (B parameter not supported)")
                model <- fastCrr(model_formula, data=sampled_df)
            }}
            
            # Check if we've exceeded the timeout
            elapsed <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
            if (elapsed > timeout_seconds) {{
                stop(paste("Model fitting timed out after", round(elapsed), "seconds"))
            }}
            
            # Return success
            list(success=TRUE, model=model, error=NULL)
        }}, error = function(e) {{
            # Return error with traceback
            error_msg <- paste("Error:", as.character(e))
            traceback_info <- paste(capture.output(traceback()), collapse="\\n")
            list(success=FALSE, model=NULL, error=paste(error_msg, traceback_info, sep="\\n"))
        }})
        
        # Check if model fitting was successful
        if (model_result$success) {{
            print("Model fitting completed successfully")
            model_result$model
        }} else {{
            stop(paste("Model fitting failed:", model_result$error))
        }}
        """
        
        # Execute R code
        ro.globalenv['rdf'] = r_df
        ro.globalenv['feat'] = ro.StrVector(feature_cols)
        ro.globalenv['seed'] = seed
        
        # Log progress for long-running operations
        logger.info(f"Starting model fitting with {bootstrap_samples} bootstrap samples...")
        progress_logger = ProgressLogger(
            total_iterations=bootstrap_samples,
            operation_name=f"Fine-Gray model fitting (event={event_value})",
            log_interval=10  # Log every 10 seconds
        )
        progress_logger.start()
        
        # Execute R code with timeout
        model = ro.r(r_code)
        
        # Stop progress logger
        progress_logger.stop()
        
        # Check if model is valid
        if ro.r('class')(model)[0] != 'crr':
            raise ModelFittingError(f"Failed to fit Fine-Gray model for event {event_value}")
        
        logger.info(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
        return model
        
    except Exception as e:
        logger.error(f"Error fitting Fine-Gray model for event {event_value}: {e}")
        raise ModelFittingError(f"Failed to fit Fine-Gray model: {e}")

def predict_cif(r_interface: Dict[str, Any],
               model: Any,
               df: pd.DataFrame,
               feature_cols: List[str],
               time_horizons: List[int] = DEFAULT_TIME_HORIZONS) -> pd.DataFrame:
    """
    Predict cumulative incidence function (CIF) at specified time horizons.
    
    Args:
        r_interface: R interface object from setup_r_environment
        model: Fitted Fine-Gray model object
        df: Input DataFrame
        feature_cols: List of feature column names
        time_horizons: List of time points (in days) to predict at
        
    Returns:
        DataFrame with predicted CIF values
    """
    start_time = time.time()
    logger.info(f"Predicting CIF at horizons: {time_horizons}...")
    
    try:
        # Convert DataFrame to R
        r_df = convert_df_to_r(r_interface, df)
        ro = r_interface['ro']
        
        # Set up prediction
        ro.globalenv['rdf'] = r_df
        ro.globalenv['feat'] = ro.StrVector(feature_cols)
        ro.globalenv['model'] = model
        ro.globalenv['horizons'] = ro.FloatVector(time_horizons)
        
        # R code for prediction
        r_code = """
        library(fastcmprsk)
        
        # Create feature dataframe
        feature_df <- rdf[, feat, drop=FALSE]
        
        # Initialize results matrix
        results <- matrix(0, nrow=nrow(feature_df), ncol=length(horizons))
        colnames(results) <- paste0("t", horizons)
        
        # Predict at each horizon
        for (i in 1:length(horizons)) {
            pred <- predict(model, feature_df, horizons[i])
            results[, i] <- pred
        }
        
        # Convert to dataframe
        results_df <- as.data.frame(results)
        results_df
        """
        
        # Log progress for long-running operations
        logger.info(f"Starting prediction for {df.shape[0]} rows at {len(time_horizons)} time horizons...")
        progress_logger = ProgressLogger(
            total_iterations=len(time_horizons),
            operation_name="CIF prediction",
            log_interval=10  # Log every 10 seconds
        )
        progress_logger.start()
        
        # Execute R code
        r_results = ro.r(r_code)
        
        # Stop progress logger
        progress_logger.stop()
        
        # Convert R dataframe to pandas
        with r_interface['localconverter'](ro.default_converter + 
                                          r_interface['pandas2ri'].converter):
            results_df = ro.conversion.rpy2py(r_results)
        
        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        return results_df
        
    except Exception as e:
        logger.error(f"Error predicting CIF: {e}")
        raise PredictionError(f"Failed to predict CIF: {e}")

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate predictions across rows to get mean CIF values.
    
    Args:
        predictions: DataFrame with predicted CIF values
        
    Returns:
        DataFrame with mean CIF values
    """
    start_time = time.time()
    logger.info("Aggregating predictions...")
    
    try:
        # Calculate mean across rows
        mean_cif = predictions.mean().reset_index()
        mean_cif.columns = ['horizon', 'risk_pct']
        
        # Extract horizon days from column names
        mean_cif['horizon_days'] = mean_cif['horizon'].str.replace('t', '').astype(int)
        
        # Convert to percentages
        mean_cif['risk_pct'] = mean_cif['risk_pct'] * 100
        
        # Reorder columns
        mean_cif = mean_cif[['horizon_days', 'risk_pct']]
        
        logger.info(f"Aggregation completed in {time.time() - start_time:.2f} seconds")
        return mean_cif
        
    except Exception as e:
        logger.error(f"Error aggregating predictions: {e}")
        raise PredictionError(f"Failed to aggregate predictions: {e}")

def create_visualization(r_interface: Dict[str, Any],
                        dialysis_risks: pd.DataFrame,
                        death_risks: pd.DataFrame,
                        output_path: str) -> str:
    """
    Create a bar chart visualization of dialysis and death risks.
    
    Args:
        r_interface: R interface object from setup_r_environment
        dialysis_risks: DataFrame with dialysis risk percentages
        death_risks: DataFrame with death risk percentages
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    start_time = time.time()
    logger.info("Creating visualization...")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Combine risks into a single dataframe
        combined_df = pd.merge(
            dialysis_risks.rename(columns={'risk_pct': 'dialysis_pct'}),
            death_risks.rename(columns={'risk_pct': 'death_pct'}),
            on='horizon_days'
        )
        
        # Convert to long format for ggplot
        long_df = pd.melt(
            combined_df,
            id_vars=['horizon_days'],
            value_vars=['dialysis_pct', 'death_pct'],
            var_name='endpoint',
            value_name='risk_pct'
        )
        
        # Convert to R dataframe
        r_long_df = convert_df_to_r(r_interface, long_df)
        ro = r_interface['ro']
        
        # Set up ggplot call
        ro.globalenv['df'] = r_long_df
        ro.globalenv['output_file'] = os.path.join(output_path, 'baseline_cif.png')
        
        # R code for visualization
        r_code = """
        library(ggplot2)
        
        # Create labels for x-axis
        df$horizon_label <- paste0(df$horizon_days %/% 365, "-year")
        
        # Create endpoint labels
        df$endpoint_label <- ifelse(df$endpoint == "dialysis_pct", "Dialysis", "Death")
        
        # Create plot
        p <- ggplot(df, aes(x = horizon_label, y = risk_pct, fill = endpoint_label)) +
            geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.8) +
            geom_text(aes(label = sprintf("%.1f%%", risk_pct)), 
                      position = position_dodge(width = 0.9), 
                      vjust = -0.5, size = 3.5) +
            scale_fill_manual(values = c("Dialysis" = "#3366CC", "Death" = "#FF9933")) +
            labs(
                title = "Baseline Competing Risk: CKD Dialysis & Mortality",
                x = "Time Horizon",
                y = "Risk (%)",
                fill = "Endpoint"
            ) +
            theme_minimal() +
            theme(
                plot.title = element_text(hjust = 0.5, face = "bold"),
                legend.position = "bottom",
                panel.grid.minor = element_blank()
            )
        
        # Save plot
        ggsave(output_file, p, width = 10, height = 6, dpi = 300)
        
        # Return success flag
        TRUE
        """
        
        # Execute R code
        result = ro.r(r_code)
        
        if not result[0]:
            raise ExportError("Failed to create visualization")
        
        logger.info(f"Visualization created in {time.time() - start_time:.2f} seconds")
        return os.path.join(output_path, 'baseline_cif.png')
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise ExportError(f"Failed to create visualization: {e}")

def export_results(combined_risks: pd.DataFrame, output_path: str) -> str:
    """
    Export results to CSV.
    
    Args:
        combined_risks: DataFrame with combined risk percentages
        output_path: Path to save the results
        
    Returns:
        Path to the saved CSV file
    """
    start_time = time.time()
    logger.info("Exporting results to CSV...")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save to CSV
        csv_path = os.path.join(output_path, 'baseline_cif.csv')
        combined_risks.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to {csv_path} in {time.time() - start_time:.2f} seconds")
        return csv_path
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise ExportError(f"Failed to export results: {e}")

def export_models(r_interface: Dict[str, Any],
                 dialysis_model: Any,
                 death_model: Any,
                 output_path: str,
                 seed: int,
                 feature_cols: List[str],
                 row_count: int) -> Dict[str, str]:
    """
    Export fitted models to RDS files and create metadata.
    
    Args:
        r_interface: R interface object from setup_r_environment
        dialysis_model: Fitted Fine-Gray model for dialysis
        death_model: Fitted Fine-Gray model for death
        output_path: Path to save the models
        seed: Random seed used for model fitting
        feature_cols: List of feature column names
        row_count: Number of rows in the input dataframe
        
    Returns:
        Dictionary with paths to saved files
    """
    start_time = time.time()
    logger.info("Exporting models...")
    
    try:
        # Ensure models directory exists
        models_dir = os.path.join(output_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Set up R environment
        ro = r_interface['ro']
        ro.globalenv['dialysis_model'] = dialysis_model
        ro.globalenv['death_model'] = death_model
        ro.globalenv['dialysis_path'] = os.path.join(models_dir, 'fg_fit_dialysis.rds')
        ro.globalenv['death_path'] = os.path.join(models_dir, 'fg_fit_death.rds')
        
        # R code for saving models
        r_code = """
        # Save models with compression
        saveRDS(dialysis_model, file = dialysis_path, compress = "xz")
        saveRDS(death_model, file = death_path, compress = "xz")
        
        # Get MD5 checksums
        library(tools)
        dialysis_md5 <- md5sum(dialysis_path)
        death_md5 <- md5sum(death_path)
        
        # Get R version and package info
        r_version <- R.version.string
        fastcmprsk_version <- packageVersion("fastcmprsk")
        
        # Return as list
        list(
            dialysis_md5 = dialysis_md5,
            death_md5 = death_md5,
            r_version = r_version,
            fastcmprsk_version = as.character(fastcmprsk_version)
        )
        """
        
        # Execute R code
        r_result = ro.r(r_code)
        
        # Extract results
        with r_interface['localconverter'](ro.default_converter + 
                                          r_interface['pandas2ri'].converter):
            result_dict = {k: r_result.rx2(k)[0] for k in r_result.names}
        
        # Create metadata
        metadata = {
            'seed': seed,
            'r_version': result_dict['r_version'],
            'package_versions': {
                'fastcmprsk': result_dict['fastcmprsk_version']
            },
            'datetime': datetime.datetime.now().isoformat(),
            'row_count': row_count,
            'feature_count': len(feature_cols),
            'features': feature_cols,
            'md5_checksums': {
                'fg_fit_dialysis.rds': result_dict['dialysis_md5'],
                'fg_fit_death.rds': result_dict['death_md5']
            }
        }
        
        # Save metadata
        meta_path = os.path.join(models_dir, 'model_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models exported in {time.time() - start_time:.2f} seconds")
        return {
            'dialysis_model': os.path.join(models_dir, 'fg_fit_dialysis.rds'),
            'death_model': os.path.join(models_dir, 'fg_fit_death.rds'),
            'metadata': meta_path
        }
        
    except Exception as e:
        logger.error(f"Error exporting models: {e}")
        raise ExportError(f"Failed to export models: {e}")

def load_model(r_interface: Dict[str, Any], model_path: str) -> Any:
    """
    Load a saved Fine-Gray model from an RDS file.
    
    Args:
        r_interface: R interface object from setup_r_environment
        model_path: Path to the RDS file
        
    Returns:
        Loaded model object
    """
    start_time = time.time()
    logger.info(f"Loading model from {model_path}...")
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set up R environment
        ro = r_interface['ro']
        ro.globalenv['model_path'] = model_path
        
        # R code for loading model
        r_code = """
        # Load model
        model <- readRDS(model_path)
        
        # Check if it's a crr object
        if (!inherits(model, "crr")) {
            stop("Loaded object is not a Fine-Gray model (crr class)")
        }
        
        # Return model
        model
        """
        
        # Execute R code
        model = ro.r(r_code)
        
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RInterfaceError(f"Failed to load model: {e}")

def predict_with_loaded_model(r_interface: Dict[str, Any],
                             model: Any,
                             df: pd.DataFrame,
                             feature_cols: List[str],
                             time_horizons: List[int] = DEFAULT_TIME_HORIZONS) -> pd.DataFrame:
    """
    Predict using a loaded model.
    
    Args:
        r_interface: R interface object from setup_r_environment
        model: Loaded Fine-Gray model object
        df: Input DataFrame
        feature_cols: List of feature column names
        time_horizons: List of time points (in days) to predict at
        
    Returns:
        DataFrame with predicted CIF values
    """
    # This function is similar to predict_cif but specifically for loaded models
    return predict_cif(r_interface, model, df, feature_cols, time_horizons)

class ProgressLogger:
    """
    Utility class for logging progress of long-running operations.
    """
    def __init__(self, total_iterations: int, operation_name: str, log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            total_iterations: Total number of iterations
            operation_name: Name of the operation
            log_interval: Interval in seconds between log messages
        """
        self.total_iterations = total_iterations
        self.operation_name = operation_name
        self.log_interval = log_interval
        self.start_time = None
        self.running = False
        self.thread = None
    
    def _log_progress(self):
        """Log progress at regular intervals."""
        import threading
        
        while self.running:
            elapsed = time.time() - self.start_time
            logger.info(
                f"{self.operation_name} in progress... "
                f"Elapsed time: {elapsed:.1f} seconds"
            )
            time.sleep(self.log_interval)
    
    def start(self):
        """Start the progress logger."""
        import threading
        
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._log_progress)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the progress logger."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        elapsed = time.time() - self.start_time
        logger.info(f"{self.operation_name} completed in {elapsed:.1f} seconds")


def run_baseline_cif(df: pd.DataFrame,
                    feature_cols: Optional[List[str]] = None,
                    output_path: str = "./output",
                    seed: int = 42,
                    n_threads: Optional[int] = None,
                    silent: bool = False) -> Dict[str, Any]:
    """
    Main function to run the baseline competing-risk Fine-Gray analysis.
    
    This function is designed to work directly with in-memory pandas DataFrames,
    making it ideal for integration into data processing pipelines where the
    DataFrame is already available from previous processing steps.
    
    Args:
        df (pd.DataFrame): Input dataframe with duration, endpoint, and feature columns.
                          In pipeline contexts, this would be the DataFrame from previous steps.
        feature_cols (list, optional): List of feature column names. If None, uses default from YAML.
                                      In pipeline contexts, this could be the output of a feature selection step.
        output_path (str, optional): Path to save outputs. Defaults to "./output".
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_threads (Optional[int], optional): Number of threads to use. If None, respects env vars or auto-detects.
        silent (bool, optional): Whether to suppress progress logs. Defaults to False.
        
    Returns:
        dict: Dictionary containing results (risks at each horizon for dialysis and death)
              that can be used in subsequent pipeline steps.
    
    Example (Pipeline Integration):
        # In a pipeline context
        processed_df = previous_pipeline_step()
        selected_features = feature_selection_step(processed_df)
        
        fine_gray_results = run_baseline_cif(
            df=processed_df,
            feature_cols=selected_features,
            output_path=output_dir
        )
        
        next_pipeline_step(fine_gray_results)
    """
    # Set up logging level based on silent flag
    if silent:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    
    start_time = time.time()
    logger.info(f"Starting baseline competing-risk analysis with seed {seed}")
    
    try:
        # Step 1: Validate dataframe
        df_clean, mapping = validate_dataframe(df, DEFAULT_MAPPING_FILE)
        
        # Get column names
        duration_col = mapping.get('duration')
        event_col = mapping.get('event')
        
        # Get endpoint values
        endpoint_mapping = mapping.get('endpoint', {})
        dialysis_code = endpoint_mapping.get('dialysis', 1)
        death_code = endpoint_mapping.get('mortality', 2)
        censored_code = endpoint_mapping.get('censored', 0)
        
        # Get feature columns
        if feature_cols is None:
            feature_cols = mapping.get('features', [])
            logger.info(f"Using default feature columns from mapping: {feature_cols}")
        
        # Step 2: Set up R environment
        r_interface, success = setup_r_environment(seed, n_threads)
        if not success:
            raise RInterfaceError("Failed to set up R environment")
        
        # Step 3: Fit Fine-Gray models
        # Fit model for dialysis
        dialysis_model = fit_fine_gray_model(
            r_interface,
            df_clean,
            duration_col,
            event_col,
            dialysis_code,
            feature_cols,
            bootstrap_samples=200,
            censoring_value=censored_code
        )
        
        # Fit model for death
        death_model = fit_fine_gray_model(
            r_interface,
            df_clean,
            duration_col,
            event_col,
            death_code,
            feature_cols,
            bootstrap_samples=200,
            censoring_value=censored_code
        )
        
        # Step 4: Predict at time horizons
        time_horizons = DEFAULT_TIME_HORIZONS
        
        # Predict for dialysis
        dialysis_predictions = predict_cif(
            r_interface,
            dialysis_model,
            df_clean,
            feature_cols,
            time_horizons
        )
        
        # Predict for death
        death_predictions = predict_cif(
            r_interface,
            death_model,
            df_clean,
            feature_cols,
            time_horizons
        )
        
        # Step 5: Aggregate predictions
        dialysis_risks = aggregate_predictions(dialysis_predictions)
        death_risks = aggregate_predictions(death_predictions)
        
        # Combine risks
        combined_risks = pd.merge(
            dialysis_risks.rename(columns={'risk_pct': 'dialysis_pct'}),
            death_risks.rename(columns={'risk_pct': 'death_pct'}),
            on='horizon_days'
        )
        
        # Step 6: Create visualization
        viz_path = create_visualization(
            r_interface,
            dialysis_risks,
            death_risks,
            output_path
        )
        
        # Step 7: Export results
        csv_path = export_results(combined_risks, output_path)
        
        # Step 8: Export models
        model_paths = export_models(
            r_interface,
            dialysis_model,
            death_model,
            output_path,
            seed,
            feature_cols,
            df_clean.shape[0]
        )
        
        # Step 9: Return results
        results = {
            'dialysis_risks': dialysis_risks.to_dict('records'),
            'death_risks': death_risks.to_dict('records'),
            'combined_risks': combined_risks.to_dict('records'),
            'visualization_path': viz_path,
            'csv_path': csv_path,
            'model_paths': model_paths,
            'runtime_seconds': time.time() - start_time
        }
        
        logger.info(f"Baseline competing-risk analysis completed in {results['runtime_seconds']:.2f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Error in baseline competing-risk analysis: {e}")
        raise


def load_and_predict(model_path: str,
                    df: pd.DataFrame,
                    feature_cols: List[str],
                    time_horizons: List[int] = DEFAULT_TIME_HORIZONS,
                    seed: int = 42,
                    n_threads: Optional[int] = None) -> pd.DataFrame:
    """
    Load a saved Fine-Gray model and predict on new data.
    
    This function is designed to work with in-memory DataFrames in pipeline contexts,
    allowing for model reuse without refitting. It's particularly useful in production
    pipelines where you want to apply a previously trained model to new data.
    
    Args:
        model_path: Path to the saved model RDS file
        df: Input DataFrame (in pipeline contexts, this would be from previous steps)
        feature_cols: List of feature column names
        time_horizons: List of time points (in days) to predict at
        seed: Random seed for reproducibility
        n_threads: Number of threads to use
        
    Returns:
        DataFrame with predicted CIF values that can be used in subsequent pipeline steps
    
    Example (Pipeline Integration):
        # In a production pipeline
        new_patient_data = data_preprocessing_step()
        
        # Load previously trained model and predict
        predictions = load_and_predict(
            model_path="models/fg_fit_dialysis.rds",
            df=new_patient_data,
            feature_cols=model_features
        )
        
        risk_assessment = post_processing_step(predictions)
    """
    try:
        # Set up R environment
        r_interface, success = setup_r_environment(seed, n_threads)
        if not success:
            raise RInterfaceError("Failed to set up R environment")
        
        # Load model
        model = load_model(r_interface, model_path)
        
        # Predict
        predictions = predict_with_loaded_model(
            r_interface,
            model,
            df,
            feature_cols,
            time_horizons
        )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in load_and_predict: {e}")
        raise


if __name__ == "__main__":
    """
    Command-line interface for standalone usage.
    
    Note: This is an alternative entry point for standalone usage.
    For integration into pipelines, import and use the run_baseline_cif function directly.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run baseline competing-risk Fine-Gray analysis (standalone mode)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="./output", help="Path to output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threads", type=int, help="Number of threads to use")
    parser.add_argument("--silent", action="store_true", help="Suppress progress logs")
    args = parser.parse_args()
    
    try:
        # Load data from CSV (only needed in standalone mode)
        logger.info(f"Loading data from {args.input} (standalone mode)")
        df = pd.read_csv(args.input)
        
        # Run analysis
        results = run_baseline_cif(
            df,
            output_path=args.output,
            seed=args.seed,
            n_threads=args.threads,
            silent=args.silent
        )
        
        print(f"Analysis completed successfully. Results saved to {args.output}")
        print(f"Dialysis risks: {results['dialysis_risks']}")
        print(f"Death risks: {results['death_risks']}")
        print("\nNote: For pipeline integration, import and use the run_baseline_cif function directly.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)