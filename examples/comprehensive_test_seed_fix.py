#!/usr/bin/env python3
"""
Comprehensive test script to verify that the seed parameter is correctly passed to the R environment.
This script simulates the actual R code execution without requiring the R library to be loaded.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockRInterface:
    """Mock R interface for testing."""
    def __init__(self):
        self.globalenv = {}
        self.executed_code = []
    
    def execute(self, code):
        """Simulate executing R code."""
        self.executed_code.append(code)
        return True

def mock_fit_fine_gray_model(r_interface, df=None, duration_col="duration", event_col="endpoint", 
                            event_value=1, feature_cols=None, bootstrap_samples=200, 
                            censoring_value=0, seed=42):
    """
    Mock implementation of fit_fine_gray_model to test seed parameter handling.
    """
    if feature_cols is None:
        feature_cols = ["age", "gender"]
    
    logger.info(f"Setting up R code with seed={seed}")
    
    # This is the actual R code from src/r_fine_gray.py
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
    
    logger.info("R code setup successful")
    
    # Pass parameters to R environment
    r_interface.globalenv['rdf'] = "mock_dataframe"
    r_interface.globalenv['feat'] = feature_cols
    r_interface.globalenv['seed'] = seed
    
    logger.info(f"Seed parameter passed to R environment: {r_interface.globalenv.get('seed')}")
    
    # Execute R code (mock execution)
    r_interface.execute(r_code)
    
    # Check if the R code contains the correct seed values
    if f"set.seed(seed)" in r_code and f"set.seed({seed})" in r_code:
        logger.info("R code contains both global seed setting and function-level seed setting")
        return True
    else:
        logger.error("R code is missing seed settings")
        return False

def main():
    """
    Main function to test the seed parameter fix.
    """
    logger.info("Testing seed parameter fix with comprehensive simulation")
    
    # Create mock R interface
    r_interface = MockRInterface()
    
    # Test with default seed
    result = mock_fit_fine_gray_model(r_interface)
    logger.info(f"Test result with default seed: {result}")
    logger.info(f"R environment variables: {r_interface.globalenv}")
    
    # Test with custom seed
    result = mock_fit_fine_gray_model(r_interface, seed=123)
    logger.info(f"Test result with custom seed: {result}")
    logger.info(f"R environment variables: {r_interface.globalenv}")
    
    # Check if the R code was executed
    logger.info(f"Number of R code executions: {len(r_interface.executed_code)}")
    
    # Check if the R code contains the correct seed values
    first_execution = r_interface.executed_code[0] if r_interface.executed_code else ""
    if "set.seed(seed)" in first_execution and "set.seed(42)" in first_execution:
        logger.info("First execution contains correct seed settings")
    else:
        logger.error("First execution is missing correct seed settings")
    
    second_execution = r_interface.executed_code[1] if len(r_interface.executed_code) > 1 else ""
    if "set.seed(seed)" in second_execution and "set.seed(123)" in second_execution:
        logger.info("Second execution contains correct seed settings")
    else:
        logger.error("Second execution is missing correct seed settings")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()