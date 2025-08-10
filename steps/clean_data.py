"""
Data Cleaning Step for CKD Risk Prediction

This module contains the ZenML step for cleaning and preprocessing data.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import the data processing classes from src/data_cleaning.py
from src.data_cleaning import KidneyDataProcessor, UrineDataProcessor, ComorbidityProcessor
from src.data_checker import load_column_structure, check_dataframe


@step
def clean_data(
    cr_df: pd.DataFrame,
    hb_df: pd.DataFrame,
    a1c_df: pd.DataFrame,
    alb_df: pd.DataFrame,
    po4_df: pd.DataFrame,
    ca_df: pd.DataFrame,
    ca_adjusted_df: pd.DataFrame,
    hco3_df: pd.DataFrame,
    upcr_df: pd.DataFrame,
    uacr_df: pd.DataFrame,
    icd10_df: pd.DataFrame,
    operation_df: pd.DataFrame,
    death_df: pd.DataFrame,
    demo_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
          pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
          pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
          pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean and preprocess the lab and demographic data.
    
    Args:
        cr_df: DataFrame containing creatinine data (key, date, code, cr)
        hb_df: DataFrame containing hemoglobin data (key, date, code, hb)
        a1c_df: DataFrame containing hemoglobin A1c data (key, date, code, a1c)
        alb_df: DataFrame containing albumin data (key, date, code, alb)
        po4_df: DataFrame containing phosphate data (key, date, code, po4)
        ca_df: DataFrame containing calcium data (key, date, code, ca)
        ca_adjusted_df: DataFrame containing adjusted calcium data (key, date, code, ca_adjusted)
        upcr_df: DataFrame containing urine protein creatinine ratio data (key, date, code, upacr)
        uacr_df: DataFrame containing urine albumin creatinine ratio data (key, date, code, upacr)
        icd10_df: DataFrame containing ICD-10 diagnosis data (key, date, icd10)
        operation_df: DataFrame containing operation data (key, date)
        death_df: DataFrame containing death data (key, date, cause)
        demo_df: DataFrame containing demographic data (key, dob, gender)
        
    Returns:
        Tuple of DataFrames containing:
        - patient_df: DataFrame with demographic data and endpoints
        - icd10_df_clean: Cleaned ICD-10 diagnosis data
        - cr_df_clean: Cleaned creatinine data
        - hb_df_clean: Cleaned hemoglobin data
        - a1c_df_clean: Cleaned hemoglobin A1c data
        - alb_df_clean: Cleaned albumin data
        - po4_df_clean: Cleaned phosphate data
        - ca_df_clean: Cleaned calcium data
        - ca_adjusted_df_clean: Cleaned adjusted calcium data
        - hco3_df_clean: Cleaned bicarbonate data
        - upcr_df_clean: Cleaned urine protein creatinine ratio data
        - uacr_df_clean: Cleaned urine albumin creatinine ratio data
        - operation_df_clean: Cleaned operation data
        - death_df_clean: Cleaned death data
    """
    try:
        # Print information about each DataFrame
        print(f"Cleaning creatinine data with {len(cr_df)} rows and {len(cr_df.columns)} columns")
        print(f"Creatinine data columns: {cr_df.columns.tolist()}")
        
        print(f"Cleaning hemoglobin data with {len(hb_df)} rows and {len(hb_df.columns)} columns")
        print(f"Hemoglobin data columns: {hb_df.columns.tolist()}")
        
        print(f"Cleaning hemoglobin A1c data with {len(a1c_df)} rows and {len(a1c_df.columns)} columns")
        print(f"Hemoglobin A1c data columns: {a1c_df.columns.tolist()}")
        
        print(f"Cleaning albumin data with {len(alb_df)} rows and {len(alb_df.columns)} columns")
        print(f"Albumin data columns: {alb_df.columns.tolist()}")
        
        print(f"Cleaning phosphate data with {len(po4_df)} rows and {len(po4_df.columns)} columns")
        print(f"Phosphate data columns: {po4_df.columns.tolist()}")
        
        print(f"Cleaning calcium data with {len(ca_df)} rows and {len(ca_df.columns)} columns")
        print(f"Calcium data columns: {ca_df.columns.tolist()}")
        
        print(f"Cleaning adjusted calcium data with {len(ca_adjusted_df)} rows and {len(ca_adjusted_df.columns)} columns")
        print(f"Adjusted calcium data columns: {ca_adjusted_df.columns.tolist()}")
        
        print(f"Cleaning bicarbonate data with {len(hco3_df)} rows and {len(hco3_df.columns)} columns")
        print(f"Bicarbonate data columns: {hco3_df.columns.tolist()}")
        
        print(f"Cleaning urine protein creatinine ratio data with {len(upcr_df)} rows and {len(upcr_df.columns)} columns")
        print(f"Urine protein creatinine ratio data columns: {upcr_df.columns.tolist()}")
        
        print(f"Cleaning urine albumin creatinine ratio data with {len(uacr_df)} rows and {len(uacr_df.columns)} columns")
        print(f"Urine albumin creatinine ratio data columns: {uacr_df.columns.tolist()}")
        
        # Process urine protein data to predict albumin-creatinine ratio
        if not upcr_df.empty and 'upacr' in upcr_df.columns:
            print("\n=== Processing urine protein data to predict albumin-creatinine ratio ===\n")
            upcr_df = UrineDataProcessor.process_urine_data(upcr_df)
        
        print(f"Cleaning ICD-10 diagnosis data with {len(icd10_df)} rows and {len(icd10_df.columns)} columns")
        print(f"ICD-10 diagnosis data columns: {icd10_df.columns.tolist()}")
        
        # Process ICD-10 data to calculate Charlson Comorbidity Index and identify hypertension
        if not icd10_df.empty and 'icd10' in icd10_df.columns:
            print("\n=== Processing ICD-10 data to calculate Charlson Comorbidity Index ===\n")
            cci_df = ComorbidityProcessor.process_icd10_data(icd10_df)
            
            cci_score_df = ComorbidityProcessor.calculate_cci_score(cci_df, demo_df)
            
            print("\n=== Identifying hypertension diagnoses from ICD-10 data ===\n")
            hypertension_df = ComorbidityProcessor.identify_hypertension(icd10_df)
        
        print(f"Cleaning operation data with {len(operation_df)} rows and {len(operation_df.columns)} columns")
        print(f"Operation data columns: {operation_df.columns.tolist()}")
        
        print(f"Cleaning death data with {len(death_df)} rows and {len(death_df.columns)} columns")
        print(f"Death data columns: {death_df.columns.tolist()}")
        
        print(f"Cleaning demographic data with {len(demo_df)} rows and {len(demo_df.columns)} columns")
        print(f"Demographic data columns: {demo_df.columns.tolist()}")
        
        # Use KidneyDataProcessor to clean creatinine data and get endpoint data
        egfr_df, endpoint_df = KidneyDataProcessor.clean_creatinine(cr_df)
        
        # Use KidneyDataProcessor to calculate eGFR using the cleaned creatinine data and demographic data
        egfr_df = KidneyDataProcessor.calculate_egfr(egfr_df, demo_df)
        
        # Use KidneyDataProcessor to find persistent low eGFR timepoints (eGFR < 60)
        ckd_start_df = KidneyDataProcessor.find_ckd_timepoint(egfr_df, threshold=60.0)
        
        # Use KidneyDataProcessor to find patients with eGFR < 10 persistently
        # sub10_df = KidneyDataProcessor.find_ckd_timepoint(egfr_df, threshold=10.0)
        
        # Make copies of the other data to avoid modifying the originals
        cr_df_clean = cr_df.copy() if not cr_df.empty else pd.DataFrame()
        hb_df_clean = hb_df.copy() if not hb_df.empty else pd.DataFrame()
        a1c_df_clean = a1c_df.copy() if not a1c_df.empty else pd.DataFrame()
        alb_df_clean = alb_df.copy() if not alb_df.empty else pd.DataFrame()
        po4_df_clean = po4_df.copy() if not po4_df.empty else pd.DataFrame()
        ca_df_clean = ca_df.copy() if not ca_df.empty else pd.DataFrame()
        ca_adjusted_df_clean = ca_adjusted_df.copy() if not ca_adjusted_df.empty else pd.DataFrame()
        hco3_df_clean = hco3_df.copy() if not hco3_df.empty else pd.DataFrame()
        upcr_df_clean = upcr_df.copy() if not upcr_df.empty else pd.DataFrame()
        uacr_df_clean = uacr_df.copy() if not uacr_df.empty else pd.DataFrame()
        icd10_df_clean = icd10_df.copy() if not icd10_df.empty else pd.DataFrame()
        operation_df_clean = operation_df.copy() if not operation_df.empty else pd.DataFrame()
        death_df_clean = death_df.copy() if not death_df.empty else pd.DataFrame()
        demo_df_clean = demo_df.copy() if not demo_df.empty else pd.DataFrame()
        
        # Clean demographic data
        if not demo_df_clean.empty:
            if 'key' in demo_df_clean.columns:
                # Convert key to numeric if it's not already
                demo_df_clean['key'] = pd.to_numeric(demo_df_clean['key'], errors='coerce')
            
            if 'dob' in demo_df_clean.columns:
                # Ensure dob is in datetime format
                demo_df_clean['dob'] = pd.to_datetime(demo_df_clean['dob'], errors='coerce')
                
                # Calculate age
                current_date = datetime.now()
                demo_df_clean['age'] = (current_date - demo_df_clean['dob']).dt.days / 365.25
                
                # Create age groups only for non-NaN values
                valid_age_mask = demo_df_clean['age'].notna()
                
                # Initialize age_group column as object type to properly handle categorical values
                demo_df_clean['age_group'] = pd.Series(dtype='object')
                
                # Apply categorization only to valid values
                if valid_age_mask.any():
                    demo_df_clean.loc[valid_age_mask, 'age_group'] = pd.cut(
                        demo_df_clean.loc[valid_age_mask, 'age'],
                        bins=[0, 18, 35, 50, 65, 100],
                        labels=['0-18', '19-35', '36-50', '51-65', '65+']
                    )
                    
                    print(f"Created age groups for {valid_age_mask.sum()} patients")
                    print(f"Missing age groups for {(~valid_age_mask).sum()} patients")
            
            if 'gender' in demo_df_clean.columns:
                # Gender is already encoded as 1 for male, 0 for female
                # Do NOT impute missing values - leave them as NaN for later MICE imputation
                pass
        
        # Function to clean lab data
        def clean_lab_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
            if df.empty:
                return df
            
            # Convert key to numeric if it's not already
            if 'key' in df.columns:
                df['key'] = pd.to_numeric(df['key'], errors='coerce')
            
            # Ensure date is in datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Convert value column to numeric, but don't fill NaN values
            if value_col in df.columns:
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                # Don't fill NaN values - leave them for later MICE imputation
                # Report the number of missing values
                missing_count = df[value_col].isna().sum()
                if missing_count > 0:
                    print(f"  {value_col}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
            
            return df
        
        # Clean each lab DataFrame
        hb_df_clean = clean_lab_df(hb_df_clean, 'hemoglobin')
        a1c_df_clean = clean_lab_df(a1c_df_clean, 'a1c')
        alb_df_clean = clean_lab_df(alb_df_clean, 'albumin')
        po4_df_clean = clean_lab_df(po4_df_clean, 'phosphate')
        ca_df_clean = clean_lab_df(ca_df_clean, 'ca')
        ca_adjusted_df_clean = clean_lab_df(ca_adjusted_df_clean, 'ca_adjusted')
        hco3_df_clean = clean_lab_df(hco3_df_clean, 'hco3')
        upcr_df_clean = clean_lab_df(upcr_df_clean, 'upacr')
        
        # combining UACR and UPCR data
        if 'predicted_uacr' in upcr_df_clean.columns:
            print("\n=== Cleaning urine albumin-creatinine ratio ===\n")
            
            # Clean original UACR data first
            if not uacr_df_clean.empty:
                uacr_df_clean = clean_lab_df(uacr_df_clean, 'upacr')
                print(f"Cleaned {len(uacr_df_clean)} original UACR rows")
            
            # Create a copy of upcr_df_clean with predicted_uacr
            predicted_uacr_df = upcr_df_clean.copy()
            
            # Add a column to mark the source of UACR values
            if not uacr_df_clean.empty:
                uacr_df_clean['uacr_source'] = 'original'
            
            # Add the predicted values and mark them
            predicted_uacr_df['upacr'] = predicted_uacr_df['predicted_uacr']
            predicted_uacr_df['uacr_source'] = 'predicted'
            
            print(f"Prepared {len(predicted_uacr_df)} predicted UACR rows")
            
            # Combine both dataframes
            if not uacr_df_clean.empty:
                combined_df = pd.concat([uacr_df_clean, predicted_uacr_df], ignore_index=True)
                print(f"Combined {len(uacr_df_clean)} original and {len(predicted_uacr_df)} predicted rows")
            else:
                combined_df = predicted_uacr_df
                print(f"Using only {len(predicted_uacr_df)} predicted rows (no original data)")
            
            # Sort by key and date
            combined_df = combined_df.sort_values(by=['key', 'date'])
            
            # For each key and date, decide which value to use
            # Group by key and date
            grouped = combined_df.groupby(['key', 'date'])
            
            # Initialize the final dataframe
            final_rows = []
            
            for (key, date), group in grouped:
                if len(group) == 1:
                    # Only one measurement for this key and date
                    final_rows.append(group.iloc[0])
                else:
                    # Multiple measurements - check if original UACR exists
                    original_rows = group[group['uacr_source'] == 'original']
                    if not original_rows.empty:
                        # Use the original UACR value
                        final_rows.append(original_rows.iloc[0])
                    else:
                        # Use the predicted UACR value
                        final_rows.append(group[group['uacr_source'] == 'predicted'].iloc[0])
            
            # Create the final dataframe
            uacr_df_clean = pd.DataFrame(final_rows)
            print(f"Final UACR dataset has {len(uacr_df_clean)} rows after combining original and predicted values")
        else:
            # Clean original UACR data if no predicted values are available
            uacr_df_clean = clean_lab_df(uacr_df_clean, 'upacr')
        
        # Create a patient-centric DataFrame
        # Start with demographic data
        if not demo_df_clean.empty and 'key' in demo_df_clean.columns:
            patient_df = demo_df_clean.copy()
        else:
            # If no demographic data, create a DataFrame with unique patient keys from lab data
            all_keys = set()
            for df in [egfr_df, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, upcr_df_clean, uacr_df_clean, icd10_df_clean, operation_df_clean, death_df_clean]:
                if not df.empty and 'key' in df.columns:
                    all_keys.update(df['key'].dropna().unique())
            
            patient_df = pd.DataFrame({'key': list(all_keys)})
        
        # make sure only dialysis related operation involved in endpoint calculation
        operation_df_clean = operation_df_clean[operation_df_clean['is_dialysis']]
        
        processed_endpoint_df = KidneyDataProcessor.process_endpoints(
            endpoint_df=endpoint_df,
            operation_df=operation_df_clean,
            death_df=death_df_clean,
            egfr_df=egfr_df,
            ckd_start_df=ckd_start_df
        )
        # Add processed endpoint data if available
        if not processed_endpoint_df.empty and 'key' in processed_endpoint_df.columns:
            # Select relevant columns for merging
            endpoint_cols = ['key', 'endpoint', 'endpoint_date', 'endpoint_source']
            if 'first_sub_60_date' in processed_endpoint_df.columns:
                endpoint_cols.append('first_sub_60_date')
            
            # Only keep columns that exist in the DataFrame
            existing_cols = [col for col in endpoint_cols if col in processed_endpoint_df.columns]
            endpoint_subset = processed_endpoint_df[existing_cols]
            
            # Merge with patient DataFrame
            patient_df = pd.merge(patient_df, endpoint_subset, on='key', how='left')
            
            # Create binary indicators for endpoint types
            if 'endpoint' in patient_df.columns:
                # Only create indicators for non-NaN endpoint values
                valid_endpoint_mask = patient_df['endpoint'].notna()
                
                # Initialize endpoint indicator columns with NaN
                patient_df['endpoint_renal'] = np.nan
                patient_df['endpoint_death'] = np.nan
                patient_df['endpoint_censored'] = np.nan
                
                # Apply conditions only to valid values
                if valid_endpoint_mask.any():
                    patient_df.loc[valid_endpoint_mask, 'endpoint_renal'] = (patient_df.loc[valid_endpoint_mask, 'endpoint'] == 1).astype(int)
                    patient_df.loc[valid_endpoint_mask, 'endpoint_death'] = (patient_df.loc[valid_endpoint_mask, 'endpoint'] == 2).astype(int)
                    patient_df.loc[valid_endpoint_mask, 'endpoint_censored'] = (patient_df.loc[valid_endpoint_mask, 'endpoint'] == 0).astype(int)
                
                patient_df = patient_df.dropna(subset=['endpoint', 'endpoint_date', 'endpoint_source'], how='all')
                print(f"Added endpoint information for {patient_df['endpoint'].notna().sum()} patients")
                print(f"Endpoint distribution: {patient_df['endpoint'].value_counts(dropna=False)}")
        # Count missing values for reporting
        missing_counts = patient_df.isna().sum()
        missing_percent = (missing_counts / len(patient_df) * 100).round(2)
        
        # Report columns with significant missing data
        significant_missing = missing_percent[missing_percent > 5].sort_values(ascending=False)
        if not significant_missing.empty:
            print("\n=== Columns with significant missing data (>5%) ===")
            for col, pct in significant_missing.items():
                print(f"  {col}: {pct}% missing ({missing_counts[col]} values)")
               
        print(f"Final cleaned patient data has {len(patient_df)} rows and {len(patient_df.columns)} columns")
        if 'gender' in patient_df.columns:
            print(f"Missing values in gender column: {patient_df['gender'].isna().sum()} ({patient_df['gender'].isna().sum() / len(patient_df) * 100:.2f}%)")
        
        print(patient_df.columns)
        print(icd10_df_clean.columns)
        print(cr_df_clean.columns)
        print(hb_df_clean.columns)
        print(a1c_df_clean.columns)
        print(alb_df_clean.columns)
        print(po4_df_clean.columns)
        print(ca_df_clean.columns)
        print(ca_adjusted_df_clean.columns)
        print(hco3_df_clean.columns)
        print(upcr_df_clean.columns)
        print(uacr_df_clean.columns)
        print(operation_df_clean.columns)
        print(death_df_clean.columns)
        print(cci_df.columns)
        print(cci_score_df.columns)
        print(hypertension_df.columns)
        print(egfr_df.columns)
        
        # Load the expected column structure from the YAML file
        output_structure_yaml_path = "src/default_clean_data_output_dataframe_structure.yml"
        column_structure = load_column_structure(output_structure_yaml_path)
        if not column_structure:
            print("Warning: Could not load column structure from YAML file. Using default structure.")
        else:
            print("Checking and fixing column structure for all DataFrames")
            
            # Check and fix column structure for each DataFrame
            patient_df = check_dataframe(patient_df, 'patient_df', column_structure)
            icd10_df_clean = check_dataframe(icd10_df_clean, 'icd10_df_clean', column_structure)
            cr_df_clean = check_dataframe(cr_df_clean, 'cr_df_clean', column_structure)
            hb_df_clean = check_dataframe(hb_df_clean, 'hb_df_clean', column_structure)
            a1c_df_clean = check_dataframe(a1c_df_clean, 'a1c_df_clean', column_structure)
            alb_df_clean = check_dataframe(alb_df_clean, 'alb_df_clean', column_structure)
            po4_df_clean = check_dataframe(po4_df_clean, 'po4_df_clean', column_structure)
            ca_df_clean = check_dataframe(ca_df_clean, 'ca_df_clean', column_structure)
            ca_adjusted_df_clean = check_dataframe(ca_adjusted_df_clean, 'ca_adjusted_df_clean', column_structure)
            hco3_df_clean = check_dataframe(hco3_df_clean, 'hco3_df_clean', column_structure)
            upcr_df_clean = check_dataframe(upcr_df_clean, 'upcr_df_clean', column_structure)
            uacr_df_clean = check_dataframe(uacr_df_clean, 'uacr_df_clean', column_structure)
            operation_df_clean = check_dataframe(operation_df_clean, 'operation_df_clean', column_structure)
            death_df_clean = check_dataframe(death_df_clean, 'death_df_clean', column_structure)
            cci_df = check_dataframe(cci_df, 'cci_df', column_structure)
            cci_score_df = check_dataframe(cci_score_df, 'cci_score_df', column_structure)
            hypertension_df = check_dataframe(hypertension_df, 'hypertension_df', column_structure)
            egfr_df = check_dataframe(egfr_df, 'egfr_df', column_structure)
        
        # Return a tuple of all cleaned dataframes
        return (
            patient_df,
            icd10_df_clean,
            cr_df_clean,
            hb_df_clean,
            a1c_df_clean,
            alb_df_clean,
            po4_df_clean,
            ca_df_clean,
            ca_adjusted_df_clean,
            hco3_df_clean,
            upcr_df_clean,
            uacr_df_clean,
            operation_df_clean,
            death_df_clean,
            cci_df,
            cci_score_df,
            hypertension_df,
            egfr_df
        )
    
    except Exception as e:
        print(f"Error cleaning data: {e}")
        # Return empty DataFrames if there's an error
        empty_df = pd.DataFrame()
        
        # Try to load the column structure to create properly structured empty DataFrames
        try:
            output_structure_yaml_path = "src/default_clean_data_output_dataframe_structure.yml"
            column_structure = load_column_structure(output_structure_yaml_path)
            
            if column_structure:
                # Create empty DataFrames with the expected column structure
                patient_df = check_dataframe(empty_df, 'patient_df', column_structure)
                icd10_df_clean = check_dataframe(empty_df, 'icd10_df_clean', column_structure)
                cr_df_clean = check_dataframe(empty_df, 'cr_df_clean', column_structure)
                hb_df_clean = check_dataframe(empty_df, 'hb_df_clean', column_structure)
                a1c_df_clean = check_dataframe(empty_df, 'a1c_df_clean', column_structure)
                alb_df_clean = check_dataframe(empty_df, 'alb_df_clean', column_structure)
                po4_df_clean = check_dataframe(empty_df, 'po4_df_clean', column_structure)
                ca_df_clean = check_dataframe(empty_df, 'ca_df_clean', column_structure)
                ca_adjusted_df_clean = check_dataframe(empty_df, 'ca_adjusted_df_clean', column_structure)
                hco3_df_clean = check_dataframe(empty_df, 'hco3_df_clean', column_structure)
                upcr_df_clean = check_dataframe(empty_df, 'upcr_df_clean', column_structure)
                uacr_df_clean = check_dataframe(empty_df, 'uacr_df_clean', column_structure)
                operation_df_clean = check_dataframe(empty_df, 'operation_df_clean', column_structure)
                death_df_clean = check_dataframe(empty_df, 'death_df_clean', column_structure)
                cci_df = check_dataframe(empty_df, 'cci_df', column_structure)
                cci_score_df = check_dataframe(empty_df, 'cci_score_df', column_structure)
                hypertension_df = check_dataframe(empty_df, 'hypertension_df', column_structure)
                egfr_df = check_dataframe(empty_df, 'egfr_df', column_structure)
                
                return (
                    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean,
                    alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean,
                    upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df,
                    cci_score_df, hypertension_df, egfr_df
                )
        except Exception as inner_e:
            print(f"Error creating structured empty DataFrames: {inner_e}")
        
        # Fallback to simple empty DataFrames
        return (
            empty_df, empty_df, empty_df, empty_df, empty_df,
            empty_df, empty_df, empty_df, empty_df, empty_df,
            empty_df, empty_df, empty_df, empty_df, empty_df,
            empty_df, empty_df, empty_df
        )