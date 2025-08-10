"""
KFRE.py
========
Object-oriented implementation of the 4- and 8-variable Non-North-American
Kidney-Failure-Risk-Equation (KFRE).

YAML Mapping File Format
------------------------
The YAML mapping file should contain a 'kfre' key with a dictionary mapping
standard KFRE variable names to user column names:

```yaml
kfre: {
  'age' : 'age',
  'sex': 'gender',
  'egfr': 'egfr',
  'acr': 'uacr',
  'albumin': 'albumin',
  'phosphate': 'phosphate',
  'bicarbonate': 'bicarbonate',
  'calcium': 'calcium'
}
```

The keys (left side) must be exactly as shown above, as they are the standard
KFRE variable names. The values (right side) should be the column names in
your DataFrame.

Required keys for 4-variable KFRE: 'age', 'sex', 'egfr', 'acr'
Required keys for 8-variable KFRE: all of the above plus 'albumin', 'phosphate',
'bicarbonate', 'calcium'

Public API
----------
KFRECalculator(mapping_yaml_path).add_kfre_risk(df) -> df_with_risk
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional, List, Any

__all__ = ["KFRECalculator", "calculate_kfre_4var", "calculate_kfre_8var", "calculate_kfre_risk", "KFRE"]


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


# ---- Unit Conversion Functions ----

def convert_albumin(value: float, unit: Optional[str] = None) -> float:
    """
    Convert albumin to g/L (standard unit for KFRE).
    
    Parameters
    ----------
    value : float
        Albumin value
    unit : str, optional
        Unit of the value. Options: 'g/L', 'g/dL'
        If None, assumes value is already in g/L
        
    Returns
    -------
    float
        Albumin value in g/L
    """
    if pd.isna(value):
        return np.nan
        
    if unit is None or unit.lower() == 'g/l':
        return value
    elif unit.lower() == 'g/dl':
        return value * 10.0
    else:
        raise ValueError(f"Unsupported albumin unit: {unit}. Use 'g/L' or 'g/dL'.")


def convert_phosphate(value: float, unit: Optional[str] = None) -> float:
    """
    Convert phosphate to mmol/L (standard unit for KFRE).
    
    Parameters
    ----------
    value : float
        Phosphate value
    unit : str, optional
        Unit of the value. Options: 'mmol/L', 'mg/dL'
        If None, assumes value is already in mmol/L
        
    Returns
    -------
    float
        Phosphate value in mmol/L
    """
    if pd.isna(value):
        return np.nan
        
    if unit is None or unit.lower() == 'mmol/l':
        return value
    elif unit.lower() == 'mg/dl':
        return value * 0.3229  # Conversion factor from mg/dL to mmol/L
    else:
        raise ValueError(f"Unsupported phosphate unit: {unit}. Use 'mmol/L' or 'mg/dL'.")


def convert_calcium(value: float, unit: Optional[str] = None) -> float:
    """
    Convert calcium to mmol/L (standard unit for KFRE).
    
    Parameters
    ----------
    value : float
        Calcium value
    unit : str, optional
        Unit of the value. Options: 'mmol/L', 'mg/dL'
        If None, assumes value is already in mmol/L
        
    Returns
    -------
    float
        Calcium value in mmol/L
    """
    if pd.isna(value):
        return np.nan
        
    if unit is None or unit.lower() == 'mmol/l':
        return value
    elif unit.lower() == 'mg/dl':
        return value * 0.2495  # Conversion factor from mg/dL to mmol/L
    else:
        raise ValueError(f"Unsupported calcium unit: {unit}. Use 'mmol/L' or 'mg/dL'.")


def convert_bicarbonate(value: float, unit: Optional[str] = None) -> float:
    """
    Convert bicarbonate to mmol/L (standard unit for KFRE).
    
    Parameters
    ----------
    value : float
        Bicarbonate value
    unit : str, optional
        Unit of the value. Options: 'mmol/L', 'mEq/L'
        If None, assumes value is already in mmol/L
        
    Returns
    -------
    float
        Bicarbonate value in mmol/L
    """
    if pd.isna(value):
        return np.nan
        
    if unit is None or unit.lower() in ['mmol/l', 'meq/l']:
        # mmol/L and mEq/L are equivalent for bicarbonate
        return value
    else:
        raise ValueError(f"Unsupported bicarbonate unit: {unit}. Use 'mmol/L' or 'mEq/L'.")


class KFRECalculator:
    """
    Object-oriented implementation of the 4- and 8-variable Non-North-American
    Kidney-Failure-Risk-Equation (KFRE).
    """
    
    # ---- Tangri Non-North-American coefficients (units already converted) ----
    _COEF_4V: Dict[str, float] = {
        "age": -0.01985,
        "egfr": -0.09886,
        "lnacr": 0.35066,
        "sex": 0.14842,  # male = 1
    }

    _COEF_8V: Dict[str, float] = {
        **_COEF_4V,
        "albumin": -0.03727,
        "phosphate": 0.90638,
        "bicarbonate": -0.07257,
        "calcium": -0.79539,
    }

    # Baseline survivals (non-North-American, Tangri et al.)
    _BASE_S0 = {
        "4v2y": 0.9878,
        "4v5y": 0.9570,
        "8v2y": 0.9878,
        "8v5y": 0.9570,
    }

    def __init__(self, mapping_yaml: Union[str, Path]):
        """
        Initialize the KFRECalculator with a mapping file.
        
        Parameters
        ----------
        mapping_yaml : str or pathlib.Path
            Path to YAML file that maps user column names to required KFRE names.
        """
        self.mapping = self._load_mapping(mapping_yaml)

    # --------------------------------------------------------------------- #
    #                                PUBLIC                                 #
    # --------------------------------------------------------------------- #
    def add_kfre_risk(self, df: pd.DataFrame, units: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Return a copy of `df` with four extra columns: 4v2y, 4v5y, 8v2y, 8v5y.

        Any row lacking the required inputs for a given formula receives NaN
        in the corresponding risk column.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        units : dict, optional
            Dictionary specifying units for biochemical values
            Keys: 'calcium', 'phosphate', 'albumin', 'bicarbonate'
            Values: unit strings (e.g., 'mmol/L', 'mg/dL')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added KFRE risk columns
        """
        # Default units dictionary
        if units is None:
            units = {
                'calcium': 'mmol/L',
                'phosphate': 'mmol/L',
                'albumin': 'g/L',
                'bicarbonate': 'mmol/L'
            }
            
        # Make a copy of the DataFrame
        df_copy = df.copy()

        # Standardize columns
        std_df = self._standardise_columns(df_copy)
        
        # Convert units if needed
        if 'calcium' in std_df.columns and units.get('calcium') != 'mmol/L':
            std_df['calcium'] = std_df['calcium'].apply(lambda x: convert_calcium(x, units.get('calcium')))
        
        if 'phosphate' in std_df.columns and units.get('phosphate') != 'mmol/L':
            std_df['phosphate'] = std_df['phosphate'].apply(lambda x: convert_phosphate(x, units.get('phosphate')))
        
        if 'albumin' in std_df.columns and units.get('albumin') != 'g/L':
            std_df['albumin'] = std_df['albumin'].apply(lambda x: convert_albumin(x, units.get('albumin')))
        
        if 'bicarbonate' in std_df.columns and units.get('bicarbonate') != 'mmol/L':
            std_df['bicarbonate'] = std_df['bicarbonate'].apply(lambda x: convert_bicarbonate(x, units.get('bicarbonate')))

        # --- Compute LPs --------------------------------------------------- #
        lp4 = self._linear_predictor(std_df, self._COEF_4V)
        lp8 = self._linear_predictor(std_df, self._COEF_8V)

        # --- Risk conversion ---------------------------------------------- #
        risks = {
            "4v2y": self._cox_risk(lp4, self._BASE_S0["4v2y"]),
            "4v5y": self._cox_risk(lp4, self._BASE_S0["4v5y"]),
            "8v2y": self._cox_risk(lp8, self._BASE_S0["8v2y"]),
            "8v5y": self._cox_risk(lp8, self._BASE_S0["8v5y"]),
        }

        for k, v in risks.items():
            df_copy[k] = v

        return df_copy

    # --------------------------------------------------------------------- #
    #                               PRIVATE                                 #
    # --------------------------------------------------------------------- #
    def _load_mapping(self, path: Union[str, Path]) -> Dict[str, str]:
        """
        Load and validate the YAML mapping file.
        
        Parameters
        ----------
        path : str or pathlib.Path
            Path to the YAML mapping file
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping standard KFRE variable names to user column names
            
        Raises
        ------
        FileNotFoundError
            If the mapping file does not exist
        yaml.YAMLError
            If the mapping file is not valid YAML
        KeyError
            If the mapping file does not contain the 'kfre' key
        ValidationError
            If the mapping file is missing required keys
        """
        # Check if file exists
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")
        
        try:
            # Load YAML file
            with open(path_obj, "r", encoding="utf-8") as fh:
                yaml_content = yaml.safe_load(fh)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        
        # Extract KFRE mapping
        mapping = self._extract_mapping(yaml_content)
        
        # Validate required keys
        self._validate_mapping(mapping)
        
        return mapping
    
    def _extract_mapping(self, yaml_content: Dict) -> Dict[str, str]:
        """
        Extract KFRE mapping from YAML content, handling different formats.
        
        Parameters
        ----------
        yaml_content : Dict
            Parsed YAML content
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping standard KFRE variable names to user column names
            
        Raises
        ------
        KeyError
            If no valid mapping can be extracted
        """
        # Try different possible locations for the mapping
        if 'kfre' in yaml_content:
            return yaml_content['kfre']
        elif 'mapping' in yaml_content and 'kfre' in yaml_content['mapping']:
            return yaml_content['mapping']['kfre']
        elif 'columns' in yaml_content and 'kfre' in yaml_content['columns']:
            return yaml_content['columns']['kfre']
        else:
            # If we can't find a mapping, check if the top-level keys match what we need
            required_keys = {'age', 'sex', 'egfr', 'acr', 'albumin', 'phosphate', 'bicarbonate', 'calcium'}
            if required_keys.issubset(yaml_content.keys()):
                return {k: yaml_content[k] for k in required_keys if k in yaml_content}
            
            # If we still can't find a mapping, raise an error
            raise KeyError("Could not find KFRE mapping in YAML file")
    
    def _validate_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Validate that the mapping contains all required keys.
        
        Parameters
        ----------
        mapping : Dict[str, str]
            Dictionary mapping standard KFRE variable names to user column names
            
        Raises
        ------
        ValidationError
            If the mapping is missing required keys
        """
        # Define required keys for 4-variable and 8-variable KFRE
        required_keys_4v = {'age', 'sex', 'egfr', 'acr'}
        required_keys_8v = required_keys_4v.union({'albumin', 'phosphate', 'bicarbonate', 'calcium'})
        
        # Check if all required keys are present
        missing_keys_4v = required_keys_4v.difference(mapping.keys())
        missing_keys_8v = required_keys_8v.difference(mapping.keys())
        
        # Prepare error message
        error_msgs = []
        if missing_keys_4v:
            error_msgs.append(f"Missing required keys for 4-variable KFRE: {sorted(missing_keys_4v)}")
        if missing_keys_8v and not missing_keys_4v:  # Only show 8v message if 4v is complete
            error_msgs.append(f"Missing required keys for 8-variable KFRE: {sorted(missing_keys_8v)}")
        
        # Raise error if any keys are missing
        if error_msgs:
            raise ValidationError("\n".join(error_msgs))

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and handle gender/sex variations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized columns
        """
        # Make a copy of the DataFrame
        df_std = df.copy()
        
        # Create a mapping from user columns to standard names
        col_mapping = {user_col: std_name for std_name, user_col in self.mapping.items() if user_col in df.columns}
        
        # Rename columns if present
        df_std = df_std.rename(columns=col_mapping)
        
        # Handle gender/sex variations
        if 'sex' not in df_std.columns and 'gender' in df_std.columns:
            # If gender is present but sex is not, use gender as sex
            df_std['sex'] = df_std['gender']
        
        # Ensure all required columns exist
        for std_name in self.mapping.keys():
            if std_name not in df_std.columns:
                df_std[std_name] = np.nan
        
        # Standardize sex encoding: 1 for male, 0 for female
        if 'sex' in df_std.columns and pd.api.types.is_object_dtype(df_std['sex']):
            # If sex is a string, convert to numeric
            sex_map = {'male': 1, 'm': 1, 'man': 1, '1': 1, 'true': 1,
                      'female': 0, 'f': 0, 'woman': 0, '0': 0, 'false': 0}
            df_std['sex'] = df_std['sex'].str.lower().map(sex_map)
        
        # Add derived ln(ACR) column
        df_std['lnacr'] = np.log(df_std['acr'].astype(float).replace({0: np.nan}))
        
        # Remove duplicate columns if any
        df_std = df_std.loc[:, ~df_std.columns.duplicated()]
        
        return df_std

    @staticmethod
    def _linear_predictor(df: pd.DataFrame, coefs: Dict[str, float]) -> np.ndarray:
        """
        Vectorised dot-product of standardised DataFrame with coefficient dict.
        Rows with any NaN among required variables return NaN LP.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with standardized columns
        coefs : dict
            Dictionary of coefficients
            
        Returns
        -------
        np.ndarray
            Linear predictor values
        """
        # Extract variables in fixed order
        # Extract only the variables needed for this model
        vars_needed = list(coefs.keys())
        
        # Create a new DataFrame with only the columns we need
        # If a column appears multiple times, use the first occurrence
        df_subset = pd.DataFrame()
        for var in vars_needed:
            if var not in df.columns:
                raise ValueError(f"Required variable '{var}' not found in DataFrame")
            # Only add the column if it's not already in the subset
            if var not in df_subset.columns:
                df_subset[var] = df[var]
        
        # Extract data as numpy array
        X = df_subset.astype(float).to_numpy()
        beta = np.array([coefs[v] for v in vars_needed], dtype=float)

        # Calculate linear predictor
        lp = np.dot(X, beta)

        # If any required input in a row is NaN, make LP NaN
        lp[np.any(np.isnan(X), axis=1)] = np.nan
        return lp

    @staticmethod
    def _cox_risk(lp: np.ndarray, s0: float) -> np.ndarray:
        """
        Convert linear predictor to absolute risk at time horizon using
        Risk = 1 - S0 ** exp(lp).  Handles NaN transparently.
        
        Parameters
        ----------
        lp : np.ndarray
            Linear predictor values
        s0 : float
            Baseline survival
            
        Returns
        -------
        np.ndarray
            Risk values (0-1)
        """
        with np.errstate(over="ignore", invalid="ignore"):
            risk = 1.0 - np.power(s0, np.exp(lp))
        return risk


# --------------------------------------------------------------------- #
#                         Standalone Functions                          #
# --------------------------------------------------------------------- #

def calculate_kfre_4var(age: float, male: bool, egfr: float, acr: float, 
                        years: int = 2, region: str = 'non_north_american') -> float:
    """
    Calculate 4-variable KFRE risk.
    
    Parameters
    ----------
    age : float
        Age in years
    male : bool
        True if male, False if female
    egfr : float
        eGFR in mL/min/1.73 m²
    acr : float
        ACR in mg/g
    years : int, optional
        Time horizon in years (2 or 5)
    region : str, optional
        'north_american' or 'non_north_american'
        
    Returns
    -------
    float
        KFRE risk (0-1)
    """
    # Check for null values
    if pd.isna(age) or pd.isna(male) or pd.isna(egfr) or pd.isna(acr):
        return np.nan
    
    # Convert male boolean to 1/0
    sex = 1 if male else 0
    
    # Calculate ln(ACR)
    lnacr = np.log(max(acr, 1e-10))  # Avoid log(0)
    
    # Use non-North American coefficients (only option in this implementation)
    coef = {
        "age": -0.01985,
        "egfr": -0.09886,
        "lnacr": 0.35066,
        "sex": 0.14842,  # male = 1
    }
    
    # Calculate linear predictor
    lp = (coef["age"] * age + 
          coef["egfr"] * egfr + 
          coef["lnacr"] * lnacr + 
          coef["sex"] * sex)
    
    # Get baseline survival based on years
    if years == 2:
        s0 = 0.9878
    elif years == 5:
        s0 = 0.9570
    else:
        raise ValueError(f"Unsupported time horizon: {years}. Use 2 or 5 years.")
    
    # Calculate risk
    risk = 1.0 - np.power(s0, np.exp(lp))
    
    return risk


def calculate_kfre_8var(age: float, male: bool, egfr: float, acr: float, 
                        calcium: float, phosphate: float, albumin: float, bicarbonate: float,
                        years: int = 2, units: Optional[Dict[str, str]] = None, 
                        region: str = 'non_north_american') -> float:
    """
    Calculate 8-variable KFRE risk.
    
    Parameters
    ----------
    age : float
        Age in years
    male : bool
        True if male, False if female
    egfr : float
        eGFR in mL/min/1.73 m²
    acr : float
        ACR in mg/g
    calcium : float
        Calcium value
    phosphate : float
        Phosphate value
    albumin : float
        Albumin value
    bicarbonate : float
        Bicarbonate value
    years : int, optional
        Time horizon in years (2 or 5)
    units : dict, optional
        Dictionary specifying units for biochemical values
        Keys: 'calcium', 'phosphate', 'albumin', 'bicarbonate'
        Values: unit strings (e.g., 'mmol/L', 'mg/dL')
    region : str, optional
        'north_american' or 'non_north_american'
        
    Returns
    -------
    float
        KFRE risk (0-1)
    """
    # Check for null values
    if (pd.isna(age) or pd.isna(male) or pd.isna(egfr) or pd.isna(acr) or
        pd.isna(calcium) or pd.isna(phosphate) or pd.isna(albumin) or pd.isna(bicarbonate)):
        return np.nan
    
    # Default units dictionary
    if units is None:
        units = {
            'calcium': 'mmol/L',
            'phosphate': 'mmol/L',
            'albumin': 'g/L',
            'bicarbonate': 'mmol/L'
        }
    
    # Convert values to standard units
    calcium_std = convert_calcium(calcium, units.get('calcium'))
    phosphate_std = convert_phosphate(phosphate, units.get('phosphate'))
    albumin_std = convert_albumin(albumin, units.get('albumin'))
    bicarbonate_std = convert_bicarbonate(bicarbonate, units.get('bicarbonate'))
    
    # Convert male boolean to 1/0
    sex = 1 if male else 0
    
    # Calculate ln(ACR)
    lnacr = np.log(max(acr, 1e-10))  # Avoid log(0)
    
    # Use non-North American coefficients (only option in this implementation)
    coef = {
        "age": -0.01985,
        "egfr": -0.09886,
        "lnacr": 0.35066,
        "sex": 0.14842,  # male = 1
        "albumin": -0.03727,
        "phosphate": 0.90638,
        "bicarbonate": -0.07257,
        "calcium": -0.79539,
    }
    
    # Calculate linear predictor
    lp = (coef["age"] * age + 
          coef["egfr"] * egfr + 
          coef["lnacr"] * lnacr + 
          coef["sex"] * sex +
          coef["albumin"] * albumin_std +
          coef["phosphate"] * phosphate_std +
          coef["bicarbonate"] * bicarbonate_std +
          coef["calcium"] * calcium_std)
    
    # Get baseline survival based on years
    if years == 2:
        s0 = 0.9878
    elif years == 5:
        s0 = 0.9570
    else:
        raise ValueError(f"Unsupported time horizon: {years}. Use 2 or 5 years.")
    
    # Calculate risk
    risk = 1.0 - np.power(s0, np.exp(lp))
    
    return risk


def calculate_kfre_risk(df: pd.DataFrame, output_col_prefix: str = 'kfre',
                        age_col: str = 'age', gender_col: str = 'gender', 
                        egfr_col: str = 'egfr', acr_col: str = 'uacr',
                        calcium_col: str = 'calcium', phosphate_col: str = 'phosphate',
                        albumin_col: str = 'albumin', bicarbonate_col: str = 'bicarbonate',
                        male_value: Any = 1, units: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Add KFRE risk columns to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    output_col_prefix : str, optional
        Prefix for output columns
    age_col : str, optional
        Column name for age
    gender_col : str, optional
        Column name for gender/sex
    egfr_col : str, optional
        Column name for eGFR
    acr_col : str, optional
        Column name for ACR
    calcium_col : str, optional
        Column name for calcium
    phosphate_col : str, optional
        Column name for phosphate
    albumin_col : str, optional
        Column name for albumin
    bicarbonate_col : str, optional
        Column name for bicarbonate
    male_value : any, optional
        Value that represents male in the gender column
    units : dict, optional
        Dictionary specifying units for biochemical values
        Keys: 'calcium', 'phosphate', 'albumin', 'bicarbonate'
        Values: unit strings (e.g., 'mmol/L', 'mg/dL')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added KFRE risk columns
    """
    # Create a mapping dictionary
    mapping = {
        'age': age_col,
        'sex': gender_col,
        'egfr': egfr_col,
        'acr': acr_col,
        'albumin': albumin_col,
        'phosphate': phosphate_col,
        'bicarbonate': bicarbonate_col,
        'calcium': calcium_col
    }
    
    # Create a temporary mapping file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(f"kfre: {mapping}")
        temp_file = f.name
    
    try:
        # Create a KFRECalculator with the temporary mapping file
        calculator = KFRECalculator(temp_file)
        
        # Make a copy of the DataFrame
        df_copy = df.copy()
        
        # Convert gender to sex if needed
        if gender_col in df_copy.columns:
            df_copy['sex'] = (df_copy[gender_col] == male_value).astype(int)
        
        # Calculate risks
        result_df = calculator.add_kfre_risk(df_copy, units=units)
        
        # Rename the risk columns if needed
        if output_col_prefix != '':
            rename_dict = {
                '4v2y': f'{output_col_prefix}_4v2y',
                '4v5y': f'{output_col_prefix}_4v5y',
                '8v2y': f'{output_col_prefix}_8v2y',
                '8v5y': f'{output_col_prefix}_8v5y'
            }
            result_df = result_df.rename(columns=rename_dict)
        
        return result_df
    
    finally:
        # Clean up the temporary file
        os.unlink(temp_file)


class KFRE:
    """
    Higher-level class for KFRE calculations with additional functionality.
    """
    
    def __init__(self, mapping_yaml: Optional[Union[str, Path]] = None):
        """
        Initialize the KFRE class.
        
        Parameters
        ----------
        mapping_yaml : str or pathlib.Path, optional
            Path to YAML file that maps user column names to required KFRE names.
            If None, uses a default mapping where column names are the same as KFRE variable names.
        """
        self.df = None
        self.filtered_df = None
        self.mapping_yaml = mapping_yaml
        
        # Create a default mapping if none is provided
        if mapping_yaml is None:
            import tempfile
            import os
            
            default_mapping = {
                'age': 'age',
                'sex': 'sex',
                'egfr': 'egfr',
                'acr': 'acr',
                'albumin': 'albumin',
                'phosphate': 'phosphate',
                'bicarbonate': 'bicarbonate',
                'calcium': 'calcium'
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                f.write(f"kfre: {default_mapping}")
                self.mapping_yaml = f.name
    
    def read_dataframe(self, df: pd.DataFrame) -> None:
        """
        Read a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        """
        self.df = df.copy()
        self.filtered_df = None
    
    def filter_for_dialysis_endpoint(self) -> None:
        """
        Filter the DataFrame for dialysis endpoint.
        
        This method filters the DataFrame to include only rows where the endpoint
        is dialysis (typically coded as 1 in the endpoint column).
        
        The filtered DataFrame is stored in the `filtered_df` attribute.
        """
        if self.df is None:
            raise ValueError("No DataFrame has been read. Call read_dataframe() first.")
        
        # Load the mapping file to get the endpoint column name
        with open(self.mapping_yaml, "r", encoding="utf-8") as fh:
            yaml_content = yaml.safe_load(fh)
        
        # Try to find the endpoint mapping
        endpoint_col = None
        if 'endpoint' in yaml_content:
            endpoint_mapping = yaml_content['endpoint']
            if isinstance(endpoint_mapping, dict) and 'dialysis' in endpoint_mapping:
                dialysis_value = endpoint_mapping['dialysis']
                endpoint_col = 'endpoint'  # Default name
        
        # If we couldn't find the endpoint mapping, look for an 'event' key
        if endpoint_col is None and 'event' in yaml_content:
            endpoint_col = yaml_content['event']
            dialysis_value = 1  # Default value for dialysis
        
        # If we still couldn't find it, use default values
        if endpoint_col is None:
            endpoint_col = 'endpoint'
            dialysis_value = 1
        
        # Filter the DataFrame
        if endpoint_col in self.df.columns:
            self.filtered_df = self.df[self.df[endpoint_col] == dialysis_value].copy()
        else:
            raise ValueError(f"Endpoint column '{endpoint_col}' not found in DataFrame.")
    
    def calculate_risks(self, output_col_prefix: str = 'kfre') -> pd.DataFrame:
        """
        Calculate KFRE risks.
        
        Parameters
        ----------
        output_col_prefix : str, optional
            Prefix for output columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added KFRE risk columns
        """
        if self.df is None:
            raise ValueError("No DataFrame has been read. Call read_dataframe() first.")
        
        # Use the filtered DataFrame if available, otherwise use the original
        df_to_use = self.filtered_df if self.filtered_df is not None else self.df
        
        # Create a KFRECalculator with the mapping file
        calculator = KFRECalculator(self.mapping_yaml)
        
        # Calculate risks
        result_df = calculator.add_kfre_risk(df_to_use)
        
        # Rename the risk columns if needed
        if output_col_prefix != '':
            rename_dict = {
                '4v2y': f'{output_col_prefix}_4v2y',
                '4v5y': f'{output_col_prefix}_4v5y',
                '8v2y': f'{output_col_prefix}_8v2y',
                '8v5y': f'{output_col_prefix}_8v5y'
            }
            result_df = result_df.rename(columns=rename_dict)
        
        return result_df


# ------------------------------------------------------------------------- #
#                           Convenience CLI                                 #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch KFRE calculation.")
    parser.add_argument("csv_in", help="Path to input CSV.")
    parser.add_argument("yaml_map", help="Path to YAML column mapping.")
    parser.add_argument("csv_out", help="Path to output CSV with KFRE risks.")
    args = parser.parse_args()

    calculator = KFRECalculator(args.yaml_map)
    df_in = pd.read_csv(args.csv_in)
    df_out = calculator.add_kfre_risk(df_in)
    df_out.to_csv(args.csv_out, index=False)
    print(f"Saved with KFRE columns to {args.csv_out}")