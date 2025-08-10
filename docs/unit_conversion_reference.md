# Unit Conversion Reference for CKD Preprocessing

This document provides unit conversion references for users who may have data in different units than those used in the CKD preprocessing pipeline.

## Laboratory Values - Expected Units and Conversions

### 1. Creatinine
- **Expected Unit**: **µmol/L** (micromoles per liter)
- **Common Alternative**: mg/dL (milligrams per deciliter)
- **Conversion**: 
  - mg/dL → µmol/L: multiply by **88.4**
  - µmol/L → mg/dL: divide by **88.4**
- **Example**: 1.5 mg/dL = 132.6 µmol/L

### 2. Hemoglobin
- **Expected Unit**: **g/dL** (grams per deciliter)
- **Common Alternative**: g/L (grams per liter)
- **Conversion**: 
  - g/L → g/dL: divide by **10**
  - g/dL → g/L: multiply by **10**
- **Example**: 120 g/L = 12.0 g/dL

### 3. Albumin
- **Expected Unit**: **g/L** (grams per liter)
- **Common Alternative**: g/dL (grams per deciliter)
- **Conversion**: 
  - g/dL → g/L: multiply by **10**
  - g/L → g/dL: divide by **10**
- **Example**: 3.8 g/dL = 38 g/L

### 4. Phosphate
- **Expected Unit**: **mmol/L** (millimoles per liter)
- **Common Alternative**: mg/dL (milligrams per deciliter)
- **Conversion**: 
  - mg/dL → mmol/L: multiply by **0.3229**
  - mmol/L → mg/dL: divide by **0.3229**
- **Example**: 4.0 mg/dL = 1.29 mmol/L

### 5. Calcium
- **Expected Unit**: **mmol/L** (millimoles per liter)
- **Common Alternative**: mg/dL (milligrams per deciliter)
- **Conversion**: 
  - mg/dL → mmol/L: multiply by **0.2495**
  - mmol/L → mg/dL: divide by **0.2495**
- **Example**: 9.2 mg/dL = 2.30 mmol/L

### 6. Bicarbonate
- **Expected Unit**: **mmol/L** (millimoles per liter)
- **Alternative Unit**: mEq/L (milliequivalents per liter)
- **Note**: For bicarbonate, mmol/L = mEq/L (same value)

### 7. Other Values (No Conversion Needed)
- **A1c**: % (percentage)
- **UPCR**: g/g (grams per gram)
- **UACR**: mg/g (milligrams per gram)
- **eGFR**: mL/min/1.73m² (milliliters per minute per 1.73 square meters)

## Quick Reference Table

| Lab Test | Expected Unit | If You Have | Multiply By |
|----------|--------------|-------------|-------------|
| Creatinine | µmol/L | mg/dL | 88.4 |
| Hemoglobin | g/dL | g/L | 0.1 |
| Albumin | g/L | g/dL | 10 |
| Phosphate | mmol/L | mg/dL | 0.3229 |
| Calcium | mmol/L | mg/dL | 0.2495 |
| Bicarbonate | mmol/L | mEq/L | 1 (same) |

## Example Conversion

If your lab report shows:
- Creatinine: 1.7 mg/dL
- Albumin: 3.5 g/dL
- Phosphate: 3.8 mg/dL
- Calcium: 9.5 mg/dL

Convert to expected units:
```python
patient_data = {
    'creatinine': 1.7 * 88.4,    # = 150.3 µmol/L
    'albumin': 3.5 * 10,          # = 35 g/L
    'phosphate': 3.8 * 0.3229,    # = 1.23 mmol/L
    'calcium': 9.5 * 0.2495,      # = 2.37 mmol/L
    # ... other values
}
```

## Important Notes

1. **Always verify units** from your laboratory reports before conversion
2. **Different labs may use different units** - check the report carefully
3. **The preprocessor expects the units shown above** - incorrect units will lead to incorrect predictions
4. **When in doubt**, consult with healthcare professionals about unit conversions

## Regional Variations

- **North America**: Often uses mg/dL for creatinine, albumin, calcium, phosphate
- **Europe/Asia**: Often uses SI units (µmol/L, mmol/L, g/L)
- **UK**: Mixed usage depending on the laboratory

Always check your local laboratory's reporting units!