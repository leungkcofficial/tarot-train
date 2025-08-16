import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Typography,
  Alert,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  InputAdornment,
  Divider,
  Tooltip,
  IconButton,
  LinearProgress,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import {
  Biotech,
  ExpandMore,
  Info,
  Calculate,
  Timeline,
  Warning,
  CheckCircle,
  Error,
  Add,
  Delete
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useSession, LabValue } from '../../contexts/SessionContext';
import { formatDate } from '../../services/api';

/**
 * Laboratory values step component with unit conversion and validation
 */
const LabValuesStep: React.FC = () => {
  const { state, addLabValue, updateLabValue: updateLabValueContext, removeLabValue } = useSession();
  const { demographics, labValues } = state.data;
  const [validationResults, setValidationResults] = useState<any>(null);
  const [isValidating, setIsValidating] = useState(false);

  // Required laboratory parameters
  const labParameters = [
    {
      parameter: 'creatinine',
      label: 'Serum Creatinine',
      units: ['µmol/L', 'mg/dL'],
      defaultUnit: 'µmol/L',
      normalRange: '53-115 µmol/L',
      description: 'Used for eGFR calculation (CKD-EPI 2021)',
      required: true,
      icon: <Calculate />
    },
    {
      parameter: 'hemoglobin',
      label: 'Hemoglobin',
      units: ['g/dL', 'g/L'],
      defaultUnit: 'g/dL',
      normalRange: '12-16 g/dL',
      description: 'Anemia marker in CKD progression',
      required: true,
      icon: <Timeline />
    },
    {
      parameter: 'phosphate',
      label: 'Serum Phosphate',
      units: ['mmol/L', 'mg/dL'],
      defaultUnit: 'mmol/L',
      normalRange: '0.8-1.5 mmol/L',
      description: 'Mineral metabolism marker',
      required: true,
      icon: <Biotech />
    },
    {
      parameter: 'bicarbonate',
      label: 'Serum Bicarbonate',
      units: ['mEq/L', 'mmol/L'],
      defaultUnit: 'mmol/L',
      normalRange: '22-28 mEq/L',
      description: 'Acid-base balance marker',
      required: true,
      icon: <Timeline />
    },
    {
      parameter: 'albumin',
      label: 'Serum Albumin',
      units: ['g/L', 'g/dL'],
      defaultUnit: 'g/L',
      normalRange: '35-50 g/L',
      description: 'Nutritional and synthetic marker',
      required: true,
      icon: <Biotech />
    },
    {
      parameter: 'uacr',
      label: 'UACR (Urine Albumin-to-Creatinine Ratio)',
      units: ['mg/g', 'mg/mmol'],
      defaultUnit: 'mg/g',
      normalRange: '<30 mg/g',
      description: 'Preferred proteinuria marker (takes priority over UPCR)',
      required: false, // Made optional since UPCR can substitute
      icon: <Biotech />,
      specialRequired: 'uacr_or_upcr'
    },
    {
      parameter: 'upcr',
      label: 'UPCR (Urine Protein-to-Creatinine Ratio)',
      units: ['mg/g', 'g/g'],
      defaultUnit: 'mg/g',
      normalRange: '<150 mg/g',
      description: 'Alternative proteinuria marker (used if UACR not available)',
      required: false,
      icon: <Biotech />,
      specialRequired: 'uacr_or_upcr'
    }
  ];

  // Get current values for a parameter (supporting multiple time points)
  const getCurrentValues = (parameter: string) => {
    return labValues.filter(lab => lab.parameter === parameter).sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  };
  
  // Get most recent value for a parameter
  const getCurrentValue = (parameter: string) => {
    const values = getCurrentValues(parameter);
    return values.length > 0 ? values[0] : undefined;
  };

  // Update lab value
  const updateLabValue = (parameter: string, updates: Partial<LabValue>) => {
    const existing = labValues.find(lab => lab.parameter === parameter);
    
    if (existing) {
      updateLabValueContext(existing.id, updates);
    } else {
      const newLabValue: Omit<LabValue, 'id'> = {
        parameter,
        value: '',
        unit: labParameters.find(p => p.parameter === parameter)?.defaultUnit || '',
        date: new Date(),
        ...updates
      };
      addLabValue(newLabValue);
    }
  };

  // Handle value change
  const handleValueChange = (parameter: string, value: string) => {
    const numericValue = value === '' ? '' : parseFloat(value);
    updateLabValue(parameter, { value: numericValue });
  };

  // Handle unit change
  const handleUnitChange = (parameter: string, unit: string) => {
    updateLabValue(parameter, { unit });
  };

  // Handle date change
  const handleDateChange = (parameter: string, date: Date | null) => {
    if (date) {
      updateLabValue(parameter, { date });
    }
  };

  // Add new lab entry for a parameter
  const addNewLabEntry = (parameter: string) => {
    const paramConfig = labParameters.find(p => p.parameter === parameter);
    const newLabValue: Omit<LabValue, 'id'> = {
      parameter,
      value: '',
      unit: paramConfig?.defaultUnit || '',
      date: new Date(),
    };
    addLabValue(newLabValue);
  };

  // Update specific lab entry
  const updateSpecificLabEntry = (id: string, updates: Partial<LabValue>) => {
    updateLabValueContext(id, updates);
  };

  // Remove specific lab entry
  const removeSpecificLabEntry = (id: string) => {
    removeLabValue(id);
  };

  // Validate inputs periodically
  useEffect(() => {
    const validateInputs = async () => {
      if (demographics.age && demographics.gender && labValues.length > 0) {
        try {
          setIsValidating(true);
          // TODO: Implement validation API call if needed
          // For now, just clear validation state
          setValidationResults(null);
        } catch (error) {
          console.error('Validation error:', error);
        } finally {
          setIsValidating(false);
        }
      }
    };

    const timer = setTimeout(validateInputs, 1000);
    return () => clearTimeout(timer);
  }, [labValues, demographics]);

  // Check if required parameters are complete
  const requiredParams = labParameters.filter(p => p.required);
  const completedRequired = requiredParams.filter(param => {
    const values = getCurrentValues(param.parameter);
    return values.some(value => value && value.value !== '' && value.value !== 0);
  });

  // Check for special requirements (UACR OR UPCR)
  const hasUacrOrUpcr = 
    getCurrentValues('uacr').some(value => value && value.value !== '' && value.value !== 0) ||
    getCurrentValues('upcr').some(value => value && value.value !== '' && value.value !== 0);

  const isStepComplete = completedRequired.length === requiredParams.length && hasUacrOrUpcr;

  // Get validation status for a parameter
  const getValidationStatus = (parameter: string) => {
    if (!validationResults) return null;
    
    const processedValue = validationResults.processed_values?.find(
      (pv: any) => pv.parameter === parameter
    );
    
    if (processedValue?.warnings?.length > 0) {
      return 'warning';
    }
    
    const error = validationResults.errors?.find((err: string) => 
      err.toLowerCase().includes(parameter)
    );
    
    if (error) return 'error';
    
    return 'success';
  };

  // Calculate eGFR if creatinine is available
  const calculateEGFR = () => {
    if (validationResults?.egfr_info) {
      return validationResults.egfr_info;
    }
    return null;
  };

  // Render multiple entries for a parameter
  const renderParameterEntries = (param: any) => {
    const currentValues = getCurrentValues(param.parameter);
    const validationStatus = getValidationStatus(param.parameter);
    
    return (
      <Grid item xs={12} key={param.parameter}>
        <Card variant="outlined">
          <CardContent>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              {param.icon}
              <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
                {param.label}
              </Typography>
              <Tooltip title={param.description}>
                <IconButton size="small">
                  <Info fontSize="small" />
                </IconButton>
              </Tooltip>
              {validationStatus === 'success' && <CheckCircle color="success" fontSize="small" />}
              {validationStatus === 'warning' && <Warning color="warning" fontSize="small" />}
              {validationStatus === 'error' && <Error color="error" fontSize="small" />}
              {param.required && <Chip label="Required" size="small" color="error" />}
              {param.specialRequired === 'uacr_or_upcr' && <Chip label="UACR or UPCR" size="small" color="warning" />}
            </Box>
            
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
              Normal range: {param.normalRange} | {param.description}
            </Typography>
            
            {/* Existing entries */}
            {currentValues.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  Recorded Values ({currentValues.length}):
                </Typography>
                <List dense>
                  {currentValues.map((labValue, index) => (
                    <ListItem key={labValue.id} sx={{ px: 0 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={3}>
                          <TextField
                            label="Value"
                            type="number"
                            size="small"
                            value={labValue.value || ''}
                            onChange={(e) => updateSpecificLabEntry(labValue.id, { value: parseFloat(e.target.value) || '' })}
                            inputProps={{ step: 'any', min: 0 }}
                            error={validationStatus === 'error'}
                          />
                        </Grid>
                        <Grid item xs={3}>
                          <FormControl size="small" fullWidth>
                            <InputLabel>Unit</InputLabel>
                            <Select
                              value={labValue.unit}
                              onChange={(e) => updateSpecificLabEntry(labValue.id, { unit: e.target.value })}
                              label="Unit"
                            >
                              {param.units.map((unit: string) => (
                                <MenuItem key={unit} value={unit}>
                                  {unit}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </Grid>
                        <Grid item xs={4}>
                          <DatePicker
                            label="Date"
                            value={new Date(labValue.date)}
                            onChange={(date) => date && updateSpecificLabEntry(labValue.id, { date })}
                            maxDate={new Date()}
                            slotProps={{
                              textField: {
                                size: 'small',
                                fullWidth: true
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={2}>
                          <IconButton
                            onClick={() => removeSpecificLabEntry(labValue.id)}
                            disabled={currentValues.length === 1 && param.required}
                            color="error"
                            size="small"
                          >
                            <Delete />
                          </IconButton>
                        </Grid>
                      </Grid>
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
            
            {/* Add new entry button */}
            <Button
              startIcon={<Add />}
              onClick={() => addNewLabEntry(param.parameter)}
              variant="outlined"
              size="small"
              sx={{ mt: 1 }}
            >
              Add {currentValues.length === 0 ? 'Value' : 'Another Time Point'}
            </Button>
            
            {currentValues.length === 0 && param.required && (
              <Alert severity="error" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  This parameter is required. Please add at least one value.
                </Typography>
              </Alert>
            )}
          </CardContent>
        </Card>
      </Grid>
    );
  };

  const egfrInfo = calculateEGFR();

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box className="form-section">
        <Typography variant="h5" className="form-section-title">
          <Biotech />
          Laboratory Values
          <Chip label="Required" size="small" color="error" sx={{ ml: 2 }} />
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Enter recent laboratory results (within the last 6 months). You can add multiple time points for each parameter. 
          All required values must be provided, with at least one entry per parameter.
        </Typography>

        {/* Progress indicator */}
        {isValidating && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Validating inputs...
            </Typography>
            <LinearProgress />
          </Box>
        )}

        {/* eGFR Calculation Result */}
        {egfrInfo && (
          <Alert
            severity={egfrInfo.egfr >= 60 ? 'error' : egfrInfo.egfr >= 30 ? 'warning' : 'info'}
            sx={{ mb: 3 }}
          >
            <Typography variant="body2">
              <strong>Calculated eGFR:</strong> {egfrInfo.egfr} mL/min/1.73m² 
              <Chip label={egfrInfo.egfr_stage} size="small" sx={{ ml: 1 }} />
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              {egfrInfo.message}
            </Typography>
          </Alert>
        )}

        {/* Laboratory Parameters */}
        <Box sx={{ mb: 3 }}>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Laboratory Parameters
            </Typography>
            <Chip
              label={`${completedRequired.length}/${requiredParams.length} Required Complete`}
              color={completedRequired.length === requiredParams.length ? 'success' : 'warning'}
              size="small"
            />
          </Box>
          
          <Grid container spacing={3}>
            {labParameters.map((param) => renderParameterEntries(param))}
          </Grid>
        </Box>


        {/* Validation Results */}
        {validationResults && validationResults.warnings?.length > 0 && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Input Warnings:
            </Typography>
            {validationResults.warnings.map((warning: any, index: number) => (
              <Typography key={index} variant="body2">
                • {warning.field}: {warning.message}
              </Typography>
            ))}
          </Alert>
        )}

        {/* Summary */}
        <Box sx={{ mt: 4, p: 2, bgcolor: isStepComplete ? 'success.50' : 'warning.50', borderRadius: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Laboratory Values Summary:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            <Chip
              label={`Required Parameters: ${completedRequired.length}/${requiredParams.length}`}
              color={completedRequired.length === requiredParams.length ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={`UACR or UPCR: ${hasUacrOrUpcr ? 'Provided' : 'Missing'}`}
              color={hasUacrOrUpcr ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={`Total Entries: ${labValues.length}`}
              color="info"
              size="small"
            />
            <Chip
              label={isStepComplete ? 'Ready to proceed' : 'Incomplete'}
              color={isStepComplete ? 'success' : 'warning'}
              size="small"
              sx={{ fontWeight: 600 }}
            />
          </Box>
        </Box>

        {/* Next Steps Preview */}
        {isStepComplete && (
          <Alert severity="info" sx={{ mt: 3 }}>
            <Typography variant="body2">
              <strong>Next:</strong> Medical history (comorbidities and diagnosis dates) will be collected to calculate 
              the Charlson Comorbidity Index and improve prediction accuracy.
            </Typography>
          </Alert>
        )}
      </Box>
    </LocalizationProvider>
  );
};

export default LabValuesStep;