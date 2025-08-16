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
  LinearProgress
} from '@mui/material';
import {
  Biotech,
  ExpandMore,
  Info,
  Calculate,
  Timeline,
  Warning,
  CheckCircle,
  Error
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
      units: ['mg/dL', 'µmol/L'],
      defaultUnit: 'mg/dL',
      normalRange: '0.6-1.4 mg/dL',
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
      units: ['mg/dL', 'mmol/L'],
      defaultUnit: 'mg/dL',
      normalRange: '2.5-4.5 mg/dL',
      description: 'Mineral metabolism marker',
      required: true,
      icon: <Biotech />
    },
    {
      parameter: 'bicarbonate',
      label: 'Serum Bicarbonate',
      units: ['mEq/L', 'mmol/L'],
      defaultUnit: 'mEq/L',
      normalRange: '22-28 mEq/L',
      description: 'Acid-base balance marker',
      required: true,
      icon: <Timeline />
    },
    {
      parameter: 'uacr',
      label: 'UACR (Urine Albumin-to-Creatinine Ratio)',
      units: ['mg/g', 'mg/mmol'],
      defaultUnit: 'mg/g',
      normalRange: '<30 mg/g',
      description: 'Preferred proteinuria marker',
      required: false,
      icon: <Biotech />
    },
    {
      parameter: 'upcr',
      label: 'UPCR (Urine Protein-to-Creatinine Ratio)',
      units: ['mg/g', 'g/g'],
      defaultUnit: 'mg/g',
      normalRange: '<150 mg/g',
      description: 'Alternative proteinuria marker (converted to UACR)',
      required: false,
      icon: <Biotech />
    }
  ];

  // Get current value for a parameter
  const getCurrentValue = (parameter: string) => {
    return labValues.find(lab => lab.parameter === parameter);
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
    const value = getCurrentValue(param.parameter);
    return value && value.value !== '' && value.value !== 0;
  });

  // Check if at least one urine parameter is provided
  const urineParams = labParameters.filter(p => ['uacr', 'upcr'].includes(p.parameter));
  const hasUrineParam = urineParams.some(param => {
    const value = getCurrentValue(param.parameter);
    return value && value.value !== '' && value.value !== 0;
  });

  const isStepComplete = completedRequired.length === requiredParams.length && hasUrineParam;

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
          Enter recent laboratory results (within the last 6 months). All required values and at least one urine parameter (UACR or UPCR) must be provided.
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

        {/* Required Parameters */}
        <Accordion defaultExpanded sx={{ mb: 3 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Required Laboratory Tests
              </Typography>
              <Chip
                label={`${completedRequired.length}/${requiredParams.length} Complete`}
                color={completedRequired.length === requiredParams.length ? 'success' : 'warning'}
                size="small"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {requiredParams.map((param) => {
                const currentValue = getCurrentValue(param.parameter);
                const validationStatus = getValidationStatus(param.parameter);
                
                return (
                  <Grid item xs={12} md={6} key={param.parameter}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" gap={1} mb={2}>
                          {param.icon}
                          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
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
                        </Box>
                        
                        <Box display="flex" gap={2} mb={2}>
                          <TextField
                            label="Value"
                            type="number"
                            value={currentValue?.value || ''}
                            onChange={(e) => handleValueChange(param.parameter, e.target.value)}
                            inputProps={{ step: 'any', min: 0 }}
                            sx={{ flex: 1 }}
                            error={validationStatus === 'error'}
                          />
                          
                          <FormControl sx={{ minWidth: 100 }}>
                            <InputLabel>Unit</InputLabel>
                            <Select
                              value={currentValue?.unit || param.defaultUnit}
                              onChange={(e) => handleUnitChange(param.parameter, e.target.value)}
                              label="Unit"
                            >
                              {param.units.map((unit) => (
                                <MenuItem key={unit} value={unit}>
                                  {unit}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </Box>
                        
                        <DatePicker
                          label="Test Date"
                          value={currentValue?.date ? new Date(currentValue.date) : new Date()}
                          onChange={(date) => handleDateChange(param.parameter, date)}
                          maxDate={new Date()}
                          slotProps={{
                            textField: {
                              size: 'small',
                              fullWidth: true
                            }
                          }}
                        />
                        
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Normal range: {param.normalRange}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Urine Parameters */}
        <Accordion sx={{ mb: 3 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Urine Protein Assessment
              </Typography>
              <Chip
                label={hasUrineParam ? 'Complete' : 'Select One'}
                color={hasUrineParam ? 'success' : 'warning'}
                size="small"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Alert severity="info" sx={{ mb: 3 }}>
              <Typography variant="body2">
                <strong>Choose one:</strong> UACR is preferred. If only UPCR is available, it will be converted to UACR using our validated formula.
              </Typography>
            </Alert>
            
            <Grid container spacing={3}>
              {urineParams.map((param) => {
                const currentValue = getCurrentValue(param.parameter);
                const validationStatus = getValidationStatus(param.parameter);
                
                return (
                  <Grid item xs={12} md={6} key={param.parameter}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" gap={1} mb={2}>
                          {param.icon}
                          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
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
                        </Box>
                        
                        <Box display="flex" gap={2} mb={2}>
                          <TextField
                            label="Value"
                            type="number"
                            value={currentValue?.value || ''}
                            onChange={(e) => handleValueChange(param.parameter, e.target.value)}
                            inputProps={{ step: 'any', min: 0 }}
                            sx={{ flex: 1 }}
                            error={validationStatus === 'error'}
                          />
                          
                          <FormControl sx={{ minWidth: 100 }}>
                            <InputLabel>Unit</InputLabel>
                            <Select
                              value={currentValue?.unit || param.defaultUnit}
                              onChange={(e) => handleUnitChange(param.parameter, e.target.value)}
                              label="Unit"
                            >
                              {param.units.map((unit) => (
                                <MenuItem key={unit} value={unit}>
                                  {unit}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </Box>
                        
                        <DatePicker
                          label="Test Date"
                          value={currentValue?.date ? new Date(currentValue.date) : new Date()}
                          onChange={(date) => handleDateChange(param.parameter, date)}
                          maxDate={new Date()}
                          slotProps={{
                            textField: {
                              size: 'small',
                              fullWidth: true
                            }
                          }}
                        />
                        
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Normal range: {param.normalRange}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </AccordionDetails>
        </Accordion>

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
              label={`Required Tests: ${completedRequired.length}/${requiredParams.length}`}
              color={completedRequired.length === requiredParams.length ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={hasUrineParam ? 'Urine Protein: Provided' : 'Urine Protein: Missing'}
              color={hasUrineParam ? 'success' : 'error'}
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
              <strong>Next:</strong> Medical history (comorbidities) can be added to improve prediction accuracy, 
              or you can proceed directly to risk analysis.
            </Typography>
          </Alert>
        )}
      </Box>
    </LocalizationProvider>
  );
};

export default LabValuesStep;