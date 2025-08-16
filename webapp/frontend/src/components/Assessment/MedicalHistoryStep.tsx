import React, { useState } from 'react';
import {
  Box,
  Typography,
  Alert,
  Chip,
  Card,
  CardContent,
  Grid,
  FormControlLabel,
  Checkbox,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Tooltip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  TextField
} from '@mui/material';
import {
  LocalHospital,
  ExpandMore,
  Info,
  FavoriteOutlined,
  Healing,
  Psychology,
  BloodtypeOutlined,
  VisibilityOutlined,
  CheckCircle,
  Warning,
  MonitorHeart
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useSession, Comorbidity } from '../../contexts/SessionContext';

/**
 * Medical history step component for collecting comorbidities
 */
const MedicalHistoryStep: React.FC = () => {
  const { state, addComorbidity, updateComorbidity, removeComorbidity } = useSession();
  const { comorbidities } = state.data;

  // Hypertension as separate feature (not part of CCI but required for model)
  const hypertensionCondition = { id: 'hypertension', label: 'Hypertension', required: true };

  // Comorbidity categories based on Charlson Comorbidity Index (excluding hypertension)
  const comorbidityCategories = [
    {
      category: 'Cardiovascular',
      icon: <FavoriteOutlined color="error" />,
      conditions: [
        { id: 'myocardial_infarction', label: 'Myocardial Infarction (Heart Attack)', weight: 1 },
        { id: 'congestive_heart_failure', label: 'Congestive Heart Failure', weight: 1 },
        { id: 'peripheral_vascular_disease', label: 'Peripheral Vascular Disease', weight: 1 },
        { id: 'cerebrovascular_disease', label: 'Cerebrovascular Disease (Stroke/TIA)', weight: 1 }
      ]
    },
    {
      category: 'Metabolic & Endocrine',
      icon: <BloodtypeOutlined color="warning" />,
      conditions: [
        { id: 'diabetes_uncomplicated', label: 'Diabetes Mellitus (uncomplicated)', weight: 1 },
        { id: 'diabetes_complicated', label: 'Diabetes with Complications', weight: 2 }
      ]
    },
    {
      category: 'Respiratory & Other',
      icon: <Healing color="info" />,
      conditions: [
        { id: 'chronic_pulmonary_disease', label: 'Chronic Pulmonary Disease (COPD)', weight: 1 },
        { id: 'connective_tissue_disease', label: 'Connective Tissue Disease', weight: 1 },
        { id: 'peptic_ulcer_disease', label: 'Peptic Ulcer Disease', weight: 1 }
      ]
    },
    {
      category: 'Hepatic',
      icon: <VisibilityOutlined color="secondary" />,
      conditions: [
        { id: 'liver_disease_mild', label: 'Mild Liver Disease', weight: 1 },
        { id: 'liver_disease_severe', label: 'Severe Liver Disease', weight: 3 }
      ]
    },
    {
      category: 'Malignancy',
      icon: <Psychology color="error" />,
      conditions: [
        { id: 'malignancy', label: 'Malignancy (any tumor)', weight: 2 },
        { id: 'metastatic_cancer', label: 'Metastatic Solid Tumor', weight: 6 }
      ]
    },
    {
      category: 'Other Significant',
      icon: <LocalHospital color="primary" />,
      conditions: [
        { id: 'dementia', label: 'Dementia', weight: 1 },
        { id: 'hemiplegia', label: 'Hemiplegia or Paraplegia', weight: 2 },
        { id: 'aids', label: 'AIDS/HIV', weight: 6 }
      ]
    }
  ];

  // Get current selection status
  const isConditionSelected = (conditionId: string): boolean => {
    return comorbidities.some((condition: Comorbidity) => condition.condition === conditionId && condition.diagnosed);
  };

  // Toggle condition selection
  const toggleCondition = (conditionId: string, label: string, weight: number) => {
    const existing = comorbidities.find((condition: Comorbidity) => condition.condition === conditionId);
    
    if (existing) {
      // Update existing condition
      updateComorbidity(existing.id, {
        diagnosed: !existing.diagnosed
      });
    } else {
      // Add new condition
      const newCondition: Omit<Comorbidity, 'id'> = {
        condition: conditionId,
        diagnosed: true,
        diagnosisDate: new Date()
      };
      addComorbidity(newCondition);
    }
  };

  // Update diagnosis date
  const updateDiagnosisDate = (conditionId: string, date: Date | null) => {
    const existing = comorbidities.find((condition: Comorbidity) => condition.condition === conditionId);
    if (existing && date) {
      updateComorbidity(existing.id, {
        diagnosisDate: date
      });
    }
  };

  // Calculate Charlson Comorbidity Index score (excluding hypertension)
  const calculateCharlsonScore = (): number => {
    return comorbidities
      .filter((condition: Comorbidity) => condition.diagnosed && condition.condition !== 'hypertension')
      .reduce((total, condition) => {
        // Find weight from comorbidityCategories based on condition
        for (const category of comorbidityCategories) {
          const found = category.conditions.find(c => c.id === condition.condition);
          if (found) {
            return total + found.weight;
          }
        }
        return total;
      }, 0);
  };

  // Get selected conditions count with details
  const selectedConditions = comorbidities
    .filter((condition: Comorbidity) => condition.diagnosed)
    .map((condition: Comorbidity) => {
      // Find the condition details from comorbidityCategories
      for (const category of comorbidityCategories) {
        const found = category.conditions.find(c => c.id === condition.condition);
        if (found) {
          return {
            ...condition,
            label: found.label,
            weight: found.weight
          };
        }
      }
      return {
        ...condition,
        label: condition.condition,
        weight: 0
      };
    });
  const charlsonScore = calculateCharlsonScore();

  // Risk interpretation based on Charlson score
  const getRiskInterpretation = (score: number) => {
    if (score === 0) return { level: 'Low', color: 'success', description: 'No major comorbidities' };
    if (score <= 2) return { level: 'Moderate', color: 'info', description: 'Mild comorbidity burden' };
    if (score <= 4) return { level: 'High', color: 'warning', description: 'Moderate comorbidity burden' };
    return { level: 'Very High', color: 'error', description: 'Severe comorbidity burden' };
  };

  const riskInterpretation = getRiskInterpretation(charlsonScore);
  const hasHypertension = isConditionSelected('hypertension');
  const hypertensionData = comorbidities.find(c => c.condition === 'hypertension');

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box className="form-section">
        <Typography variant="h5" className="form-section-title">
          <LocalHospital />
          Medical History
          <Chip label="Required" size="small" color="error" sx={{ ml: 2 }} />
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Select diagnosed medical conditions and provide diagnosis dates. Hypertension status is required for model accuracy.
          Other conditions improve prediction accuracy using the Charlson Comorbidity Index.
        </Typography>

        {/* Hypertension Section (Required) */}
        <Card sx={{ mb: 3, bgcolor: hasHypertension ? 'success.50' : 'warning.50' }}>
          <CardContent>
            <Box display="flex" alignItems="center" gap={2} mb={2}>
              <MonitorHeart color="primary" />
              <Typography variant="h6" sx={{ fontWeight: 600, flex: 1 }}>
                Hypertension Status
              </Typography>
              <Chip label="Required" size="small" color="error" />
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Hypertension status is required as a separate model feature (not included in Charlson score).
            </Typography>
            
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={hasHypertension}
                      onChange={() => toggleCondition('hypertension', 'Hypertension', 0)}
                    />
                  }
                  label={
                    <Typography variant="body1" sx={{ fontWeight: hasHypertension ? 600 : 400 }}>
                      Patient has diagnosed hypertension
                    </Typography>
                  }
                />
              </Grid>
              
              {hasHypertension && (
                <Grid item xs={12} md={6}>
                  <DatePicker
                    label="Diagnosis Date"
                    value={hypertensionData?.diagnosisDate || null}
                    onChange={(date) => updateDiagnosisDate('hypertension', date)}
                    maxDate={new Date()}
                    slotProps={{
                      textField: {
                        fullWidth: true,
                        size: 'small',
                        helperText: 'When was hypertension first diagnosed?'
                      }
                    }}
                  />
                </Grid>
              )}
            </Grid>
            
            {hasHypertension && (
              <Alert severity="success" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  ✓ Hypertension status recorded - Required information complete
                </Typography>
              </Alert>
            )}
            
            {!hasHypertension && (
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  Please indicate hypertension status to continue
                </Typography>
              </Alert>
            )}
          </CardContent>
        </Card>

      {/* Current Status Summary */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
          <Typography variant="body2">
            <strong>Comorbidity Assessment:</strong> {selectedConditions.length} conditions selected
          </Typography>
          <Box display="flex" gap={1} alignItems="center">
            <Chip 
              label={`Charlson Score: ${charlsonScore}`} 
              color={riskInterpretation.color as any}
              size="small" 
            />
            <Chip 
              label={`${riskInterpretation.level} Risk`} 
              color={riskInterpretation.color as any}
              variant="outlined" 
              size="small" 
            />
          </Box>
        </Box>
      </Alert>

      {/* Score Interpretation */}
      {charlsonScore > 0 && (
        <Alert severity={riskInterpretation.color as any} sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Charlson Comorbidity Index:</strong> {charlsonScore} points - {riskInterpretation.description}
            <br />
            Higher scores indicate increased mortality risk and may influence CKD progression predictions.
          </Typography>
        </Alert>
      )}

      {/* Comorbidity Categories */}
      {comorbidityCategories.map((category, categoryIndex) => (
        <Accordion key={category.category} sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box display="flex" alignItems="center" gap={2}>
              {category.icon}
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {category.category}
              </Typography>
              <Chip
                label={`${category.conditions.filter(c => isConditionSelected(c.id)).length}/${category.conditions.length}`}
                color={category.conditions.some(c => isConditionSelected(c.id)) ? 'primary' : 'default'}
                size="small"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {category.conditions.map((condition) => {
                const isSelected = isConditionSelected(condition.id);
                const conditionData = comorbidities.find(c => c.condition === condition.id);
                
                return (
                  <Grid item xs={12} key={condition.id}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        bgcolor: isSelected ? 'primary.50' : 'inherit',
                        borderColor: isSelected ? 'primary.main' : 'divider'
                      }}
                    >
                      <CardContent>
                        <Grid container spacing={2} alignItems="center">
                          <Grid item xs={12} md={6}>
                            <Box display="flex" alignItems="center" gap={2}>
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={isSelected}
                                    onChange={() => toggleCondition(condition.id, condition.label, condition.weight)}
                                  />
                                }
                                label=""
                                sx={{ m: 0 }}
                              />
                              
                              <Box flex={1}>
                                <Typography 
                                  variant="body2" 
                                  sx={{ 
                                    fontWeight: isSelected ? 600 : 400,
                                    mb: 1 
                                  }}
                                >
                                  {condition.label}
                                </Typography>
                                
                                <Box display="flex" alignItems="center" gap={1}>
                                  <Chip
                                    label={`${condition.weight} point${condition.weight > 1 ? 's' : ''}`}
                                    size="small"
                                    color={condition.weight >= 3 ? 'error' : condition.weight >= 2 ? 'warning' : 'info'}
                                  />
                                  {isSelected && (
                                    <CheckCircle color="success" fontSize="small" />
                                  )}
                                </Box>
                              </Box>
                            </Box>
                          </Grid>
                          
                          {isSelected && (
                            <Grid item xs={12} md={6}>
                              <DatePicker
                                label="Diagnosis Date"
                                value={conditionData?.diagnosisDate || null}
                                onChange={(date) => updateDiagnosisDate(condition.id, date)}
                                maxDate={new Date()}
                                slotProps={{
                                  textField: {
                                    size: 'small',
                                    fullWidth: true,
                                    helperText: 'When was this condition diagnosed?'
                                  }
                                }}
                              />
                            </Grid>
                          )}
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}

      {/* Selected Conditions Summary */}
      {selectedConditions.length > 0 && (
        <Card sx={{ mt: 3, bgcolor: 'success.50', borderColor: 'success.main', borderWidth: 1, borderStyle: 'solid' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Selected Medical Conditions
            </Typography>
            
            <List dense>
              {selectedConditions.map((condition: any, index: number) => (
                <ListItem key={condition.condition} divider={index < selectedConditions.length - 1}>
                  <ListItemText
                    primary={condition.label}
                    secondary={`Charlson weight: ${condition.weight} point${(condition.weight || 0) > 1 ? 's' : ''}`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={() => toggleCondition(condition.condition, condition.label || '', condition.weight || 0)}
                    >
                      <CheckCircle color="success" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 2 }} />
            
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Total Charlson Score: {charlsonScore}
              </Typography>
              <Chip 
                label={`${riskInterpretation.level} Comorbidity Risk`}
                color={riskInterpretation.color as any}
                sx={{ fontWeight: 600 }}
              />
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Guidance */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Note:</strong> Hypertension status is required for accurate predictions. 
          Additional medical history (Charlson conditions) is optional but improves prediction accuracy.
          Only select conditions that have been formally diagnosed by a healthcare provider.
        </Typography>
      </Alert>

      {/* Summary */}
      <Box sx={{ mt: 4, p: 2, bgcolor: hasHypertension ? 'success.50' : 'warning.50', borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Medical History Summary:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          <Chip
            label={hasHypertension ? 'Hypertension: Yes' : 'Hypertension: Please specify'}
            color={hasHypertension ? 'success' : 'error'}
            size="small"
            sx={{ fontWeight: 600 }}
          />
          <Chip
            label={`Charlson Score: ${charlsonScore}`}
            color={riskInterpretation.color as any}
            size="small"
          />
          <Chip
            label={`Additional Conditions: ${selectedConditions.filter(c => c.condition !== 'hypertension').length}`}
            color="info"
            size="small"
          />
          <Chip
            label={hasHypertension ? 'Ready to proceed' : 'Incomplete'}
            color={hasHypertension ? 'success' : 'warning'}
            size="small"
            sx={{ fontWeight: 600 }}
          />
        </Box>
      </Box>

      {/* Next Steps */}
      {hasHypertension ? (
        <Alert severity="success" sx={{ mt: 3 }}>
          <Typography variant="body2">
            <strong>Ready for Risk Analysis!</strong> All required information has been collected. 
            The next step will generate personalized CKD progression risk predictions using our ensemble AI models.
          </Typography>
        </Alert>
      ) : (
        <Alert severity="warning" sx={{ mt: 3 }}>
          <Typography variant="body2">
            <strong>Hypertension Status Required:</strong> Please specify hypertension diagnosis status above to proceed to risk analysis.
          </Typography>
        </Alert>
      )}
      </Box>
    </LocalizationProvider>
  );
};

export default MedicalHistoryStep;