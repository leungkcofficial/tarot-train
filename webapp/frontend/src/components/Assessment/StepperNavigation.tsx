import React from 'react';
import {
  Stepper,
  Step,
  StepLabel,
  Box,
  Button,
  Typography,
  Chip,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Person,
  Biotech,
  LocalHospital,
  Assessment as AssessmentIcon,
  ArrowBack,
  ArrowForward
} from '@mui/icons-material';
import { useSession } from '../../contexts/SessionContext';

/**
 * Step navigation component for the assessment wizard
 */
interface StepperNavigationProps {
  onNext: () => void;
  onBack: () => void;
  canProceed: boolean;
  isLoading?: boolean;
}

const steps = [
  {
    label: 'Demographics',
    description: 'Age and gender',
    icon: <Person />,
    required: true
  },
  {
    label: 'Laboratory Values',
    description: 'Recent test results',
    icon: <Biotech />,
    required: true
  },
  {
    label: 'Medical History',
    description: 'Comorbidities (optional)',
    icon: <LocalHospital />,
    required: false
  },
  {
    label: 'Risk Prediction',
    description: 'AI analysis',
    icon: <AssessmentIcon />,
    required: true
  }
];

const StepperNavigation: React.FC<StepperNavigationProps> = ({
  onNext,
  onBack,
  canProceed,
  isLoading = false
}) => {
  const { state } = useSession();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const currentStep = state.data.currentStep;

  const getStepStatus = (stepIndex: number) => {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return 'active';
    return 'pending';
  };

  const isStepValid = (stepIndex: number) => {
    switch (stepIndex) {
      case 0: // Demographics
        return state.data.demographics.gender !== '' && 
               (state.data.demographics.age !== undefined || state.data.demographics.dateOfBirth !== undefined);
      
      case 1: // Laboratory Values
        const requiredParams = ['creatinine', 'hemoglobin', 'phosphate', 'bicarbonate'];
        const hasUrine = state.data.labValues.some(lab => ['uacr', 'upcr'].includes(lab.parameter));
        const hasRequired = requiredParams.every(param =>
          state.data.labValues.some(lab => lab.parameter === param && lab.value !== '')
        );
        return hasRequired && hasUrine;
      
      case 2: // Medical History (optional)
        return true; // Always valid since it's optional
      
      case 3: // Prediction
        return canProceed;
      
      default:
        return false;
    }
  };

  const getStepContent = (stepIndex: number) => {
    const step = steps[stepIndex];
    const isValid = isStepValid(stepIndex);
    const status = getStepStatus(stepIndex);

    return (
      <Box display="flex" alignItems="center" gap={1}>
        <Box display="flex" alignItems="center" gap={1}>
          {step.icon}
          <Box>
            <Typography variant="body2" sx={{ fontWeight: status === 'active' ? 600 : 400 }}>
              {step.label}
              {step.required && <span style={{ color: theme.palette.error.main }}> *</span>}
            </Typography>
            {!isMobile && (
              <Typography variant="caption" color="text.secondary">
                {step.description}
              </Typography>
            )}
          </Box>
        </Box>
        
        {status === 'completed' && (
          <Chip
            label="✓"
            size="small"
            color={isValid ? 'success' : 'warning'}
            sx={{ minWidth: 24, height: 20 }}
          />
        )}
      </Box>
    );
  };

  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === steps.length - 1;
  const currentStepData = steps[currentStep];

  return (
    <Box>
      {/* Progress Stepper */}
      <Box sx={{ mb: 4 }}>
        <Stepper 
          activeStep={currentStep} 
          orientation={isMobile ? 'vertical' : 'horizontal'}
          sx={{
            '& .MuiStepConnector-root': {
              top: 20
            }
          }}
        >
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel
                error={getStepStatus(index) === 'completed' && !isStepValid(index)}
                sx={{
                  '& .MuiStepLabel-labelContainer': {
                    width: '100%'
                  }
                }}
              >
                {getStepContent(index)}
              </StepLabel>
            </Step>
          ))}
        </Stepper>
      </Box>

      {/* Current Step Info */}
      <Box sx={{ mb: 3, p: 2, bgcolor: 'primary.50', borderRadius: 2, border: '1px solid', borderColor: 'primary.100' }}>
        <Box display="flex" alignItems="center" gap={2} mb={1}>
          {currentStepData.icon}
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Step {currentStep + 1}: {currentStepData.label}
            {currentStepData.required && <span style={{ color: theme.palette.error.main }}> *</span>}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {currentStepData.description}
          {currentStepData.required ? ' (Required)' : ' (Optional)'}
        </Typography>
      </Box>

      {/* Navigation Buttons */}
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Button
          startIcon={<ArrowBack />}
          onClick={onBack}
          disabled={isFirstStep || isLoading}
          sx={{ minWidth: 120 }}
        >
          Back
        </Button>

        <Box display="flex" alignItems="center" gap={2}>
          {/* Step counter */}
          <Typography variant="body2" color="text.secondary">
            {currentStep + 1} of {steps.length}
          </Typography>

          {/* Progress indicator */}
          <Box
            sx={{
              width: 100,
              height: 4,
              bgcolor: 'grey.200',
              borderRadius: 2,
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                width: `${((currentStep + 1) / steps.length) * 100}%`,
                height: '100%',
                bgcolor: 'primary.main',
                transition: 'width 0.3s ease'
              }}
            />
          </Box>
        </Box>

        <Button
          variant="contained"
          endIcon={isLastStep ? <AssessmentIcon /> : <ArrowForward />}
          onClick={onNext}
          disabled={!canProceed || isLoading}
          sx={{ minWidth: 120 }}
        >
          {isLastStep ? (isLoading ? 'Analyzing...' : 'Predict Risk') : 'Next'}
        </Button>
      </Box>

      {/* Validation Messages */}
      {!canProceed && currentStep < 3 && (
        <Box sx={{ mt: 2 }}>
          {currentStep === 0 && !isStepValid(0) && (
            <Typography variant="body2" color="error">
              Please provide age and gender information.
            </Typography>
          )}
          {currentStep === 1 && !isStepValid(1) && (
            <Typography variant="body2" color="error">
              Please provide all required laboratory values: creatinine, hemoglobin, phosphate, bicarbonate, and either UACR or UPCR.
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default StepperNavigation;