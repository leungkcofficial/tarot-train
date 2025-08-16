import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, Alert } from '@mui/material';
import { Helmet } from 'react-helmet-async';

import StepperNavigation from '../components/Assessment/StepperNavigation';
import DemographicsStep from '../components/Assessment/DemographicsStep';
import LabValuesStep from '../components/Assessment/LabValuesStep';
import MedicalHistoryStep from '../components/Assessment/MedicalHistoryStep';
import { useSession } from '../contexts/SessionContext';
import { ApiService } from '../services/api';
import DataTransformationService from '../services/dataTransformation';

/**
 * Assessment page with multi-step form wizard
 */
const AssessmentPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setStep, setPrediction } = useSession();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentStep = state.data.currentStep;
  const totalSteps = 4;

  // Validation for each step
  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 0: // Demographics
        return state.data.demographics.gender !== '' && 
               state.data.demographics.age !== undefined;
      
      case 1: // Laboratory Values
        const requiredParams = ['creatinine', 'hemoglobin', 'phosphate', 'bicarbonate', 'albumin'];
        const hasRequired = requiredParams.every(param =>
          state.data.labValues.some(lab => lab.parameter === param && lab.value !== '' && lab.value !== 0)
        );
        // Check for UACR OR UPCR - at least one is required
        const hasUacrOrUpcr = 
          state.data.labValues.some(lab => lab.parameter === 'uacr' && lab.value !== '' && lab.value !== 0) ||
          state.data.labValues.some(lab => lab.parameter === 'upcr' && lab.value !== '' && lab.value !== 0);
        
        return hasRequired && hasUacrOrUpcr;
      
      case 2: // Medical History - Hypertension status is required
        const hasHypertension = state.data.comorbidities.some(c => c.condition === 'hypertension');
        return hasHypertension;
      
      case 3: // Prediction
        return isStepValid(0) && isStepValid(1) && isStepValid(2);
      
      default:
        return false;
    }
  };

  const handleNext = async () => {
    if (currentStep < totalSteps - 1) {
      // Move to next step
      setStep(currentStep + 1);
    } else {
      // Final step - generate prediction
      await generatePrediction();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setStep(currentStep - 1);
    }
  };

  const generatePrediction = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Validate using the new data transformation service
      const transformedRequest = DataTransformationService.transformToFeatureMatrix(
        state.data.demographics,
        state.data.labValues,
        state.data.comorbidities
      );

      // Validate feature matrix completeness
      const validationErrors = DataTransformationService.validateFeatureMatrix(transformedRequest);
      if (validationErrors.length > 0) {
        setError(`Validation errors: ${validationErrors.join(', ')}`);
        setIsLoading(false);
        return;
      }

      // Add session ID if available
      if (state.data.sessionId) {
        transformedRequest.session_id = state.data.sessionId;
      }

      // Debug: log feature matrix summary
      if (process.env.NODE_ENV === 'development') {
        const summary = DataTransformationService.getFeatureMatrixSummary(transformedRequest.feature_matrix);
        console.log('Feature Matrix Summary:', summary);
        console.log('Transformed Request:', transformedRequest);
      }

      // Generate prediction using temporal API
      const prediction = await ApiService.predictTemporalRisk(transformedRequest);
      
      // Store prediction and navigate to results
      setPrediction(prediction);
      navigate('/results');

    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.message || 'Failed to generate prediction. Please check your input and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return <DemographicsStep />;
      case 1:
        return <LabValuesStep />;
      case 2:
        return <MedicalHistoryStep />;
      case 3:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h4" gutterBottom>
              Ready to Generate Risk Prediction
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
              Click "Predict Risk" to analyze the provided information using our AI ensemble models.
              The analysis will take a few seconds and provide personalized risk predictions with clinical interpretations.
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                {error}
              </Alert>
            )}
            
            <Alert severity="info" sx={{ textAlign: 'left' }}>
              <Typography variant="body2">
                <strong>Privacy Notice:</strong> Your data will be processed temporarily for analysis only.
                No personal information is stored or logged by the system.
              </Typography>
            </Alert>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <>
      <Helmet>
        <title>CKD Risk Assessment - TAROT</title>
        <meta name="description" content="Complete the clinical assessment to receive AI-generated CKD progression risk predictions" />
      </Helmet>

      <Box sx={{ maxWidth: 1200, mx: 'auto', py: 4 }}>
        <Typography variant="h3" component="h1" textAlign="center" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
          CKD Risk Assessment
        </Typography>

        <Typography variant="h6" textAlign="center" color="text.secondary" sx={{ mb: 6 }}>
          Complete this clinical assessment to receive personalized CKD progression risk predictions
        </Typography>

        <StepperNavigation
          onNext={handleNext}
          onBack={handleBack}
          canProceed={isStepValid(currentStep)}
          isLoading={isLoading}
        />

        <Box sx={{ mt: 4 }}>
          {renderCurrentStep()}
        </Box>
      </Box>
    </>
  );
};

export default AssessmentPage;