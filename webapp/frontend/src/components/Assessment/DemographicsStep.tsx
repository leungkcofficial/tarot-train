import React from 'react';
import {
  Box,
  TextField,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Typography,
  Alert,
  Chip,
  InputAdornment,
  Divider
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { Person, Cake, Calculate } from '@mui/icons-material';
import { useSession, Demographics } from '../../contexts/SessionContext';
import { differenceInYears, parseISO, format } from 'date-fns';

/**
 * Demographics step component for collecting patient age and gender
 */
const DemographicsStep: React.FC = () => {
  const { state, updateDemographics } = useSession();
  const { demographics } = state.data;

  // Calculate age from date of birth
  const calculateAge = (dateOfBirth: Date): number => {
    return differenceInYears(new Date(), dateOfBirth);
  };

  // Handle date of birth change
  const handleDateOfBirthChange = (date: Date | null) => {
    if (date) {
      const calculatedAge = calculateAge(date);
      updateDemographics({
        dateOfBirth: date,
        age: calculatedAge
      });
    } else {
      updateDemographics({
        dateOfBirth: undefined,
        age: demographics.age // Keep manual age if DOB is cleared
      });
    }
  };

  // Handle direct age input
  const handleAgeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    if (value === '') {
      updateDemographics({ age: undefined });
    } else {
      const age = parseInt(value);
      if (!isNaN(age) && age >= 0 && age <= 120) {
        updateDemographics({ 
          age,
          dateOfBirth: undefined // Clear DOB when entering age directly
        });
      }
    }
  };

  // Handle gender change
  const handleGenderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const gender = event.target.value as 'male' | 'female';
    updateDemographics({ gender });
  };

  // Validation
  const hasValidAge = (demographics.age !== undefined && demographics.age >= 18) || 
                     (demographics.dateOfBirth !== undefined && calculateAge(demographics.dateOfBirth) >= 18);
  const hasValidGender = demographics.gender !== '';
  const isValid = hasValidAge && hasValidGender;

  // Age warnings
  const ageWarning = demographics.age !== undefined && demographics.age < 18;
  const ageVeryHigh = demographics.age !== undefined && demographics.age > 100;

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box className="form-section">
        <Typography variant="h5" className="form-section-title">
          <Person />
          Patient Demographics
          <Chip label="Required" size="small" color="error" sx={{ ml: 2 }} />
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Basic demographic information for risk calculation. This tool is designed for adults ≥18 years.
        </Typography>

        {/* Age Section */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Cake />
            Age Information
            <span className="required-indicator">*</span>
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            You can either enter the age directly or select the date of birth (age will be calculated automatically).
          </Typography>

          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', mb: 2 }}>
            {/* Date of Birth */}
            <DatePicker
              label="Date of Birth"
              value={demographics.dateOfBirth || null}
              onChange={handleDateOfBirthChange}
              maxDate={new Date()}
              minDate={new Date(1900, 0, 1)}
              slotProps={{
                textField: {
                  sx: { minWidth: 200 },
                  helperText: demographics.dateOfBirth 
                    ? `Age: ${calculateAge(demographics.dateOfBirth)} years`
                    : 'Optional - age will be calculated'
                }
              }}
            />

            <Typography variant="body2" sx={{ alignSelf: 'center', mx: 1, color: 'text.secondary' }}>
              OR
            </Typography>

            {/* Direct Age Input */}
            <TextField
              label="Age"
              type="number"
              value={demographics.age || ''}
              onChange={handleAgeChange}
              InputProps={{
                endAdornment: <InputAdornment position="end">years</InputAdornment>,
                startAdornment: <InputAdornment position="start"><Calculate /></InputAdornment>
              }}
              sx={{ minWidth: 150 }}
              helperText="Direct age entry (18-120 years)"
              inputProps={{ min: 18, max: 120 }}
            />
          </Box>

          {/* Age Validation Messages */}
          {ageWarning && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Age Restriction:</strong> This tool is designed for adults ≥18 years. 
                For pediatric patients, please consult a pediatric nephrologist.
              </Typography>
            </Alert>
          )}

          {ageVeryHigh && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Age Verification:</strong> Please verify that the age {demographics.age} years is correct.
                Very high ages may affect prediction accuracy.
              </Typography>
            </Alert>
          )}

          {hasValidAge && !ageWarning && !ageVeryHigh && (
            <Alert severity="success" sx={{ mb: 2 }}>
              <Typography variant="body2">
                ✓ Age: {demographics.age} years - Eligible for risk assessment
              </Typography>
            </Alert>
          )}
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Gender Section */}
        <Box sx={{ mb: 4 }}>
          <FormControl component="fieldset" required>
            <FormLabel component="legend">
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Person />
                Gender
                <span className="required-indicator">*</span>
              </Typography>
            </FormLabel>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Gender affects eGFR calculation using the CKD-EPI equation.
            </Typography>
            
            <RadioGroup
              value={demographics.gender}
              onChange={handleGenderChange}
              row
              sx={{ mt: 1 }}
            >
              <FormControlLabel
                value="male"
                control={<Radio />}
                label="Male"
                sx={{ mr: 4 }}
              />
              <FormControlLabel
                value="female"
                control={<Radio />}
                label="Female"
              />
            </RadioGroup>
          </FormControl>

          {hasValidGender && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                ✓ Gender: {demographics.gender.charAt(0).toUpperCase() + demographics.gender.slice(1)} 
                - Will be used for eGFR calculation
              </Typography>
            </Alert>
          )}
        </Box>

        {/* Validation Summary */}
        <Box sx={{ mt: 4, p: 2, bgcolor: isValid ? 'success.50' : 'warning.50', borderRadius: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Demographics Summary:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            <Chip 
              label={`Age: ${demographics.age || 'Not provided'}`}
              color={hasValidAge ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={`Gender: ${demographics.gender || 'Not selected'}`}
              color={hasValidGender ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={`CKD Diagnosis: ${demographics.ckdDiagnosisDate ? 'Provided' : 'Optional'}`}
              color="info"
              size="small"
            />
            <Chip
              label={isValid ? 'Ready to proceed' : 'Incomplete'}
              color={isValid ? 'success' : 'warning'}
              size="small"
              sx={{ fontWeight: 600 }}
            />
          </Box>
        </Box>

        {/* CKD Diagnosis Date */}
        <Divider sx={{ my: 3 }} />
        
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Calculate />
            CKD Diagnosis Date
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            When did you first learn that you had CKD (Chronic Kidney Disease)? This is optional - if you don't remember the exact date, leave it blank.
            The observation period will be calculated from the earliest available date (diagnosis or first lab test).
          </Typography>

          <DatePicker
            label="CKD Diagnosis Date"
            value={demographics.ckdDiagnosisDate || null}
            onChange={(date) => updateDemographics({ ckdDiagnosisDate: date || undefined })}
            maxDate={new Date()}
            minDate={new Date(1980, 0, 1)}
            slotProps={{
              textField: {
                sx: { minWidth: 250 },
                helperText: demographics.ckdDiagnosisDate 
                  ? 'This date will help calculate your observation period'
                  : 'Optional - leave blank if you don\'t remember'
              }
            }}
          />
        </Box>

        {/* Next Steps Preview */}
        {isValid && (
          <Alert severity="info" sx={{ mt: 3 }}>
            <Typography variant="body2">
              <strong>Next:</strong> Laboratory values (creatinine, hemoglobin, phosphate, bicarbonate, albumin, and UACR/UPCR) 
              will be collected to calculate eGFR and assess CKD progression risk. The observation period will be calculated 
              automatically from your earliest available dates.
            </Typography>
          </Alert>
        )}
      </Box>
    </LocalizationProvider>
  );
};

export default DemographicsStep;