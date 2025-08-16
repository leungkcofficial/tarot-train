import React from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  Container,
  Paper
} from '@mui/material';
import {
  Assessment,
  Analytics,
  Security,
  Speed,
  Verified,
  Psychology,
  LocalHospital,
  Timeline
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

/**
 * Home page with introduction and navigation to assessment
 */
const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const handleStartAssessment = () => {
    navigate('/assessment');
  };

  const features = [
    {
      icon: <Psychology color="primary" />,
      title: 'AI-Powered Prediction',
      description: 'Ensemble of 36+ deep learning models for accurate risk assessment'
    },
    {
      icon: <Analytics color="primary" />,
      title: 'Clinical Validation',
      description: 'C-index ≥0.85 with external validation on large CKD cohorts'
    },
    {
      icon: <Speed color="primary" />,
      title: 'Real-time Results',
      description: 'Sub-200ms inference with comprehensive SHAP explanations'
    },
    {
      icon: <Security color="primary" />,
      title: 'Privacy First',
      description: 'Zero data storage with session-based processing only'
    },
    {
      icon: <Verified color="primary" />,
      title: 'Evidence-Based',
      description: 'Integrated KDIGO guidelines and clinical benchmarks'
    },
    {
      icon: <Timeline color="primary" />,
      title: 'Multi-Horizon',
      description: '1-5 year risk predictions with confidence intervals'
    }
  ];

  const clinicalBenchmarks = [
    {
      threshold: '3-5%',
      timeframe: '5-year risk',
      action: 'Nephrology referral',
      color: 'success' as const,
      description: 'Consider specialist consultation'
    },
    {
      threshold: '>10%',
      timeframe: '2-year risk',
      action: 'Multidisciplinary care',
      color: 'warning' as const,
      description: 'Initiate care planning'
    },
    {
      threshold: '>40%',
      timeframe: '2-year risk',
      action: 'KRT preparation',
      color: 'error' as const,
      description: 'Urgent preparation needed'
    }
  ];

  return (
    <>
      <Helmet>
        <title>TAROT CKD Risk Prediction - Home</title>
        <meta name="description" content="Advanced AI-driven risk assessment for chronic kidney disease progression. Evidence-based predictions with clinical benchmarks." />
      </Helmet>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Hero Section */}
        <Box textAlign="center" sx={{ mb: 6 }}>
          <Typography
            variant="h2"
            component="h1"
            gutterBottom
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
              backgroundClip: 'text',
              textFillColor: 'transparent',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 2
            }}
          >
            TAROT CKD Risk Prediction
          </Typography>
          
          <Typography
            variant="h5"
            color="text.secondary"
            sx={{ mb: 4, maxWidth: 800, mx: 'auto', fontWeight: 400 }}
          >
            Advanced AI-driven risk assessment for chronic kidney disease progression.
            Predicting dialysis initiation and mortality risk with clinical precision.
          </Typography>

          <Box sx={{ mb: 4 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<Assessment />}
              onClick={handleStartAssessment}
              sx={{
                py: 2,
                px: 4,
                fontSize: '1.1rem',
                fontWeight: 600,
                boxShadow: '0 4px 20px rgba(25, 118, 210, 0.3)',
                '&:hover': {
                  boxShadow: '0 6px 24px rgba(25, 118, 210, 0.4)',
                  transform: 'translateY(-2px)'
                },
                transition: 'all 0.3s ease'
              }}
            >
              Start Risk Assessment
            </Button>
          </Box>

          <Box display="flex" justifyContent="center" gap={1} flexWrap="wrap">
            <Chip icon={<LocalHospital />} label="For Healthcare Professionals" color="primary" />
            <Chip icon={<Security />} label="HIPAA Compliant Design" variant="outlined" />
            <Chip icon={<Verified />} label="Clinically Validated" variant="outlined" />
          </Box>
        </Box>

        {/* Clinical Alert */}
        <Alert
          severity="warning"
          icon={<LocalHospital />}
          sx={{ mb: 6, borderRadius: 2 }}
        >
          <Typography variant="h6" gutterBottom>
            Healthcare Professional Tool
          </Typography>
          <Typography>
            This tool is designed for healthcare professionals and should not replace clinical judgment.
            All predictions require professional interpretation within the clinical context.
          </Typography>
        </Alert>

        {/* Features Section */}
        <Typography variant="h3" component="h2" textAlign="center" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
          Advanced Clinical Features
        </Typography>

        <Grid container spacing={3} sx={{ mb: 6 }}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)'
                  }
                }}
              >
                <CardContent sx={{ p: 3, textAlign: 'center' }}>
                  <Box sx={{ mb: 2 }}>
                    {feature.icon}
                  </Box>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    {feature.title}
                  </Typography>
                  <Typography color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Clinical Benchmarks */}
        <Paper sx={{ p: 4, mb: 6, borderRadius: 3, background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)' }}>
          <Typography variant="h4" component="h2" textAlign="center" gutterBottom sx={{ fontWeight: 600, mb: 4 }}>
            KDIGO Clinical Benchmarks
          </Typography>
          <Typography textAlign="center" color="text.secondary" sx={{ mb: 4 }}>
            Evidence-based thresholds integrated into risk assessment
          </Typography>

          <Grid container spacing={3}>
            {clinicalBenchmarks.map((benchmark, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Box
                  textAlign="center"
                  sx={{
                    p: 3,
                    bgcolor: 'white',
                    borderRadius: 2,
                    height: '100%',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
                    border: `2px solid`,
                    borderColor: `${benchmark.color}.main`
                  }}
                >
                  <Typography variant="h4" color={`${benchmark.color}.main`} gutterBottom sx={{ fontWeight: 700 }}>
                    {benchmark.threshold}
                  </Typography>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                    {benchmark.timeframe}
                  </Typography>
                  <Chip
                    label={benchmark.action}
                    color={benchmark.color}
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {benchmark.description}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Patient Eligibility */}
        <Card sx={{ mb: 6, borderRadius: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h4" component="h2" textAlign="center" gutterBottom sx={{ fontWeight: 600, mb: 4 }}>
              Patient Eligibility Criteria
            </Typography>

            <Grid container spacing={4}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" color="success.main" gutterBottom sx={{ fontWeight: 600 }}>
                  ✅ Eligible Patients
                </Typography>
                <Box component="ul" sx={{ pl: 3, '& li': { mb: 1 } }}>
                  <li>Age ≥18 years</li>
                  <li>CKD Stage 3-5 (eGFR 10-60 mL/min/1.73m²)</li>
                  <li>Recent laboratory results available</li>
                  <li>Stable clinical condition</li>
                  <li>Complete demographic information</li>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="h6" color="error.main" gutterBottom sx={{ fontWeight: 600 }}>
                  ❌ Exclusions
                </Typography>
                <Box component="ul" sx={{ pl: 3, '& li': { mb: 1 } }}>
                  <li>Age &lt;18 years (pediatric patients)</li>
                  <li>eGFR &gt;60 (normal kidney function)</li>
                  <li>eGFR &lt;10 (urgent dialysis needed)</li>
                  <li>Acute kidney injury</li>
                  <li>Active malignancy with short prognosis</li>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Call to Action */}
        <Box textAlign="center" sx={{ py: 6 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
            Ready to Begin?
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
            The assessment takes 3-5 minutes and provides immediate results with clinical interpretation.
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={<Assessment />}
            onClick={handleStartAssessment}
            sx={{
              py: 2,
              px: 6,
              fontSize: '1.1rem',
              fontWeight: 600
            }}
          >
            Start Assessment Now
          </Button>
        </Box>
      </Container>
    </>
  );
};

export default HomePage;