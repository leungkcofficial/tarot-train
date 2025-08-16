import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Divider
} from '@mui/material';
import {
  Analytics,
  Assessment,
  ExpandMore,
  TrendingUp,
  DatasetOutlined,
  ModelTraining,
  Verified,
  Timeline
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { ApiService, PerformanceMetrics } from '../services/api';

/**
 * Performance metrics page showing model validation results
 */
const PerformancePage: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        setLoading(true);
        const data = await ApiService.getPerformanceMetrics();
        setPerformanceData(data);
      } catch (err: any) {
        setError(err.message || 'Failed to load performance metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading performance metrics...
        </Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          <Typography variant="h6">Error Loading Performance Data</Typography>
          <Typography>{error}</Typography>
        </Alert>
      </Container>
    );
  }

  if (!performanceData) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="info">
          <Typography>Performance metrics are not available at this time.</Typography>
        </Alert>
      </Container>
    );
  }

  const { study_info, metrics, key_findings, clinical_interpretation } = performanceData;

  // Performance score visualization
  const getScoreColor = (score: number) => {
    if (score >= 0.85) return 'success';
    if (score >= 0.75) return 'warning';
    return 'error';
  };

  const getScoreLevel = (score: number) => {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.75) return 'Good';
    if (score >= 0.65) return 'Fair';
    return 'Poor';
  };

  return (
    <>
      <Helmet>
        <title>Performance Metrics - TAROT CKD Risk Prediction</title>
        <meta name="description" content="Clinical validation results and performance metrics for TAROT CKD risk prediction models." />
      </Helmet>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Header */}
        <Box textAlign="center" sx={{ mb: 6 }}>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
            <Analytics sx={{ mr: 2, verticalAlign: 'middle' }} />
            Model Performance & Validation
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
            Clinical validation results demonstrating the accuracy and reliability of TAROT CKD risk prediction models
          </Typography>
        </Box>

        {/* Study Information */}
        <Card sx={{ mb: 4, borderRadius: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              <DatasetOutlined sx={{ mr: 1, verticalAlign: 'middle' }} />
              Study Overview
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  {study_info.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  {study_info.description}
                </Typography>
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                    Validation Method:
                  </Typography>
                  <Chip label={study_info.validation_method} color="primary" />
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                  Study Characteristics:
                </Typography>
                <Box component="ul" sx={{ pl: 3, '& li': { mb: 1 } }}>
                  <li><strong>Sample Size:</strong> {study_info.sample_size}</li>
                  <li><strong>Time Horizons:</strong> {study_info.time_horizons.join(', ')}</li>
                  <li><strong>Datasets:</strong> {study_info.datasets.join(', ')}</li>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Key Performance Metrics */}
        <Card sx={{ mb: 4, borderRadius: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
              Key Performance Metrics
            </Typography>

            <Grid container spacing={3}>
              {/* C-Index Scores */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3, height: '100%', bgcolor: 'success.50' }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    Concordance Index (C-Index)
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Measures discriminative ability (0.5 = random, 1.0 = perfect)
                  </Typography>
                  
                  {metrics.dialysis_cindex && (
                    <Box sx={{ mb: 2 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                        <Typography variant="body2">Dialysis Risk</Typography>
                        <Chip
                          label={`${metrics.dialysis_cindex.toFixed(3)} ${getScoreLevel(metrics.dialysis_cindex)}`}
                          color={getScoreColor(metrics.dialysis_cindex)}
                          size="small"
                        />
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={metrics.dialysis_cindex * 100}
                        sx={{ height: 8, borderRadius: 4 }}
                        color={getScoreColor(metrics.dialysis_cindex)}
                      />
                    </Box>
                  )}
                  
                  {metrics.mortality_cindex && (
                    <Box sx={{ mb: 2 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                        <Typography variant="body2">Mortality Risk</Typography>
                        <Chip
                          label={`${metrics.mortality_cindex.toFixed(3)} ${getScoreLevel(metrics.mortality_cindex)}`}
                          color={getScoreColor(metrics.mortality_cindex)}
                          size="small"
                        />
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={metrics.mortality_cindex * 100}
                        sx={{ height: 8, borderRadius: 4 }}
                        color={getScoreColor(metrics.mortality_cindex)}
                      />
                    </Box>
                  )}
                </Paper>
              </Grid>

              {/* Calibration Metrics */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3, height: '100%', bgcolor: 'info.50' }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    Calibration Performance
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    How well predicted risks match observed outcomes
                  </Typography>
                  
                  {metrics.calibration_slope && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        <strong>Calibration Slope:</strong> {metrics.calibration_slope.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Ideal = 1.0 (perfect calibration)
                      </Typography>
                    </Box>
                  )}
                  
                  {metrics.calibration_intercept && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        <strong>Calibration Intercept:</strong> {metrics.calibration_intercept.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Ideal = 0.0 (no systematic bias)
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Detailed Performance by Time Horizon */}
        <Accordion sx={{ mb: 4 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box display="flex" alignItems="center" gap={2}>
              <Timeline />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Performance by Time Horizon
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Time Horizon</strong></TableCell>
                    <TableCell align="center"><strong>Dialysis C-Index</strong></TableCell>
                    <TableCell align="center"><strong>Mortality C-Index</strong></TableCell>
                    <TableCell align="center"><strong>Sensitivity</strong></TableCell>
                    <TableCell align="center"><strong>Specificity</strong></TableCell>
                    <TableCell align="center"><strong>PPV</strong></TableCell>
                    <TableCell align="center"><strong>NPV</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {study_info.time_horizons.map((horizon) => {
                    const horizonMetrics = metrics.by_horizon?.[horizon] || {};
                    
                    return (
                      <TableRow key={horizon}>
                        <TableCell>{horizon}</TableCell>
                        <TableCell align="center">
                          {horizonMetrics.dialysis_cindex ? (
                            <Chip
                              label={horizonMetrics.dialysis_cindex.toFixed(3)}
                              color={getScoreColor(horizonMetrics.dialysis_cindex)}
                              size="small"
                            />
                          ) : '-'}
                        </TableCell>
                        <TableCell align="center">
                          {horizonMetrics.mortality_cindex ? (
                            <Chip
                              label={horizonMetrics.mortality_cindex.toFixed(3)}
                              color={getScoreColor(horizonMetrics.mortality_cindex)}
                              size="small"
                            />
                          ) : '-'}
                        </TableCell>
                        <TableCell align="center">
                          {horizonMetrics.sensitivity ? `${(horizonMetrics.sensitivity * 100).toFixed(1)}%` : '-'}
                        </TableCell>
                        <TableCell align="center">
                          {horizonMetrics.specificity ? `${(horizonMetrics.specificity * 100).toFixed(1)}%` : '-'}
                        </TableCell>
                        <TableCell align="center">
                          {horizonMetrics.ppv ? `${(horizonMetrics.ppv * 100).toFixed(1)}%` : '-'}
                        </TableCell>
                        <TableCell align="center">
                          {horizonMetrics.npv ? `${(horizonMetrics.npv * 100).toFixed(1)}%` : '-'}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>

        {/* Key Findings */}
        <Card sx={{ mb: 4, borderRadius: 3, bgcolor: 'warning.50' }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
              Key Validation Findings
            </Typography>
            
            <Grid container spacing={3}>
              {key_findings.map((finding, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 2 }}>
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>
                      • {finding}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>

        {/* Clinical Interpretation */}
        <Card sx={{ borderRadius: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              <Verified sx={{ mr: 1, verticalAlign: 'middle' }} />
              Clinical Interpretation
            </Typography>
            
            {Object.entries(clinical_interpretation).map(([section, content]) => (
              <Box key={section} sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textTransform: 'capitalize' }}>
                  {section.replace('_', ' ')}:
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  {content}
                </Typography>
                <Divider sx={{ mt: 2 }} />
              </Box>
            ))}
          </CardContent>
        </Card>

        {/* Clinical Context Alert */}
        <Alert severity="warning" sx={{ mt: 4 }}>
          <Typography variant="body2">
            <strong>Clinical Context:</strong> These performance metrics are derived from clinical validation studies 
            on CKD patient cohorts. Model performance may vary in different populations or clinical settings. 
            Always interpret predictions within the complete clinical context and in consultation with nephrology specialists.
          </Typography>
        </Alert>
      </Container>
    </>
  );
};

export default PerformancePage;