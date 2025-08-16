import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  useTheme,
  useMediaQuery,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore,
  TrendingUp,
  TrendingDown,
  Info,
  Psychology,
  Analytics,
  Timeline
} from '@mui/icons-material';
import { PredictionResult } from '../../contexts/SessionContext';

interface ShapVisualizationProps {
  prediction: PredictionResult;
}

/**
 * SHAP values visualization component for model explainability
 */
const ShapVisualization: React.FC<ShapVisualizationProps> = ({ prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // Extract SHAP values for different time horizons
  const shapData = prediction.shap_values;
  const timeHorizons = [1, 2, 3, 4, 5];

  // Feature importance data processing
  const processShapData = (timeHorizon: number) => {
    const dialysisShap = shapData?.dialysis || {};
    const mortalityShap = shapData?.mortality || {};
    
    // Combine and sort by absolute importance
    const features: Array<{
      feature: string;
      displayName: string;
      dialysis_impact: number;
      mortality_impact: number;
      combined_impact: number;
      category: string;
    }> = [];

    const featureMapping: Record<string, { name: string; category: string }> = {
      'age': { name: 'Age', category: 'Demographics' },
      'gender': { name: 'Gender', category: 'Demographics' },
      'egfr': { name: 'eGFR', category: 'Laboratory' },
      'hemoglobin': { name: 'Hemoglobin', category: 'Laboratory' },
      'phosphate': { name: 'Phosphate', category: 'Laboratory' },
      'bicarbonate': { name: 'Bicarbonate', category: 'Laboratory' },
      'uacr': { name: 'UACR', category: 'Laboratory' },
      'charlson_score': { name: 'Charlson Comorbidity Index', category: 'Comorbidities' },
      'diabetes': { name: 'Diabetes', category: 'Comorbidities' },
      'hypertension': { name: 'Hypertension', category: 'Comorbidities' },
      'cardiovascular_disease': { name: 'Cardiovascular Disease', category: 'Comorbidities' },
      'malignancy': { name: 'Malignancy', category: 'Comorbidities' }
    };

    // Process each feature
    Object.keys({ ...dialysisShap, ...mortalityShap }).forEach(feature => {
      const dialysisValue = dialysisShap[feature] || 0;
      const mortalityValue = mortalityShap[feature] || 0;
      const combinedValue = Math.abs(dialysisValue) + Math.abs(mortalityValue);
      
      if (combinedValue > 0.001) { // Filter out very small impacts
        features.push({
          feature,
          displayName: featureMapping[feature]?.name || feature,
          dialysis_impact: dialysisValue,
          mortality_impact: mortalityValue,
          combined_impact: combinedValue,
          category: featureMapping[feature]?.category || 'Other'
        });
      }
    });

    // Sort by combined impact (descending)
    return features.sort((a, b) => b.combined_impact - a.combined_impact);
  };

  // Get feature importance for 2-year predictions (most clinically relevant)
  const twoYearFeatures = useMemo(() => processShapData(2), [shapData]);

  // Create waterfall chart data for SHAP values
  const createWaterfallData = (timeHorizon: number, outcome: 'dialysis' | 'mortality') => {
    const features = processShapData(timeHorizon);
    const shapValues = features.map(f => 
      outcome === 'dialysis' ? f.dialysis_impact : f.mortality_impact
    );
    const featureNames = features.map(f => f.displayName);
    
    // Base prediction (without features)
    const basePrediction = 0.1; // Approximate population average
    
    // Calculate cumulative values for waterfall
    let cumulative = basePrediction;
    const cumulativeValues = [basePrediction];
    const colors = [];
    
    shapValues.forEach(value => {
      cumulative += value;
      cumulativeValues.push(cumulative);
      colors.push(value > 0 ? '#d32f2f' : '#2e7d32'); // Red for increases, green for decreases
    });

    return {
      x: ['Baseline', ...featureNames, 'Final Prediction'],
      y: [basePrediction, ...shapValues, 0],
      type: 'waterfall' as const,
      name: `${outcome} Risk Factors`,
      connector: { line: { color: '#666', width: 2 } },
      increasing: { marker: { color: '#d32f2f' } },
      decreasing: { marker: { color: '#2e7d32' } },
      totals: { marker: { color: '#1976d2' } },
      textinfo: 'value+percent initial',
      hovertemplate: '<b>%{x}</b><br>Impact: %{y:.3f}<br>%{text}<extra></extra>',
      text: featureNames.map((name, i) => 
        `${shapValues[i] > 0 ? 'Increases' : 'Decreases'} risk by ${Math.abs(shapValues[i] * 100).toFixed(1)}%`
      )
    };
  };

  // Feature importance bar chart
  const createFeatureImportanceChart = (): { data: any[]; layout: any; config: any } | null => {
    if (twoYearFeatures.length === 0) return null;

    const topFeatures = twoYearFeatures.slice(0, 10); // Top 10 most important features
    
    return {
      data: [
        {
          x: topFeatures.map((f: any) => f.combined_impact),
          y: topFeatures.map((f: any) => f.displayName),
          type: 'bar',
          orientation: 'h',
          marker: {
            color: topFeatures.map((f: any) => 
              f.dialysis_impact > 0 ? '#d32f2f' : '#2e7d32'
            ),
            colorscale: [
              [0, '#2e7d32'],
              [0.5, '#ffa726'], 
              [1, '#d32f2f']
            ]
          },
          hovertemplate: '<b>%{y}</b><br>Combined Impact: %{x:.3f}<extra></extra>'
        }
      ],
      layout: {
        title: {
          text: 'Feature Importance (2-Year Risk)',
          font: { size: 16, family: 'Inter, sans-serif' },
          x: 0.5
        },
        xaxis: {
          title: 'Impact Score',
          gridcolor: '#f0f0f0',
          font: { family: 'Inter, sans-serif' }
        },
        yaxis: {
          title: '',
          autorange: 'reversed',
          font: { family: 'Inter, sans-serif' }
        },
        margin: { l: 150, r: 50, t: 60, b: 60 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
        font: { family: 'Inter, sans-serif' },
        height: 400
      },
      config: {
        displayModeBar: !isMobile,
        responsive: true,
        displaylogo: false
      }
    };
  };

  // Get category-wise feature grouping
  const getFeaturesByCategory = () => {
    const categories: Record<string, typeof twoYearFeatures> = {};
    
    twoYearFeatures.forEach(feature => {
      if (!categories[feature.category]) {
        categories[feature.category] = [];
      }
      categories[feature.category].push(feature);
    });
    
    return categories;
  };

  const featuresByCategory = getFeaturesByCategory();

  // Category icons
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Demographics': return <Psychology />;
      case 'Laboratory': return <Analytics />;
      case 'Comorbidities': return <Timeline />;
      default: return <Info />;
    }
  };

  // Get impact description
  const getImpactDescription = (dialysisImpact: number, mortalityImpact: number) => {
    const descriptions = [];
    
    if (Math.abs(dialysisImpact) > 0.01) {
      descriptions.push(
        `${dialysisImpact > 0 ? 'Increases' : 'Decreases'} dialysis risk by ${Math.abs(dialysisImpact * 100).toFixed(1)}%`
      );
    }
    
    if (Math.abs(mortalityImpact) > 0.01) {
      descriptions.push(
        `${mortalityImpact > 0 ? 'Increases' : 'Decreases'} mortality risk by ${Math.abs(mortalityImpact * 100).toFixed(1)}%`
      );
    }
    
    return descriptions.join('; ');
  };

  const featureImportanceChart = createFeatureImportanceChart();

  if (!shapData || twoYearFeatures.length === 0) {
    return (
      <Alert severity="info">
        <Typography variant="body2">
          SHAP explanation values are being calculated. This provides insights into which factors 
          most strongly influence your personalized risk prediction.
        </Typography>
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        <Psychology color="primary" sx={{ mr: 1, verticalAlign: 'middle' }} />
        AI Model Explanation
      </Typography>

      <Alert severity="info" sx={{ mb: 4 }}>
        <Typography variant="body2">
          <strong>SHAP (SHapley Additive exPlanations)</strong> values explain how each factor contributes 
          to your personalized risk prediction. Positive values increase risk, while negative values decrease risk.
        </Typography>
      </Alert>

      {/* Feature Importance Chart */}
      {featureImportanceChart && (
        <Card sx={{ mb: 4, borderRadius: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Most Influential Factors
            </Typography>
            <Plot
              data={featureImportanceChart.data}
              layout={featureImportanceChart.layout}
              config={featureImportanceChart.config}
              style={{ width: '100%' }}
            />
          </CardContent>
        </Card>
      )}

      {/* Feature Categories */}
      <Grid container spacing={3}>
        {Object.entries(featuresByCategory).map(([category, features]) => (
          <Grid item xs={12} key={category}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={2}>
                  {getCategoryIcon(category)}
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {category} Factors
                  </Typography>
                  <Chip
                    label={`${features.length} factor${features.length > 1 ? 's' : ''}`}
                    size="small"
                    color="primary"
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Factor</strong></TableCell>
                        <TableCell align="center"><strong>Impact Direction</strong></TableCell>
                        <TableCell align="center"><strong>Dialysis Risk</strong></TableCell>
                        <TableCell align="center"><strong>Mortality Risk</strong></TableCell>
                        <TableCell><strong>Clinical Interpretation</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {features.map((feature, index) => {
                        const overallImpact = feature.dialysis_impact + feature.mortality_impact;
                        
                        return (
                          <TableRow key={feature.feature}>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {feature.displayName}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              {overallImpact > 0 ? (
                                <TrendingUp color="error" />
                              ) : (
                                <TrendingDown color="success" />
                              )}
                            </TableCell>
                            <TableCell align="center">
                              <Chip
                                label={`${feature.dialysis_impact > 0 ? '+' : ''}${(feature.dialysis_impact * 100).toFixed(1)}%`}
                                size="small"
                                color={feature.dialysis_impact > 0 ? 'error' : 'success'}
                                variant={Math.abs(feature.dialysis_impact) > 0.02 ? 'filled' : 'outlined'}
                              />
                            </TableCell>
                            <TableCell align="center">
                              <Chip
                                label={`${feature.mortality_impact > 0 ? '+' : ''}${(feature.mortality_impact * 100).toFixed(1)}%`}
                                size="small"
                                color={feature.mortality_impact > 0 ? 'error' : 'success'}
                                variant={Math.abs(feature.mortality_impact) > 0.02 ? 'filled' : 'outlined'}
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {getImpactDescription(feature.dialysis_impact, feature.mortality_impact) || 'Minimal impact'}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </AccordionDetails>
            </Accordion>
          </Grid>
        ))}
      </Grid>

      {/* Key Insights */}
      <Card sx={{ mt: 4, borderRadius: 3, bgcolor: 'primary.50' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
            Key Insights from AI Analysis
          </Typography>
          
          <Grid container spacing={2}>
            {twoYearFeatures.slice(0, 3).map((feature, index) => (
              <Grid item xs={12} sm={4} key={feature.feature}>
                <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 2 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                    #{index + 1}: {feature.displayName}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {getImpactDescription(feature.dialysis_impact, feature.mortality_impact)}
                  </Typography>
                  
                  <Box display="flex" alignItems="center" gap={1} sx={{ mt: 1 }}>
                    {feature.dialysis_impact + feature.mortality_impact > 0 ? (
                      <TrendingUp color="error" fontSize="small" />
                    ) : (
                      <TrendingDown color="success" fontSize="small" />
                    )}
                    <Typography variant="caption" color="text.secondary">
                      {feature.dialysis_impact + feature.mortality_impact > 0 ? 'Risk Factor' : 'Protective Factor'}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Clinical Context */}
      <Alert severity="warning" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Clinical Context Required:</strong> This AI explanation shows statistical associations 
          from training data. Clinical interpretation should always consider the complete patient context, 
          recent changes, and treatment response. Discuss these results with your healthcare team.
        </Typography>
      </Alert>
    </Box>
  );
};

export default ShapVisualization;