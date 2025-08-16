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
  useTheme,
  useMediaQuery
} from '@mui/material';
import { TrendingUp, Warning, CheckCircle } from '@mui/icons-material';
import { PredictionResult } from '../../contexts/SessionContext';

interface RiskVisualizationProps {
  prediction: PredictionResult;
}

/**
 * Risk visualization component with interactive Plotly charts
 */
const RiskVisualization: React.FC<RiskVisualizationProps> = ({ prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // Time horizons (1-5 years)
  const timeHorizons = [1, 2, 3, 4, 5];
  
  // Convert predictions to percentages
  const dialysisRisk = prediction.predictions.dialysis_risk.map(r => r * 100);
  const mortalityRisk = prediction.predictions.mortality_risk.map(r => r * 100);
  
  // Confidence intervals
  const dialysisCI = prediction.confidence_intervals ? {
    lower: prediction.confidence_intervals.dialysis_lower.map(r => r * 100),
    upper: prediction.confidence_intervals.dialysis_upper.map(r => r * 100)
  } : null;
  
  const mortalityCI = prediction.confidence_intervals ? {
    lower: prediction.confidence_intervals.mortality_lower.map(r => r * 100),
    upper: prediction.confidence_intervals.mortality_upper.map(r => r * 100)
  } : null;

  // Clinical benchmarks
  const benchmarks = prediction.clinical_benchmarks;
  const nephrology_threshold = benchmarks.nephrology_referral_threshold * 100; // 5%
  const multidisciplinary_threshold = benchmarks.multidisciplinary_care_threshold * 100; // 10%
  const krt_threshold = benchmarks.krt_preparation_threshold * 100; // 40%

  // Risk assessment
  const twoYearDialysisRisk = dialysisRisk[1]; // 2-year risk
  const fiveYearDialysisRisk = dialysisRisk[4]; // 5-year risk
  
  const getRiskLevel = (risk2y: number, risk5y: number) => {
    if (risk2y > krt_threshold) return 'very-high';
    if (risk2y > multidisciplinary_threshold) return 'high';
    if (risk5y > nephrology_threshold) return 'moderate';
    return 'low';
  };

  const riskLevel = getRiskLevel(twoYearDialysisRisk, fiveYearDialysisRisk);

  // Risk colors and labels
  const riskConfig = {
    'low': { color: '#2e7d32', label: 'Low Risk', icon: <CheckCircle />, bgColor: '#e8f5e8' },
    'moderate': { color: '#f57c00', label: 'Moderate Risk', icon: <TrendingUp />, bgColor: '#fff8e1' },
    'high': { color: '#d32f2f', label: 'High Risk', icon: <Warning />, bgColor: '#ffebee' },
    'very-high': { color: '#b71c1c', label: 'Very High Risk', icon: <Warning />, bgColor: '#ffcdd2' }
  };

  // Plotly configuration
  const plotConfig: any = {
    displayModeBar: !isMobile,
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
  };

  const plotLayout = useMemo((): any => ({
    title: {
      text: 'CKD Progression Risk Over Time',
      font: { size: isMobile ? 16 : 20, family: 'Inter, sans-serif' },
      x: 0.5
    },
    xaxis: {
      title: 'Time Horizon (Years)',
      tickmode: 'array',
      tickvals: timeHorizons,
      ticktext: timeHorizons.map(t => `${t}Y`),
      gridcolor: '#f0f0f0',
      font: { family: 'Inter, sans-serif' }
    },
    yaxis: {
      title: 'Risk (%)',
      range: [0, Math.max(100, Math.max(...dialysisRisk, ...mortalityRisk) * 1.1)],
      gridcolor: '#f0f0f0',
      font: { family: 'Inter, sans-serif' }
    },
    legend: {
      orientation: isMobile ? 'h' : 'v',
      x: isMobile ? 0.5 : 1.02,
      y: isMobile ? -0.2 : 1,
      xanchor: isMobile ? 'center' : 'left',
      font: { family: 'Inter, sans-serif' }
    },
    margin: {
      l: 60,
      r: isMobile ? 20 : 120,
      t: 80,
      b: isMobile ? 100 : 60
    },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    font: { family: 'Inter, sans-serif' },
    height: isMobile ? 400 : 500,
    shapes: [
      // KDIGO benchmarks as horizontal lines
      {
        type: 'line',
        x0: 0.8, x1: 5.2,
        y0: nephrology_threshold, y1: nephrology_threshold,
        line: { color: '#2e7d32', width: 2, dash: 'dot' }
      },
      {
        type: 'line',
        x0: 0.8, x1: 2.2,
        y0: multidisciplinary_threshold, y1: multidisciplinary_threshold,
        line: { color: '#f57c00', width: 2, dash: 'dot' }
      },
      {
        type: 'line',
        x0: 0.8, x1: 2.2,
        y0: krt_threshold, y1: krt_threshold,
        line: { color: '#d32f2f', width: 2, dash: 'dot' }
      }
    ],
    annotations: [
      {
        x: 5,
        y: nephrology_threshold + 2,
        text: '5Y: Nephrology Referral (5%)',
        showarrow: false,
        font: { size: 10, color: '#2e7d32' }
      },
      {
        x: 2,
        y: multidisciplinary_threshold + 3,
        text: '2Y: Multidisciplinary Care (10%)',
        showarrow: false,
        font: { size: 10, color: '#f57c00' }
      },
      {
        x: 2,
        y: krt_threshold + 3,
        text: '2Y: KRT Preparation (40%)',
        showarrow: false,
        font: { size: 10, color: '#d32f2f' }
      }
    ]
  }), [isMobile, dialysisRisk, mortalityRisk, timeHorizons, nephrology_threshold, multidisciplinary_threshold, krt_threshold]);

  const plotData = useMemo(() => {
    const data: any[] = [
      // Dialysis risk line
      {
        x: timeHorizons,
        y: dialysisRisk,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Dialysis Risk',
        line: { color: '#1976d2', width: 3 },
        marker: { size: 8 }
      },
      // Mortality risk line  
      {
        x: timeHorizons,
        y: mortalityRisk,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Mortality Risk',
        line: { color: '#dc004e', width: 3 },
        marker: { size: 8 }
      }
    ];

    // Add confidence intervals if available
    if (dialysisCI) {
      data.push({
        x: [...timeHorizons, ...timeHorizons.slice().reverse()],
        y: [...dialysisCI.upper, ...dialysisCI.lower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(25, 118, 210, 0.1)',
        line: { color: 'rgba(255,255,255,0)' },
        name: 'Dialysis 95% CI',
        showlegend: false,
        hoverinfo: 'skip'
      });
    }

    if (mortalityCI) {
      data.push({
        x: [...timeHorizons, ...timeHorizons.slice().reverse()],
        y: [...mortalityCI.upper, ...mortalityCI.lower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(220, 0, 78, 0.1)',
        line: { color: 'rgba(255,255,255,0)' },
        name: 'Mortality 95% CI',
        showlegend: false,
        hoverinfo: 'skip'
      });
    }

    return data;
  }, [timeHorizons, dialysisRisk, mortalityRisk, dialysisCI, mortalityCI]);

  return (
    <Box>
      {/* Risk Summary Card */}
      <Card
        sx={{
          mb: 4,
          background: `linear-gradient(135deg, ${riskConfig[riskLevel].color} 0%, ${riskConfig[riskLevel].color}dd 100%)`,
          color: 'white',
          borderRadius: 3
        }}
      >
        <CardContent sx={{ p: 4, textAlign: 'center' }}>
          <Box display="flex" justifyContent="center" alignItems="center" gap={2} mb={2}>
            {riskConfig[riskLevel].icon}
            <Typography variant="h4" sx={{ fontWeight: 700 }}>
              {riskConfig[riskLevel].label}
            </Typography>
          </Box>
          
          <Grid container spacing={3} justifyContent="center">
            <Grid item xs={6} sm={3}>
              <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                {twoYearDialysisRisk.toFixed(1)}%
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                2-Year Dialysis Risk
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
                {fiveYearDialysisRisk.toFixed(1)}%
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                5-Year Dialysis Risk
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Clinical Recommendations */}
      <Box sx={{ mb: 4 }}>
        {riskLevel === 'low' && (
          <Alert severity="success" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Low Risk:</strong> Standard CKD care with annual monitoring is appropriate. 
              Continue current management plan.
            </Typography>
          </Alert>
        )}
        
        {riskLevel === 'moderate' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Moderate Risk:</strong> Consider nephrology referral (5-year risk ≥5%). 
              Enhanced monitoring and lifestyle counseling recommended.
            </Typography>
          </Alert>
        )}
        
        {riskLevel === 'high' && (
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>High Risk:</strong> Multidisciplinary care recommended (2-year risk ≥10%). 
              Begin preparation planning and patient education.
            </Typography>
          </Alert>
        )}
        
        {riskLevel === 'very-high' && (
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Very High Risk:</strong> Urgent KRT preparation needed (2-year risk ≥40%). 
              Vascular access planning and transplant evaluation should begin immediately.
            </Typography>
          </Alert>
        )}
      </Box>

      {/* Interactive Plot */}
      <Card sx={{ mb: 4, borderRadius: 3 }}>
        <CardContent sx={{ p: 3 }}>
          <Plot
            data={plotData}
            layout={plotLayout}
            config={plotConfig}
            style={{ width: '100%' }}
          />
          
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Dotted lines show KDIGO clinical benchmarks. Shaded areas represent 95% confidence intervals.
            Click legend items to show/hide data series.
          </Typography>
        </CardContent>
      </Card>

      {/* Risk Breakdown */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', borderRadius: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Dialysis Risk Timeline
              </Typography>
              {timeHorizons.map((year, index) => (
                <Box key={year} display="flex" justifyContent="space-between" alignItems="center" sx={{ py: 1 }}>
                  <Typography variant="body2">
                    {year} Year{year > 1 ? 's' : ''}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {dialysisRisk[index].toFixed(1)}%
                    </Typography>
                    {dialysisCI && (
                      <Typography variant="caption" color="text.secondary">
                        ({dialysisCI.lower[index].toFixed(1)}-{dialysisCI.upper[index].toFixed(1)}%)
                      </Typography>
                    )}
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', borderRadius: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Mortality Risk Timeline
              </Typography>
              {timeHorizons.map((year, index) => (
                <Box key={year} display="flex" justifyContent="space-between" alignItems="center" sx={{ py: 1 }}>
                  <Typography variant="body2">
                    {year} Year{year > 1 ? 's' : ''}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {mortalityRisk[index].toFixed(1)}%
                    </Typography>
                    {mortalityCI && (
                      <Typography variant="caption" color="text.secondary">
                        ({mortalityCI.lower[index].toFixed(1)}-{mortalityCI.upper[index].toFixed(1)}%)
                      </Typography>
                    )}
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskVisualization;