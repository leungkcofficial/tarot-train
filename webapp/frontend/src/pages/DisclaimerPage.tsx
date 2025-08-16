import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
  CircularProgress,
  Paper
} from '@mui/material';
import {
  Gavel,
  ExpandMore,
  Warning,
  LocalHospital,
  Security,
  Info,
  ContactSupport,
  Policy
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { ApiService, ClinicalDisclaimer } from '../services/api';

/**
 * Clinical disclaimer and legal information page
 */
const DisclaimerPage: React.FC = () => {
  const [disclaimerData, setDisclaimerData] = useState<ClinicalDisclaimer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDisclaimerData = async () => {
      try {
        setLoading(true);
        const data = await ApiService.getClinicalDisclaimer();
        setDisclaimerData(data);
      } catch (err: any) {
        setError(err.message || 'Failed to load disclaimer information');
      } finally {
        setLoading(false);
      }
    };

    fetchDisclaimerData();
  }, []);

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading disclaimer information...
        </Typography>
      </Container>
    );
  }

  if (error || !disclaimerData) {
    // Fallback static disclaimer if API fails
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
          <Gavel sx={{ mr: 2, verticalAlign: 'middle' }} />
          Clinical Disclaimer & Legal Information
        </Typography>

        <Alert severity="error" sx={{ mb: 4 }}>
          <Typography variant="h6">Important Medical Disclaimer</Typography>
          <Typography>
            This tool is for research and educational purposes only. It should not be used for clinical decision-making 
            without proper medical supervision and interpretation by qualified healthcare professionals.
          </Typography>
        </Alert>

        <Card sx={{ borderRadius: 3 }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              Medical Device Notice
            </Typography>
            <Typography variant="body1" paragraph>
              TAROT CKD Risk Prediction is a research tool and is not approved by regulatory agencies 
              as a medical device. Results should be interpreted by qualified healthcare professionals only.
            </Typography>
          </CardContent>
        </Card>
      </Container>
    );
  }

  const { title, last_updated, sections, contact_info, regulatory_note } = disclaimerData;

  const getSectionIcon = (sectionKey: string) => {
    switch (sectionKey.toLowerCase()) {
      case 'medical_disclaimer': return <LocalHospital color="error" />;
      case 'limitations': return <Warning color="warning" />;
      case 'data_privacy': return <Security color="primary" />;
      case 'regulatory': return <Policy color="secondary" />;
      case 'liability': return <Gavel color="error" />;
      default: return <Info color="info" />;
    }
  };

  const getSectionColor = (sectionKey: string) => {
    switch (sectionKey.toLowerCase()) {
      case 'medical_disclaimer': return 'error';
      case 'limitations': return 'warning';
      case 'regulatory': return 'info';
      case 'liability': return 'error';
      default: return 'info';
    }
  };

  return (
    <>
      <Helmet>
        <title>Clinical Disclaimer - TAROT CKD Risk Prediction</title>
        <meta name="description" content="Important clinical disclaimers, limitations, and legal information for TAROT CKD risk prediction tool." />
      </Helmet>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Header */}
        <Box textAlign="center" sx={{ mb: 6 }}>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
            <Gavel sx={{ mr: 2, verticalAlign: 'middle' }} />
            {title}
          </Typography>
          <Chip 
            label={`Last Updated: ${new Date(last_updated).toLocaleDateString()}`} 
            color="primary" 
            sx={{ mt: 2 }}
          />
        </Box>

        {/* Critical Warning */}
        <Alert severity="error" sx={{ mb: 6, p: 3, borderRadius: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            ⚠️ CRITICAL MEDICAL NOTICE
          </Typography>
          <Typography variant="body1" paragraph sx={{ fontWeight: 500 }}>
            This tool is for healthcare professionals only and should NEVER replace clinical judgment, 
            comprehensive patient assessment, or established clinical guidelines.
          </Typography>
          <Typography variant="body1">
            Patients should consult qualified healthcare providers for medical advice, diagnosis, and treatment decisions.
          </Typography>
        </Alert>

        {/* Regulatory Notice */}
        {regulatory_note && (
          <Paper sx={{ p: 4, mb: 4, bgcolor: 'warning.50', borderLeft: '5px solid', borderLeftColor: 'warning.main' }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              <Policy sx={{ mr: 1, verticalAlign: 'middle' }} />
              Regulatory Status
            </Typography>
            <Typography variant="body1">
              {regulatory_note}
            </Typography>
          </Paper>
        )}

        {/* Disclaimer Sections */}
        <Box sx={{ mb: 6 }}>
          {Object.entries(sections).map(([sectionKey, sectionData]) => (
            <Accordion key={sectionKey} sx={{ mb: 2, borderRadius: 2 }}>
              <AccordionSummary 
                expandIcon={<ExpandMore />}
                sx={{ 
                  bgcolor: `${getSectionColor(sectionKey)}.50`,
                  '&:hover': { bgcolor: `${getSectionColor(sectionKey)}.100` }
                }}
              >
                <Box display="flex" alignItems="center" gap={2}>
                  {getSectionIcon(sectionKey)}
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {sectionData.title}
                  </Typography>
                  <Chip 
                    label="Important" 
                    color={getSectionColor(sectionKey) as any}
                    size="small" 
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                {sectionData.content.map((paragraph, index) => (
                  <Typography 
                    key={index} 
                    variant="body1" 
                    paragraph
                    sx={{ 
                      fontSize: '1.1rem',
                      lineHeight: 1.7,
                      color: 'text.primary'
                    }}
                  >
                    {paragraph}
                  </Typography>
                ))}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>

        {/* Contact Information */}
        <Card sx={{ borderRadius: 3, bgcolor: 'primary.50' }}>
          <CardContent sx={{ p: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              <ContactSupport sx={{ mr: 1, verticalAlign: 'middle' }} />
              Contact Information
            </Typography>
            
            <Divider sx={{ my: 3 }} />
            
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 4 }}>
              {Object.entries(contact_info).map(([contactType, details]) => (
                <Box key={contactType}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, textTransform: 'capitalize' }}>
                    {contactType.replace('_', ' ')}:
                  </Typography>
                  <Typography variant="body1" sx={{ 
                    fontFamily: 'monospace',
                    bgcolor: 'white',
                    p: 2,
                    borderRadius: 2,
                    border: '1px solid',
                    borderColor: 'divider'
                  }}>
                    {details}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Footer Notices */}
        <Box sx={{ mt: 6 }}>
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>Version Information:</strong> This disclaimer applies to all versions of the TAROT CKD Risk Prediction tool. 
              Users are responsible for ensuring they are using the most current version and disclaimer.
            </Typography>
          </Alert>

          <Alert severity="warning">
            <Typography variant="body2">
              <strong>Legal Notice:</strong> By using this tool, you acknowledge that you have read, understood, 
              and agree to be bound by all terms and conditions outlined in this disclaimer. Use of this tool 
              constitutes acceptance of all limitations and responsibilities described herein.
            </Typography>
          </Alert>
        </Box>

        {/* Emergency Contact */}
        <Paper sx={{ 
          p: 4, 
          mt: 4, 
          bgcolor: 'error.50', 
          borderLeft: '5px solid', 
          borderLeftColor: 'error.main',
          textAlign: 'center'
        }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'error.main' }}>
            🚨 MEDICAL EMERGENCY
          </Typography>
          <Typography variant="body1" sx={{ fontSize: '1.1rem', fontWeight: 500 }}>
            If this is a medical emergency, do not use this tool. 
            Contact emergency services immediately or go to the nearest emergency department.
          </Typography>
        </Paper>
      </Container>
    </>
  );
};

export default DisclaimerPage;