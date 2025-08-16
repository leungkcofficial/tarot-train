import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Container,
  Paper
} from '@mui/material';
import { 
  Home, 
  Assessment, 
  ErrorOutline
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';

/**
 * 404 Not Found page
 */
const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <>
      <Helmet>
        <title>Page Not Found - TAROT CKD</title>
        <meta name="description" content="The requested page could not be found" />
      </Helmet>

      <Container maxWidth="md" sx={{ py: 8 }}>
        <Paper 
          elevation={3}
          sx={{ 
            p: 6, 
            textAlign: 'center',
            borderRadius: 3,
            background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'
          }}
        >
          <ErrorOutline 
            sx={{ 
              fontSize: 80, 
              color: 'error.main', 
              mb: 3 
            }} 
          />
          
          <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
            404
          </Typography>
          
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            Page Not Found
          </Typography>
          
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
            The page you're looking for doesn't exist or has been moved.
            Let's get you back to where you need to be.
          </Typography>

          <Box display="flex" gap={2} justifyContent="center" flexWrap="wrap">
            <Button
              variant="contained"
              size="large"
              startIcon={<Home />}
              onClick={() => navigate('/')}
              sx={{ minWidth: 150 }}
            >
              Go Home
            </Button>
            
            <Button
              variant="outlined"
              size="large"
              startIcon={<Assessment />}
              onClick={() => navigate('/assessment')}
              sx={{ minWidth: 150 }}
            >
              Start Assessment
            </Button>
          </Box>

          {/* Quick navigation links */}
          <Box sx={{ mt: 6, pt: 4, borderTop: '1px solid', borderColor: 'divider' }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Quick Navigation
            </Typography>
            <Box display="flex" gap={3} justifyContent="center" flexWrap="wrap">
              <Button 
                variant="text" 
                onClick={() => navigate('/')}
              >
                Home
              </Button>
              <Button 
                variant="text" 
                onClick={() => navigate('/assessment')}
              >
                Risk Assessment
              </Button>
              <Button 
                variant="text" 
                onClick={() => navigate('/performance')}
              >
                Model Performance
              </Button>
              <Button 
                variant="text" 
                onClick={() => navigate('/disclaimer')}
              >
                Clinical Disclaimer
              </Button>
            </Box>
          </Box>
        </Paper>
      </Container>
    </>
  );
};

export default NotFoundPage;