import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Container } from '@mui/material';
import { Helmet } from 'react-helmet-async';

import Layout from './components/Layout/Layout';
import PrivacyNotice from './components/Common/PrivacyNotice';
import HomePage from './pages/HomePage';
import AssessmentPage from './pages/AssessmentPage';
import ResultsPage from './pages/ResultsPage';
import PerformancePage from './pages/PerformancePage';
import DisclaimerPage from './pages/DisclaimerPage';
import NotFoundPage from './pages/NotFoundPage';
import { SessionProvider } from './contexts/SessionContext';

/**
 * Main App component with routing and global providers
 */
function App() {
  return (
    <>
      <Helmet>
        <title>TAROT CKD Risk Prediction</title>
        <meta name="description" content="Advanced AI-driven risk assessment for chronic kidney disease progression" />
        <meta name="keywords" content="CKD, chronic kidney disease, risk prediction, nephrology, AI, machine learning" />
        <meta name="author" content="TAROT Study" />
        
        {/* Medical disclaimer meta */}
        <meta name="medical-disclaimer" content="For healthcare professionals only. Not a substitute for clinical judgment." />
        
        {/* Privacy and security */}
        <meta name="privacy" content="No patient data stored or logged" />
        <meta name="data-retention" content="Session-based processing only" />
        
        {/* Open Graph tags */}
        <meta property="og:title" content="TAROT CKD Risk Prediction" />
        <meta property="og:description" content="Advanced AI-driven risk assessment for chronic kidney disease progression" />
        <meta property="og:type" content="website" />
        
        {/* Twitter Card tags */}
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content="TAROT CKD Risk Prediction" />
        <meta name="twitter:description" content="Advanced AI-driven risk assessment for chronic kidney disease progression" />
        
        {/* Canonical URL */}
        <link rel="canonical" href={window.location.origin} />
      </Helmet>

      <SessionProvider>
        <Layout>
          <PrivacyNotice />
          
          <Container maxWidth="xl" sx={{ py: 2 }}>
            <Routes>
              {/* Main application routes */}
              <Route path="/" element={<HomePage />} />
              <Route path="/assessment" element={<AssessmentPage />} />
              <Route path="/results" element={<ResultsPage />} />
              
              {/* Information pages */}
              <Route path="/performance" element={<PerformancePage />} />
              <Route path="/disclaimer" element={<DisclaimerPage />} />
              
              {/* Redirect legacy routes */}
              <Route path="/predict" element={<Navigate to="/assessment" replace />} />
              <Route path="/info/performance" element={<Navigate to="/performance" replace />} />
              <Route path="/info/disclaimer" element={<Navigate to="/disclaimer" replace />} />
              
              {/* 404 page */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Container>
        </Layout>
      </SessionProvider>
    </>
  );
}

export default App;