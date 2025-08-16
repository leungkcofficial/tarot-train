import React, { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Snackbar,
  Button,
  Box,
  IconButton,
  Collapse
} from '@mui/material';
import { Security, Close, ExpandMore, ExpandLess } from '@mui/icons-material';

/**
 * Privacy notice component displayed on app load
 * Emphasizes no data storage policy
 */
const PrivacyNotice: React.FC = () => {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const handleClose = () => {
    setOpen(false);
  };

  const handleExpand = () => {
    setExpanded(!expanded);
  };

  if (!open) return null;

  return (
    <Snackbar
      open={open}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      sx={{ mt: 8, maxWidth: '95%', width: 'auto' }}
    >
      <Alert
        severity="info"
        icon={<Security />}
        action={
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={handleClose}
          >
            <Close fontSize="inherit" />
          </IconButton>
        }
        sx={{
          width: '100%',
          maxWidth: 600,
          boxShadow: 3,
          '& .MuiAlert-message': {
            width: '100%'
          }
        }}
      >
        <AlertTitle>🔒 Privacy-First Design</AlertTitle>
        
        <Box>
          <strong>No patient data is stored or logged.</strong> All processing occurs in temporary session memory and is immediately discarded.
          
          <Button
            size="small"
            onClick={handleExpand}
            endIcon={expanded ? <ExpandLess /> : <ExpandMore />}
            sx={{ ml: 1, textTransform: 'none' }}
          >
            {expanded ? 'Less info' : 'More info'}
          </Button>
        </Box>

        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <Box sx={{ mt: 2, fontSize: '0.875rem', lineHeight: 1.5 }}>
            <strong>Privacy Features:</strong>
            <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
              <li>Session-based processing only</li>
              <li>Automatic data clearing after session</li>
              <li>No external data transmission</li>
              <li>Anonymous session IDs only</li>
              <li>HIPAA-compliant design principles</li>
            </ul>
            
            <strong>For Healthcare Professionals:</strong>
            <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
              <li>Clinical decision support tool</li>
              <li>Not a replacement for clinical judgment</li>
              <li>Requires professional interpretation</li>
            </ul>
          </Box>
        </Collapse>
      </Alert>
    </Snackbar>
  );
};

export default PrivacyNotice;