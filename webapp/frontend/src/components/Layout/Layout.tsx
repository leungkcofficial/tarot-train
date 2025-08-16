import React from 'react';
import { Box, AppBar, Toolbar, Typography, IconButton, Menu, MenuItem, Container, Chip } from '@mui/material';
import { 
  LocalHospital, 
  Menu as MenuIcon, 
  Info, 
  Assessment, 
  Security, 
  Analytics 
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [menuAnchor, setMenuAnchor] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    handleMenuClose();
  };

  const isCurrentPath = (path: string) => location.pathname === path;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <AppBar 
        position="sticky" 
        sx={{ 
          background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
          boxShadow: '0 2px 12px rgba(25, 118, 210, 0.15)'
        }}
      >
        <Toolbar>
          {/* Logo and Title */}
          <LocalHospital sx={{ mr: 2, fontSize: 28 }} />
          <Typography
            variant="h6"
            component="div"
            sx={{ 
              flexGrow: 1, 
              fontWeight: 600,
              cursor: 'pointer',
              '&:hover': { opacity: 0.8 }
            }}
            onClick={() => navigate('/')}
          >
            TAROT CKD Risk Prediction
          </Typography>

          {/* Privacy Indicator */}
          <Chip
            icon={<Security />}
            label="No Data Stored"
            size="small"
            sx={{ 
              mr: 2,
              backgroundColor: 'rgba(255, 255, 255, 0.15)',
              color: 'white',
              '& .MuiChip-icon': { color: 'white' }
            }}
          />

          {/* Navigation Menu */}
          <IconButton
            color="inherit"
            onClick={handleMenuOpen}
            sx={{ ml: 1 }}
          >
            <MenuIcon />
          </IconButton>
          <Menu
            anchorEl={menuAnchor}
            open={Boolean(menuAnchor)}
            onClose={handleMenuClose}
            PaperProps={{
              sx: {
                mt: 1,
                minWidth: 200,
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                borderRadius: 2
              }
            }}
          >
            <MenuItem 
              onClick={() => handleNavigate('/')}
              selected={isCurrentPath('/')}
              sx={{ py: 1.5 }}
            >
              <LocalHospital sx={{ mr: 2, color: 'primary.main' }} />
              Home
            </MenuItem>
            <MenuItem 
              onClick={() => handleNavigate('/assessment')}
              selected={isCurrentPath('/assessment')}
              sx={{ py: 1.5 }}
            >
              <Assessment sx={{ mr: 2, color: 'primary.main' }} />
              Risk Assessment
            </MenuItem>
            <MenuItem 
              onClick={() => handleNavigate('/performance')}
              selected={isCurrentPath('/performance')}
              sx={{ py: 1.5 }}
            >
              <Analytics sx={{ mr: 2, color: 'primary.main' }} />
              Model Performance
            </MenuItem>
            <MenuItem 
              onClick={() => handleNavigate('/disclaimer')}
              selected={isCurrentPath('/disclaimer')}
              sx={{ py: 1.5 }}
            >
              <Info sx={{ mr: 2, color: 'primary.main' }} />
              Clinical Disclaimer
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, backgroundColor: '#fafafa' }}>
        {children}
      </Box>

      {/* Footer */}
      <Box
        component="footer"
        sx={{
          mt: 'auto',
          py: 2,
          px: 3,
          backgroundColor: '#f5f5f5',
          borderTop: '1px solid #e0e0e0'
        }}
      >
        <Container maxWidth="xl">
          <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap">
            <Typography variant="body2" color="text.secondary">
              © 2024 TAROT Study. Healthcare Professional Tool.
            </Typography>
            <Box display="flex" gap={2} flexWrap="wrap">
              <Typography variant="body2" color="text.secondary">
                🔒 Session-based processing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                ⚕️ For clinical use only
              </Typography>
              <Typography variant="body2" color="text.secondary">
                🏥 Not a diagnostic tool
              </Typography>
            </Box>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;