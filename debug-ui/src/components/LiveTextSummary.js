import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography, CircularProgress } from '@mui/material';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';

const API_BASE_URL = 'http://localhost:8000';

function LiveTextSummary() {
  const [summary, setSummary] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let interval;
    
    const fetchSummary = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/text_summary`);
        const data = await response.json();
        
        if (data.status === 'success') {
          setSummary(data.summary);
          setIsProcessing(data.is_processing);
          setError(null);
        } else {
          setError('Failed to fetch summary');
        }
      } catch (err) {
        setError('Error fetching summary');
        console.error('Error:', err);
      }
    };

    // Initial fetch
    fetchSummary();

    // Set up polling interval
    interval = setInterval(fetchSummary, 1000);

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, []);

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Live Text Summary
        </Typography>
        {isProcessing && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FiberManualRecordIcon 
              sx={{ 
                color: 'error.main',
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                  '100%': { opacity: 1 },
                },
              }} 
            />
            <Typography variant="caption" color="error">
              Live
            </Typography>
          </Box>
        )}
      </Box>

      {error ? (
        <Typography color="error">{error}</Typography>
      ) : !summary ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <Typography>
          {summary}
        </Typography>
      )}
    </Paper>
  );
}

export default LiveTextSummary; 