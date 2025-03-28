import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Grid, Paper, CircularProgress, LinearProgress, Divider, Button, TextField } from '@mui/material';
import { List, AutoSizer } from 'react-virtualized';
import axios from 'axios';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot } from '@mui/lab';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = 'http://localhost:8000';

function BufferHealthIndicator({ health, maxFrames, currentFrames }) {
  // Calculate color based on health (processed frames ratio)
  // Red: < 33% processed, Yellow: 33-66% processed, Green: > 66% processed
  const color = health < 0.33 ? 'error' : health < 0.66 ? 'warning' : 'success';
  
  // Calculate processed frames
  const processedFrames = maxFrames - currentFrames;
  
  return (
    <Box sx={{ width: '100%', mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
        <Typography variant="subtitle2">Processing Progress</Typography>
        <Typography variant="caption" color="textSecondary">
          {processedFrames} / {maxFrames} frames processed ({(health * 100).toFixed(1)}%)
        </Typography>
      </Box>
      <LinearProgress variant="determinate" value={health * 100} color={color} />
      <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5 }}>
        {currentFrames} frames waiting in buffer
      </Typography>
    </Box>
  );
}

function ContextTimeline({ context }) {
  if (!context || !context.context) return null;
  
  return (
    <Timeline sx={{ 
      [`& .MuiTimelineItem-root:before`]: {
        flex: 0,
        padding: 0
      }
    }}>
      {context.context.map((entry, index) => (
        <TimelineItem key={index}>
          <TimelineSeparator>
            <TimelineDot color={entry.processing_metadata?.buffer_position === context.context.length - 1 ? "primary" : "grey"} />
            <TimelineConnector />
          </TimelineSeparator>
          <TimelineContent sx={{ py: '12px', px: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
              {entry.frame_key && (
                <Paper 
                  elevation={2} 
                  sx={{ 
                    width: 400, 
                    height: 240, 
                    overflow: 'hidden',
                    flexShrink: 0,
                    bgcolor: 'background.paper',
                    position: 'relative'
                  }}
                >
                  <img
                    src={`${API_BASE_URL}/frame/${entry.frame_key}`}
                    alt={`Frame ${entry.frame_index}`}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                  <Box 
                    sx={{ 
                      position: 'absolute', 
                      bottom: 0, 
                      left: 0, 
                      right: 0, 
                      bgcolor: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '4px 8px',
                      fontSize: '0.875rem',
                      textAlign: 'center'
                    }}
                  >
                    {entry.metadata?.timestamp?.toFixed(1)}s
                  </Box>
                </Paper>
              )}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  Frame {entry.frame_index} at {entry.metadata?.timestamp?.toFixed(1)}s
                </Typography>
                <Paper 
                  sx={{ 
                    mt: 1, 
                    p: 2, 
                    maxHeight: 150, 
                    overflow: 'auto',
                    bgcolor: 'background.paper',
                    '& p': { m: 0 },
                    '& h1, & h2, & h3': { fontSize: '1rem', fontWeight: 'bold', mt: 1, mb: 0.5 }
                  }}
                >
                  <ReactMarkdown>
                    {entry.description || 'No description available'}
                  </ReactMarkdown>
                </Paper>
                <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                  Tokens: {entry.processing_metadata?.token_usage} | 
                  Processing Time: {entry.processing_metadata?.processing_duration?.toFixed(2)}s | 
                  Processed: {new Date(entry.processing_metadata?.processed_at * 1000).toLocaleTimeString()}
                </Typography>
              </Box>
            </Box>
          </TimelineContent>
        </TimelineItem>
      ))}
    </Timeline>
  );
}

function BufferStats({ stats }) {
  if (!stats) return null;
  
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6">Buffer Statistics</Typography>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Typography variant="subtitle2">Frames Processed</Typography>
          <Typography>{stats.total_frames_processed}</Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="subtitle2">Frames in Buffer</Typography>
          <Typography>{stats.frames_in_buffer}</Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="subtitle2">Frame Interval</Typography>
          <Typography>
            {stats.frame_interval ? `${stats.frame_interval.toFixed(1)}s` : 'N/A'}
          </Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="subtitle2">Processing Rate</Typography>
          <Typography>
            {stats.total_frames_processed > 0 && stats.frame_interval
              ? `${(stats.total_frames_processed / (stats.frame_interval * stats.total_frames_processed)).toFixed(1)} fps`
              : 'N/A'}
          </Typography>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="subtitle2">Token Usage</Typography>
          <Typography>{stats.token_usage} tokens</Typography>
        </Grid>
      </Grid>
    </Paper>
  );
}

function VideoUploadControls({ 
  onDataCleared 
}) {
  const [file, setFile] = useState(null);
  const [frameInterval, setFrameInterval] = useState(1.0);
  const [delayInterval, setDelayInterval] = useState(0.0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [status, setStatus] = useState('');
  const [debug, setDebug] = useState('');
  const [processLogs, setProcessLogs] = useState({ upload: [], process: [] });

  const ACCEPTED_VIDEO_TYPES = ['video/mp4', 'video/quicktime'];

  useEffect(() => {
    let interval;
    if (isProcessing) {
      // Poll for logs every second when processing
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/process_logs`);
          setProcessLogs(prevLogs => ({
            upload: [...prevLogs.upload, ...response.data.upload],
            process: [...prevLogs.process, ...response.data.process]
          }));
        } catch (error) {
          console.error('Error fetching logs:', error);
        }
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isProcessing]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && ACCEPTED_VIDEO_TYPES.includes(selectedFile.type)) {
      setFile(selectedFile);
      setStatus('');
      setDebug(`Selected file: ${selectedFile.name} (${selectedFile.type}, ${selectedFile.size} bytes)`);
    } else {
      setFile(null);
      setStatus('Please select an MP4 or MOV video file');
      setDebug('Invalid file type selected');
    }
  };

  const handleUploadAndProcess = async () => {
    try {
      if (!file) {
        setStatus('Please select a video file first');
        return;
      }

      // Clear previous logs
      setProcessLogs({ upload: [], process: [] });
      setStatus('Uploading video...');
      setDebug('Creating form data for upload...');
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('frame_interval', frameInterval.toString());
      formData.append('delay_interval', delayInterval.toString());
      
      setDebug('Sending upload request...');
      // Upload video
      const uploadResponse = await axios.post(`${API_BASE_URL}/upload`, formData);
      
      setDebug(`Upload response: ${JSON.stringify(uploadResponse.data)}`);
      
      if (uploadResponse.data.status === 'success') {
        setStatus('Starting processing...');
        setDebug('Sending start processing request...');
        
        // Start processing with URL-encoded form data
        const params = new URLSearchParams();
        params.append('file_path', uploadResponse.data.file_path);
        params.append('frame_interval', frameInterval.toString());
        params.append('delay_interval', delayInterval.toString());
        params.append('redis_prefix', 'test:');
        
        const processResponse = await axios.post(`${API_BASE_URL}/start_processing`, params, {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        });
        
        setDebug(`Process response: ${JSON.stringify(processResponse.data)}`);
        
        if (processResponse.data.status === 'success') {
          setStatus('Processing started successfully');
          setIsProcessing(true);
        }
      }
    } catch (error) {
      console.error('Upload/Process error:', error);
      setDebug(`Error details: ${JSON.stringify({
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers
      }, null, 2)}`);
      setStatus(`Error: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleStopProcessing = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/stop_processing`);
      if (response.data.status === 'success') {
        setStatus('Processing stopped');
        setIsProcessing(false);
      }
    } catch (error) {
      setStatus(`Error stopping processing: ${error.message}`);
    }
  };

  const handleClearData = async () => {
    try {
      setIsClearing(true);
      setStatus('Clearing all data...');
      
      const response = await axios.post(`${API_BASE_URL}/clear_data`);
      
      if (response.data.status === 'success') {
        // Reset component state
        setFile(null);
        setProcessLogs({ upload: [], process: [] });
        setIsProcessing(false);
        setDebug('');
        
        // Notify parent component to clear its state
        if (onDataCleared) {
          onDataCleared();
        }
        
        setStatus('All data cleared successfully');
      }
    } catch (error) {
      setStatus(`Error clearing data: ${error.message}`);
      console.error('Clear data error:', error);
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Video Upload & Processing Controls
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <input
            accept="video/mp4,video/quicktime"
            style={{ display: 'none' }}
            id="video-upload"
            type="file"
            onChange={handleFileChange}
            onClick={(e) => e.target.value = null}
          />
          <label htmlFor="video-upload">
            <Button 
              variant="contained" 
              component="span" 
              fullWidth
              sx={{ mb: 1 }}
            >
              {file ? file.name : 'Select Video (MP4/MOV)'}
            </Button>
          </label>
          {file && (
            <Typography variant="caption" color="textSecondary">
              File size: {(file.size / (1024 * 1024)).toFixed(2)} MB | Type: {file.type === 'video/quicktime' ? 'MOV' : 'MP4'}
            </Typography>
          )}
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Frame Interval (seconds)"
            type="number"
            value={frameInterval}
            onChange={(e) => setFrameInterval(parseFloat(e.target.value))}
            inputProps={{ step: 0.1, min: 0.1 }}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Delay Interval (seconds)"
            type="number"
            value={delayInterval}
            onChange={(e) => setDelayInterval(parseFloat(e.target.value))}
            inputProps={{ step: 0.1, min: 0 }}
          />
        </Grid>
        <Grid item xs={12}>
          {!isProcessing ? (
            <Grid container spacing={2}>
              <Grid item xs={8}>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={handleUploadAndProcess}
                  disabled={!file || isClearing}
                >
                  Upload & Start Processing
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="contained"
                  color="error"
                  fullWidth
                  onClick={handleClearData}
                  disabled={isClearing}
                >
                  Clear All Data
                </Button>
              </Grid>
            </Grid>
          ) : (
            <Button
              variant="contained"
              color="error"
              fullWidth
              onClick={handleStopProcessing}
            >
              Stop Processing
            </Button>
          )}
        </Grid>
        {status && (
          <Grid item xs={12}>
            <Typography color={status.includes('Error') ? 'error' : 'textSecondary'}>
              {status}
            </Typography>
          </Grid>
        )}
        
        {debug && (
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>Debug Info:</Typography>
            <Paper sx={{ p: 1, bgcolor: 'grey.900', maxHeight: 200, overflow: 'auto' }}>
              <Typography
                variant="caption"
                component="pre"
                sx={{ 
                  m: 0,
                  color: 'grey.300',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all'
                }}
              >
                {debug}
              </Typography>
            </Paper>
          </Grid>
        )}

        {isProcessing && (
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>Process Logs:</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="caption">Upload Process:</Typography>
                <Paper sx={{ p: 1, bgcolor: 'grey.900', height: 200, overflow: 'auto' }}>
                  <Typography
                    variant="caption"
                    component="pre"
                    sx={{ 
                      m: 0,
                      color: 'grey.300',
                      fontFamily: 'monospace',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all'
                    }}
                  >
                    {processLogs.upload.join('\n')}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption">Processing:</Typography>
                <Paper sx={{ p: 1, bgcolor: 'grey.900', height: 200, overflow: 'auto' }}>
                  <Typography
                    variant="caption"
                    component="pre"
                    sx={{ 
                      m: 0,
                      color: 'grey.300',
                      fontFamily: 'monospace',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all'
                    }}
                  >
                    {processLogs.process.join('\n')}
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
}

function App() {
  const [frames, setFrames] = useState([]);
  const [totalFrames, setTotalFrames] = useState(0);
  const [loading, setLoading] = useState(true);
  const [context, setContext] = useState(null);
  const [error, setError] = useState(null);
  const [contextError, setContextError] = useState(null);

  useEffect(() => {
    fetchFrames();
    fetchContext();
    
    // Poll for updates every 500ms for more responsive updates
    const interval = setInterval(() => {
      fetchFrames();
      fetchContext();
    }, 500);
    
    return () => clearInterval(interval);
  }, []);

  const fetchFrames = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/frames?prefix=test:`);
      setFrames(response.data.frames);
      setTotalFrames(response.data.total_frames);
      setError(null);
    } catch (err) {
      console.error('Error fetching frames:', err);
      setError('Error fetching frames: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchContext = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/context`);
      if (response.data.error) {
        console.error('Context error:', response.data.error);
        setContextError(response.data.error);
      } else {
        setContext(response.data);
        setContextError(null);
      }
    } catch (err) {
      console.error('Error fetching context:', err);
      setContextError(err.message);
    }
  };

  const renderFrame = ({ index, key, style }) => {
    const frame = frames[index];
    if (!frame) return null;

    return (
      <Box key={key} style={{ ...style, display: 'inline-block' }}>
        <Paper 
          elevation={3} 
          sx={{ 
            p: 2, 
            m: 1, 
            width: '300px',
            height: '350px', 
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          <img 
            src={`${API_BASE_URL}/frame/${frame.key}`}
            alt={`Frame ${frame.frame_index}`}
            style={{ 
              width: '100%', 
              height: '200px', 
              objectFit: 'contain',
              marginBottom: '10px'
            }}
          />
          <Typography variant="subtitle2" gutterBottom>
            Frame {frame.frame_index} - {frame.timestamp.toFixed(2)}s
          </Typography>
          <Box sx={{ overflow: 'auto', fontSize: '0.8rem' }}>
            <pre>{JSON.stringify(frame.metadata, null, 2)}</pre>
          </Box>
        </Paper>
      </Box>
    );
  };

  const renderContext = () => {
    if (!context && !contextError) return null;

    return (
      <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Current Context
        </Typography>
        {contextError ? (
          <Typography color="error">
            Error loading context: {contextError}
            <br />
            Please ensure the data directory exists and is writable.
          </Typography>
        ) : (
          <Box sx={{ overflow: 'auto', maxHeight: '200px' }}>
            <pre>{JSON.stringify(context, null, 2)}</pre>
          </Box>
        )}
      </Paper>
    );
  };

  const handleDataCleared = () => {
    setFrames([]);
    setTotalFrames(0);
    setContext(null);
    setError(null);
    setContextError(null);
    // Force immediate refresh of data
    fetchFrames();
    fetchContext();
  };

  if (loading) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container>
        <Typography color="error" variant="h6">
          {error}
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Perspectiv Screen Understanding Demo
        </Typography>
        
        <VideoUploadControls 
          onDataCleared={handleDataCleared}
        />
        
        {loading ? (
          <CircularProgress />
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6">Frame Information</Typography>
                <Typography>Total Frames: {totalFrames}</Typography>
                {context?.buffer_stats && (
                  <>
                    <BufferHealthIndicator health={context.buffer_stats.buffer_health} maxFrames={totalFrames} currentFrames={context.buffer_stats.frames_in_buffer} />
                    <BufferStats stats={context.buffer_stats} />
                  </>
                )}
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2, maxHeight: 600, overflow: 'auto' }}>
                <Typography variant="h6">Context Timeline</Typography>
                {contextError ? (
                  <Typography color="error">{contextError}</Typography>
                ) : (
                  <ContextTimeline context={context} />
                )}
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
    </Container>
  );
}

export default App; 