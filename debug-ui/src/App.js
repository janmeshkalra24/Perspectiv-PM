import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Box, Container, Typography, Grid, Paper, CircularProgress, LinearProgress, Divider, Button, TextField, IconButton, Tooltip } from '@mui/material';
import { List, AutoSizer } from 'react-virtualized';
import axios from 'axios';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot } from '@mui/lab';
import ReactMarkdown from 'react-markdown';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import StopIcon from '@mui/icons-material/Stop';
import ForceGraph2D from 'react-force-graph-2d';
import CloseIcon from '@mui/icons-material/Close';

const API_BASE_URL = 'http://localhost:8001';

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

function ChatMessage({ message, isUser }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const handleSpeak = () => {
    if ('speechSynthesis' in window) {
      if (isSpeaking) {
        window.speechSynthesis.cancel();
        setIsSpeaking(false);
      } else {
        const utterance = new SpeechSynthesisUtterance(message.content);
        utterance.onend = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
        setIsSpeaking(true);
      }
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Paper
        sx={{
          p: 2,
          maxWidth: '70%',
          bgcolor: isUser ? 'primary.main' : 'background.paper',
          color: isUser ? 'primary.contrastText' : 'text.primary',
          borderRadius: 2,
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            {isUser ? 'You' : 'Assistant'}
          </Typography>
          {!isUser && (
            <Tooltip title={isSpeaking ? "Stop speaking" : "Speak response"}>
              <IconButton size="small" onClick={handleSpeak} color={isSpeaking ? "error" : "default"}>
                {isSpeaking ? <StopIcon /> : <VolumeUpIcon />}
              </IconButton>
            </Tooltip>
          )}
        </Box>
        {isUser ? (
          <Typography>{message.content}</Typography>
        ) : (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        )}
      </Paper>
    </Box>
  );
}

function ChatInterface({ context, currentFrame }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [sttError, setSTTError] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const chatContainerRef = useRef(null);
  const recognitionRef = useRef(null);

  // Fetch graph data periodically
  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/visualization_graph`);
        setGraphData(response.data);
      } catch (error) {
        console.error('Error fetching graph data:', error);
      }
    };

    // Initial fetch
    fetchGraphData();

    // Set up polling
    const interval = setInterval(fetchGraphData, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const contextData = {
        history: context?.context || [],
        currentFrame: currentFrame,
        messages: [...messages, userMessage]
      };

      const response = await axios.post(`${API_BASE_URL}/chat`, contextData);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.response
      }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      setSTTError('Speech recognition is not supported in your browser. Please try using Chrome, Edge, or Safari.');
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      setSTTError(null);
      try {
        recognitionRef.current.start();
      } catch (error) {
        console.error('Speech recognition start error:', error);
        setSTTError('Error starting speech recognition. Please try again.');
      }
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 3, display: 'flex', gap: 2 }}>
      {/* Chat section */}
      <Box sx={{ flex: 2 }}>
        <Typography variant="h6" gutterBottom>
          Chat Interface
        </Typography>
        <Box
          ref={chatContainerRef}
          sx={{
            height: '300px',
            overflowY: 'auto',
            mb: 2,
            p: 2,
            bgcolor: 'background.default',
            borderRadius: 1
          }}
        >
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message}
              isUser={message.role === 'user'}
            />
          ))}
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          )}
        </Box>
        <Grid container spacing={1} alignItems="center">
          <Grid item xs>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask about what's happening in the screen recording..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={isLoading || isRecording}
              error={!!sttError}
              helperText={sttError}
            />
          </Grid>
          <Grid item>
            <Tooltip title={isRecording ? "Stop recording" : "Start voice input"}>
              <IconButton
                color={isRecording ? "error" : "primary"}
                onClick={toggleRecording}
                disabled={isLoading}
                sx={{
                  animation: isRecording ? 'pulse 1.5s ease-in-out infinite' : 'none',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                    '100%': { opacity: 1 },
                  },
                }}
              >
                {isRecording ? <StopIcon /> : <MicIcon />}
              </IconButton>
            </Tooltip>
          </Grid>
          <Grid item>
            <IconButton
              color="primary"
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
            >
              <SendIcon />
            </IconButton>
          </Grid>
        </Grid>
      </Box>

      {/* Knowledge Graph section */}
      <Box sx={{ flex: 1, minWidth: 0, height: '400px' }}>
        <KnowledgeGraphVisualization graphData={graphData} />
      </Box>
    </Paper>
  );
}

function KnowledgeGraphVisualization({ graphData }) {
  const fgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);

  // Handle node click
  const handleNodeClick = useCallback(node => {
    setSelectedNode(node);
    setNodeDetails({
      label: node.label,
      type: node.type,
      group: node.group,
      details: {
        timestamp: node.timestamp,
        description: node.description
      }
    });
  }, []);

  // Color scheme for different node types
  const getNodeColor = node => {
    const colors = {
      root: '#FF9800',
      timeline: '#4CAF50',
      frame: '#81C784',
      applications: '#2196F3',
      technical_terms: '#9C27B0',
      user_actions: '#3F51B5',
      tasks: '#FF5722',
      warnings: '#f44336'
    };
    return colors[node.group] || '#999';
  };

  // Node size based on type
  const getNodeSize = node => {
    switch (node.type) {
      case 'root':
        return 15;
      case 'container':
        return 12;
      case 'frame':
        return 8;
      case 'insight':
        return 6;
      default:
        return 8;
    }
  };

  return (
    <Paper sx={{ 
      p: 1, 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      <Typography variant="subtitle2" gutterBottom>
        Knowledge Graph Insights
      </Typography>
      
      <Box sx={{ 
        flex: 1,
        minHeight: 0,
        height: '350px',
        position: 'relative',
        '& canvas': {
          borderRadius: 1
        }
      }}>
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          nodeLabel={node => node.label}
          nodeColor={getNodeColor}
          nodeRelSize={node => getNodeSize(node)}
          linkDirectional={true}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={0.005}
          backgroundColor="#1a1b1e"
          linkWidth={1.5}
          linkColor={() => 'rgba(255,255,255,0.2)'}
          d3VelocityDecay={0.3}
          cooldownTicks={50}
          onNodeClick={handleNodeClick}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = String(node.label || '');
            const fontSize = 12/globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.fillStyle = getNodeColor(node);
            ctx.beginPath();
            ctx.arc(node.x, node.y, getNodeSize(node), 0, 2 * Math.PI, false);
            ctx.fill();
            
            if (node === selectedNode) {
              ctx.strokeStyle = '#fff';
              ctx.lineWidth = 2;
              ctx.stroke();
            }
            
            const maxLineLength = 20;
            const words = label.split(' ');
            const lines = [];
            let currentLine = '';
            
            words.forEach(word => {
              if (currentLine.length + word.length > maxLineLength) {
                lines.push(currentLine);
                currentLine = word;
              } else {
                currentLine = currentLine ? `${currentLine} ${word}` : word;
              }
            });
            if (currentLine) {
              lines.push(currentLine);
            }
            
            const lineHeight = fontSize * 1.2;
            const maxWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
            
            // Draw label background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.fillRect(
              node.x - maxWidth/2 - 2,
              node.y + 10,
              maxWidth + 4,
              lineHeight * lines.length + 4
            );
            
            // Draw text
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = 'white';
            lines.forEach((line, i) => {
              ctx.fillText(
                line,
                node.x,
                node.y + 10 + lineHeight * (i + 0.5)
              );
            });
          }}
        />
        
        {/* Node details panel */}
        {nodeDetails && (
          <Paper
            sx={{
              position: 'absolute',
              top: 10,
              right: 10,
              width: '250px',
              maxHeight: '330px',
              overflow: 'auto',
              backgroundColor: 'rgba(0, 0, 0, 0.85)',
              color: 'white',
              p: 2,
              borderRadius: 1
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              {nodeDetails.label}
            </Typography>
            <Typography variant="caption" component="div" sx={{ mb: 1, opacity: 0.7 }}>
              Type: {nodeDetails.type}
            </Typography>
            {nodeDetails.details.timestamp && (
              <Typography variant="caption" component="div" sx={{ mb: 1, opacity: 0.7 }}>
                Timestamp: {nodeDetails.details.timestamp.toFixed(1)}s
              </Typography>
            )}
            {nodeDetails.details.description && (
              <>
                <Divider sx={{ my: 1, borderColor: 'rgba(255,255,255,0.1)' }} />
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {nodeDetails.details.description}
                </Typography>
              </>
            )}
            <IconButton
              size="small"
              sx={{ position: 'absolute', top: 8, right: 8, color: 'white' }}
              onClick={() => setNodeDetails(null)}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </Paper>
        )}
      </Box>

      {/* Legend */}
      <Box sx={{ mt: 1, fontSize: '0.75rem' }}>
        <Typography variant="caption" component="div" sx={{ mb: 0.5 }}>
          Legend:
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#4CAF50' }} />
              <Typography variant="caption">Timeline</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#2196F3' }} />
              <Typography variant="caption">Applications</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#9C27B0' }} />
              <Typography variant="caption">Technical Terms</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#3F51B5' }} />
              <Typography variant="caption">User Actions</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#FF5722' }} />
              <Typography variant="caption">Tasks</Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#f44336' }} />
              <Typography variant="caption">Warnings</Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>
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
  const [currentFrame, setCurrentFrame] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

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

  useEffect(() => {
    if (context?.context?.length > 0) {
      const lastFrame = context.context[context.context.length - 1];
      setCurrentFrame(lastFrame);
    }
  }, [context]);

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

  const fetchGraphData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/graph_data`);
      setGraphData(response.data);
    } catch (error) {
      console.error('Error fetching graph data:', error);
    }
  };

  useEffect(() => {
    // Add graph data fetching to the existing polling
    const interval = setInterval(() => {
      fetchGraphData();
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

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
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
      <Container maxWidth={false} sx={{ flex: 1, py: 2, display: 'flex', flexDirection: 'column' }}>
        <Grid container spacing={2} sx={{ flex: 1, minHeight: 0 }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="h5" gutterBottom>Debug UI</Typography>
              <VideoUploadControls onDataCleared={handleDataCleared} />
              <BufferStats stats={context?.buffer_stats} />
            </Paper>
            
            <Paper sx={{ p: 2, mb: 2, flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <Typography variant="h6" gutterBottom>Context Timeline</Typography>
              <Box sx={{ flex: 1, overflow: 'auto' }}>
                <ContextTimeline context={context} />
              </Box>
            </Paper>
            
            <ChatInterface 
              context={context} 
              currentFrame={currentFrame} 
            />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default App; 