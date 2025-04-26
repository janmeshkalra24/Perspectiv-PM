import React, { useState, useEffect, useRef } from 'react';
import { Box, Container, Typography, Grid, Paper, CircularProgress, LinearProgress, Divider, Button, TextField, IconButton, Tooltip } from '@mui/material';
import { List, AutoSizer } from 'react-virtualized';
import axios from 'axios';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot } from '@mui/lab';
import ReactMarkdown from 'react-markdown';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import StopIcon from '@mui/icons-material/Stop';
import MindMap from './components/MindMap';
import RefreshIcon from '@mui/icons-material/Refresh';

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
  
  // Format timestamp consistently 
  const formatTimestamp = (seconds) => {
    if (seconds === undefined || seconds === null) return "unknown time";
    
    const minutes = Math.floor(seconds / 60);
    const remainingSecs = Math.floor(seconds % 60);
    return `${minutes}:${remainingSecs.toString().padStart(2, '0')}`;
  };
  
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
                    {entry.metadata?.timestamp ? formatTimestamp(entry.metadata.timestamp) : 'N/A'}
                  </Box>
                </Paper>
              )}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  Frame {entry.frame_index} at {entry.metadata?.timestamp ? formatTimestamp(entry.metadata.timestamp) : 'N/A'}
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
  const chatContainerRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Initialize speech recognition with broader browser support
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onstart = () => {
        setIsRecording(true);
        setSTTError(null);
      };
      
      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');
        setInput(transcript);
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
        switch (event.error) {
          case 'not-allowed':
            setSTTError('Microphone access denied. Please allow microphone access in your browser settings.');
            break;
          case 'no-speech':
            setSTTError('No speech detected. Please try speaking again.');
            break;
          case 'network':
            setSTTError('Network error occurred. Please check your connection.');
            break;
          default:
            setSTTError(`Error: ${event.error}`);
        }
      };
      
      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
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
    <Paper sx={{ p: 2, mt: 3 }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6">Chat Interface</Typography>
      </Box>
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
    </Paper>
  );
}

function TextStreamInput({ onDataCleared }) {
  const [text, setText] = useState('');
  const [chunkSize, setChunkSize] = useState(1);
  const [delayInterval, setDelayInterval] = useState(1.0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [status, setStatus] = useState('');
  const [debug, setDebug] = useState('');

  const handleStartProcessing = async () => {
    try {
      if (!text) {
        setStatus('Please enter some text first');
        return;
      }

      setStatus('Starting text processing...');
      setDebug('Sending start processing request...');
      
      // Start processing with URL-encoded form data
      const params = new URLSearchParams();
      params.append('text', text);
      params.append('chunk_size', chunkSize.toString());
      params.append('delay_interval', delayInterval.toString());
      params.append('redis_prefix', 'text:');
      
      const processResponse = await axios.post(`${API_BASE_URL}/start_text_processing`, params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      
      setDebug(`Process response: ${JSON.stringify(processResponse.data)}`);
      
      if (processResponse.data.status === 'success') {
        setStatus('Text processing started successfully');
        setIsProcessing(true);
      }
    } catch (error) {
      console.error('Process error:', error);
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
      const response = await axios.post(`${API_BASE_URL}/stop_text_processing`);
      if (response.data.status === 'success') {
        setStatus('Text processing stopped');
        setIsProcessing(false);
      }
    } catch (error) {
      setStatus(`Error stopping processing: ${error.message}`);
    }
  };

  const handleClearData = async () => {
    try {
      setIsClearing(true);
      setStatus('Clearing all text data...');
      
      const response = await axios.post(`${API_BASE_URL}/clear_text_data`);
      
      if (response.data.status === 'success') {
        setText('');
        setIsProcessing(false);
        setDebug('');
        
        if (onDataCleared) {
          onDataCleared();
        }
        
        setStatus('All text data cleared successfully');
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
        Text Stream Input
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Enter text to stream"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isProcessing}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Chunk Size (sentences)"
            type="number"
            value={chunkSize}
            onChange={(e) => setChunkSize(parseInt(e.target.value))}
            inputProps={{ step: 1, min: 1 }}
            disabled={isProcessing}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Delay Interval (seconds)"
            type="number"
            value={delayInterval}
            onChange={(e) => setDelayInterval(parseFloat(e.target.value))}
            inputProps={{ step: 0.1, min: 0.1 }}
            disabled={isProcessing}
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
                  onClick={handleStartProcessing}
                  disabled={!text || isClearing}
                >
                  Start Text Processing
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
                  Clear Text Data
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
      </Grid>
    </Paper>
  );
}

function LiveTextSummary({ isProcessing }) {
  const [summary, setSummary] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    let interval;
    if (isProcessing) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/text_summary`);
          if (response.data.summary) {
            setSummary(response.data.summary);
          }
        } catch (error) {
          console.error('Error fetching summary:', error);
          setError(error.message);
        }
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isProcessing]);

  return (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ flex: 1 }}>
          Live Text Summary
        </Typography>
        {isProcessing && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: 'error.main',
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.4 },
                  '100%': { opacity: 1 }
                }
              }}
            />
            <Typography variant="caption" color="error">
              Live
            </Typography>
          </Box>
        )}
      </Box>
      
      {error ? (
        <Typography color="error">Error: {error}</Typography>
      ) : (
        <Typography>
          {summary || (isProcessing ? 'Processing text...' : 'No text being processed')}
        </Typography>
      )}
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
  const [mindMapData, setMindMapData] = useState(null);
  const [lastContextHash, setLastContextHash] = useState('');
  const [updateTimeout, setUpdateTimeout] = useState(null);
  const [manualRefreshEnabled, setManualRefreshEnabled] = useState(true);
  const [isTextProcessing, setIsTextProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

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

  // Add fallback data for mind map in case API fails
  const createDefaultMindMap = () => {
    return {
      nodes: [
        { id: 'main', name: 'Screen Recording', type: 'app' },
        { id: 'summary', name: 'Summary', type: 'summary', details: 'Analysis of screen recording content' },
        { id: 'tasks', name: 'Tasks', type: 'tasks', details: 'Identified tasks from the recording' },
        { id: 'terms', name: 'Technical Terms', type: 'terms', details: 'Technical terminology found in the recording' }
      ],
      links: [
        { source: 'main', target: 'summary' },
        { source: 'main', target: 'tasks' },
        { source: 'main', target: 'terms' }
      ]
    };
  };

  // Add fallback data for warnings in case API fails
  const createDefaultWarnings = () => {
    return {
      redFlags: [],
      warnings: [
        {
          title: "No content analyzed yet",
          time: new Date().toLocaleTimeString(),
          details: "Upload a video to analyze content and generate insights."
        }
      ]
    };
  };

  // Add a more sophisticated mind map data structure
  const createEnhancedMindMap = (contextData) => {
    if (!contextData || !contextData.context || contextData.context.length === 0) {
      return createDefaultMindMap();
    }

    try {
      // Create multiple disconnected clusters/categories for better organization
      const mindMap = {
        nodes: [
          // Main engineering categories - not all connected to a central node
          { id: 'backend', name: 'Backend Systems', type: 'app', details: 'Backend systems and architecture issues' },
          { id: 'frontend', name: 'Frontend & UI', type: 'app', details: 'Frontend, UI, and user experience concerns' },
          { id: 'infra', name: 'Infrastructure', type: 'app', details: 'Infrastructure, deployment, and DevOps concerns' },
          { id: 'process', name: 'Process & Planning', type: 'app', details: 'Development process, planning, and team concerns' },
          { id: 'product', name: 'Product Features', type: 'app', details: 'Product features, requirements, and deliverables' },
        ],
        links: []
      };
      
      // Advanced technical terminology by category for engineering context
      const advancedTerminology = {
        backend: [
          { term: 'microservices', def: 'Architecture pattern where applications are built as independent services' },
          { term: 'serverless', def: 'Cloud computing model where the cloud provider manages server infrastructure' },
          { term: 'ORM', def: 'Object-Relational Mapping - technique for converting data between incompatible type systems' },
          { term: 'GraphQL', def: 'Query language for APIs and runtime for executing those queries' },
          { term: 'caching', def: 'Storing copies of data in a high-speed data storage layer' },
          { term: 'horizontal scaling', def: 'Adding more machines to a system to handle increased load' },
          { term: 'sharding', def: 'Database partitioning technique to distribute data across multiple machines' },
          { term: 'message queue', def: 'Communication method between processes, services or systems' },
          { term: 'webhooks', def: 'User-defined HTTP callbacks triggered by specific events' },
          { term: 'idempotency', def: 'Property where an operation can be applied multiple times without changing the result' }
        ],
        frontend: [
          { term: 'state management', def: 'Pattern for managing application state across components' },
          { term: 'code splitting', def: 'Technique to split code into various bundles for on-demand loading' },
          { term: 'lazy loading', def: 'Design pattern to defer initialization of resources until needed' },
          { term: 'design system', def: 'Collection of reusable components guided by standards' },
          { term: 'accessibility', def: 'Practice of making applications usable by people with disabilities' },
          { term: 'SSR', def: 'Server-Side Rendering - rendering pages on the server instead of browser' },
          { term: 'CSR', def: 'Client-Side Rendering - rendering pages directly in the browser with JavaScript' },
          { term: 'WASM', def: 'WebAssembly - binary instruction format for stack-based virtual machines' },
          { term: 'SPA', def: 'Single-Page Application - web app that loads a single HTML page' },
          { term: 'PWA', def: 'Progressive Web App - web app that offers native-app-like experience' }
        ],
        infrastructure: [
          { term: 'kubernetes', def: 'Container orchestration system for automating deployment and scaling' },
          { term: 'CI/CD', def: 'Continuous Integration/Continuous Deployment - automating build, test, and deployment' },
          { term: 'infrastructure as code', def: 'Managing infrastructure through machine-readable definition files' },
          { term: 'blue-green deployment', def: 'Deployment strategy with two identical production environments' },
          { term: 'observability', def: 'Measuring the internal state of a system from its outputs' },
          { term: 'containerization', def: 'OS-level virtualization to deploy and run applications without launching VMs' },
          { term: 'service mesh', def: 'Infrastructure layer for service-to-service communication' },
          { term: 'chaos engineering', def: 'Practice of experimenting on a system to build confidence in its capabilities' },
          { term: 'zero-trust security', def: 'Security concept centered on the belief that organizations should not trust anything' },
          { term: 'autoscaling', def: 'Automatically adjusting computational resources based on traffic' }
        ],
        process: [
          { term: 'technical debt', def: 'Cost of additional work caused by choosing easy solution now over better approach' },
          { term: 'velocity', def: 'Measure of work completed in a sprint or time period' },
          { term: 'scrum', def: 'Framework for managing complex knowledge work with emphasis on software development' },
          { term: 'kanban', def: 'Method for managing knowledge work with emphasis on continuous delivery' },
          { term: 'pair programming', def: 'Development technique where two programmers work together on one workstation' },
          { term: 'story point', def: 'Abstract measure of effort required to implement a user story' },
          { term: 'code review', def: 'Systematic examination of code to find and fix mistakes overlooked in development' },
          { term: 'sprint planning', def: 'Event in Scrum where team decides what to complete in coming sprint' },
          { term: 'definition of done', def: 'Shared understanding of what it means for work to be complete' },
          { term: 'retrospective', def: 'Meeting held after completion of an iteration to reflect on what happened' }
        ],
        product: [
          { term: 'MVP', def: 'Minimum Viable Product - version with just enough features to satisfy early customers' },
          { term: 'user story', def: 'Informal explanation of a software feature from end-user perspective' },
          { term: 'acceptance criteria', def: 'Conditions that a software product must meet to be accepted by users' },
          { term: 'feature flag', def: 'Technique to turn functionality on/off during runtime without deployment' },
          { term: 'A/B testing', def: 'Method of comparing two versions of a webpage or app against each other' },
          { term: 'product backlog', def: 'Prioritized list of work for the development team derived from roadmap' },
          { term: 'user persona', def: 'Fictional character created to represent a user type that might use a product' },
          { term: 'user journey', def: 'Series of steps a user takes to achieve a meaningful goal with your product' },
          { term: 'OKR', def: 'Objective and Key Results - framework for defining and tracking objectives and outcomes' },
          { term: 'stakeholder management', def: 'Process of identifying and engaging people affected by the project' }
        ]
      };

      // Look for discussion around specific terms in the context
      function findTermMentionsInContext(term, context) {
        let mentions = [];
        context.forEach(frame => {
          if (!frame.description) return;
          
          // Check for mentions of the term
          if (frame.description.toLowerCase().includes(term.toLowerCase())) {
            // Extract the sentence containing the term
            const sentences = frame.description.split(/[.!?]+/);
            for (const sentence of sentences) {
              if (sentence.toLowerCase().includes(term.toLowerCase())) {
                const formattedTimestamp = frame.metadata?.timestamp 
                  ? `${Math.floor(frame.metadata.timestamp / 60)}:${(frame.metadata.timestamp % 60).toFixed(0).padStart(2, '0')}`
                  : 'unknown time';
                  
                mentions.push({
                  sentence: sentence.trim(),
                  timestamp: formattedTimestamp
                });
              }
            }
          }
        });
        return mentions;
      }
      
      // Create clear buckets for different role-related information based on PM needs
      const roleInfo = {
        engineers: new Set(),
        managers: new Set(),
        stakeholders: new Set()
      };
      
      const actionItems = new Set();
      const blockers = new Set();
      const decisions = new Set();
      const openQuestions = new Set();
      
      // Topic discussions with specific engineering focus
      const discussedTopics = {
        backend: new Set(),
        frontend: new Set(),
        infrastructure: new Set(),
        process: new Set(),
        product: new Set()
      };
      
      // Extract technical terms from context data
      const mentionedTerms = new Map();
      
      // Term detection for advanced terminology
      Object.keys(advancedTerminology).forEach(category => {
        advancedTerminology[category].forEach(termObj => {
          const foundMentions = findTermMentionsInContext(termObj.term, contextData.context);
          if (foundMentions.length > 0) {
            // Save the term with its definition and any context-specific usage
            mentionedTerms.set(termObj.term, {
              definition: termObj.def,
              category,
              mentions: foundMentions
            });
          }
        });
      });
      
      // Extract role-specific info and PM-focused insights
      contextData.context.forEach(frame => {
        if (!frame.description) return;
        
        const description = frame.description;
        const timestamp = frame.metadata?.timestamp 
          ? `${Math.floor(frame.metadata.timestamp / 60)}:${(frame.metadata.timestamp % 60).toFixed(0).padStart(2, '0')}`
          : 'unknown time';
          
        // Extract role information
        if (/\b(?:engineer|developer|programmer|coder)\b/i.test(description)) {
          const nameMatch = description.match(/\b([A-Z][a-z]+ (?:[A-Z][a-z]+)?)(?:\s+(?:is|as|the)\s+(?:an\s+)?(?:engineer|developer))/i);
          if (nameMatch && nameMatch[1]) {
            roleInfo.engineers.add(nameMatch[1]);
          }
        }
        
        if (/\b(?:manager|lead|PM|product owner)\b/i.test(description)) {
          const nameMatch = description.match(/\b([A-Z][a-z]+ (?:[A-Z][a-z]+)?)(?:\s+(?:is|as|the)\s+(?:a\s+)?(?:manager|lead|PM))/i);
          if (nameMatch && nameMatch[1]) {
            roleInfo.managers.add(nameMatch[1]);
          }
        }
        
        if (/\b(?:stakeholder|client|customer|user|executive)\b/i.test(description)) {
          const nameMatch = description.match(/\b([A-Z][a-z]+ (?:[A-Z][a-z]+)?)(?:\s+(?:is|as|the)\s+(?:a\s+)?(?:stakeholder|client))/i);
          if (nameMatch && nameMatch[1]) {
            roleInfo.stakeholders.add(nameMatch[1]);
          }
        }
        
        // Extract action items
        if (/\b(?:will|should|must|going to|need to|has to|assigned to)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:will|should|must|going to|need to|has to|assigned to)\b/i.test(sentence)) {
              actionItems.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        // Extract blockers
        if (/\b(?:blocker|blocking|blocked|impediment|obstacle|stuck)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:blocker|blocking|blocked|impediment|obstacle|stuck)\b/i.test(sentence)) {
              blockers.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        // Extract decisions
        if (/\b(?:decided|agreed|concluded|determined|resolved|approved|chose|finalized|confirmed)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:decided|agreed|concluded|determined|resolved|approved|chose|finalized|confirmed)\b/i.test(sentence)) {
              decisions.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        // Extract open questions
        if (/\b(?:question|wondering|unclear|not sure|don't know|need to figure out|tbd|to be determined)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:question|wondering|unclear|not sure|don't know|need to figure out|tbd|to be determined)\b/i.test(sentence)) {
              openQuestions.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        // Categorize discussions by topic
        if (/\b(?:api|database|backend|server|microservice|endpoint)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:api|database|backend|server|microservice|endpoint)\b/i.test(sentence)) {
              discussedTopics.backend.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        if (/\b(?:ui|ux|interface|frontend|css|design|component|user experience)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:ui|ux|interface|frontend|css|design|component|user experience)\b/i.test(sentence)) {
              discussedTopics.frontend.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        if (/\b(?:deploy|infrastructure|cloud|aws|azure|kubernetes|docker|ci\/cd|devops)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:deploy|infrastructure|cloud|aws|azure|kubernetes|docker|ci\/cd|devops)\b/i.test(sentence)) {
              discussedTopics.infrastructure.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        if (/\b(?:sprint|agile|process|velocity|timeline|deadline|scrum|kanban|standup|meeting)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:sprint|agile|process|velocity|timeline|deadline|scrum|kanban|standup|meeting)\b/i.test(sentence)) {
              discussedTopics.process.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
        
        if (/\b(?:feature|product|requirement|user story|acceptance criteria|mvp|roadmap|epic|release)\b/i.test(description)) {
          const sentences = description.split(/[.!?]+/);
          for (const sentence of sentences) {
            if (/\b(?:feature|product|requirement|user story|acceptance criteria|mvp|roadmap|epic|release)\b/i.test(sentence)) {
              discussedTopics.product.add(`${sentence.trim()} (${timestamp})`);
            }
          }
        }
      });

      // Utility function to create nodes and links
      let idCounter = 1;
      function addSubNodes(parentId, items, type, prefix) {
        // Limit to 5 items per parent to avoid overloading
        const limitedItems = Array.from(items).slice(0, 5);
        
        limitedItems.forEach(item => {
          const id = `${prefix}-${idCounter++}`;
          
          // Extract a concise name (up to first comma or parenthesis or after 20 chars)
          let conciseName = '';
          if (typeof item === 'string') {
            // Remove timestamp if present at the end in parentheses
            const withoutTimestamp = item.split(' (')[0];
            // Remove any LLM prefixes
            const withoutPrefix = withoutTimestamp
              .replace(/^(?:here is|here's|this is|i have|i've|i will|i'll|let me|let's)[^:]*:\s*/i, '')
              .trim();
            // Take first 25 chars or up to first punctuation that might separate ideas
            const firstPunctuation = withoutPrefix.search(/[,;:]/);
            if (firstPunctuation > 0 && firstPunctuation < 25) {
              conciseName = withoutPrefix.substring(0, firstPunctuation);
            } else {
              conciseName = withoutPrefix.substring(0, Math.min(25, withoutPrefix.length));
            }
            
            // Add ellipsis if truncated
            if (conciseName.length < withoutPrefix.length) {
              conciseName += '...';
            }
          } else {
            conciseName = String(item).substring(0, 25);
          }
          
          mindMap.nodes.push({
            id,
            name: conciseName,
            type,
            details: typeof item === 'string' ? item : `${item}`
          });
          mindMap.links.push({ source: parentId, target: id });
        });
      }
      
      // Add technical terms nodes (only if there are mentions)
      if (mentionedTerms.size > 0) {
        // Create term cluster nodes if needed
        const termCategories = {
          backend: { id: 'tech-backend', name: 'Backend Terms', created: false },
          frontend: { id: 'tech-frontend', name: 'Frontend Terms', created: false },
          infrastructure: { id: 'tech-infra', name: 'Infrastructure Terms', created: false },
          process: { id: 'tech-process', name: 'Process Terms', created: false },
          product: { id: 'tech-product', name: 'Product Terms', created: false }
        };
        
        // Group terms by category
        const termsByCategory = {
          backend: [],
          frontend: [],
          infrastructure: [],
          process: [],
          product: []
        };
        
        mentionedTerms.forEach((details, term) => {
          termsByCategory[details.category].push({
            term,
            details: details
          });
        });
        
        // Add nodes for each category with terms
        Object.keys(termsByCategory).forEach(category => {
          if (termsByCategory[category].length > 0) {
            // Create category node if it has terms
            if (!termCategories[category].created) {
              mindMap.nodes.push({
                id: termCategories[category].id,
                name: termCategories[category].name,
                type: 'terms',
                details: `Technical terminology related to ${category} discussed in the meeting`
              });
              termCategories[category].created = true;
              
              // Link to main category
              const mainCategoryMap = {
                backend: 'backend',
                frontend: 'frontend',
                infrastructure: 'infra',
                process: 'process',
                product: 'product'
              };
              mindMap.links.push({ 
                source: mainCategoryMap[category], 
                target: termCategories[category].id 
              });
            }
            
            // Add term nodes (limit to 5 per category to avoid crowding)
            termsByCategory[category].slice(0, 5).forEach(termObj => {
              const id = `term-${idCounter++}`;
              const details = termObj.details;
              
              // Format details with definition and context mentions
              let formattedDetails = `**${termObj.term}**: ${details.definition}\n\n**Context:**\n`;
              details.mentions.forEach(mention => {
                formattedDetails += `- "${mention.sentence}" (${mention.timestamp})\n`;
              });
              
              mindMap.nodes.push({
                id,
                name: termObj.term,
                type: 'terms',
                details: formattedDetails
              });
              
              mindMap.links.push({ 
                source: termCategories[category].id, 
                target: id 
              });
            });
          }
        });
      }
      
      // Action items - connected to Process node
      if (actionItems.size > 0) {
        const actionNodeId = 'action-items';
        mindMap.nodes.push({
          id: actionNodeId,
          name: 'Action Items',
          type: 'tasks',
          details: 'Tasks to be completed'
        });
        mindMap.links.push({ source: 'process', target: actionNodeId });
        
        addSubNodes(actionNodeId, actionItems, 'tasks', 'action');
      }
      
      // Blockers - connected to Process node
      if (blockers.size > 0) {
        const blockerNodeId = 'blockers';
        mindMap.nodes.push({
          id: blockerNodeId,
          name: 'Blockers',
          type: 'redFlags',
          details: 'Issues blocking progress'
        });
        mindMap.links.push({ source: 'process', target: blockerNodeId });
        
        addSubNodes(blockerNodeId, blockers, 'redFlags', 'blocker');
      }
      
      // Decisions - connected to Process node
      if (decisions.size > 0) {
        const decisionNodeId = 'decisions';
        mindMap.nodes.push({
          id: decisionNodeId,
          name: 'Decisions',
          type: 'summary',
          details: 'Decisions made during the meeting'
        });
        mindMap.links.push({ source: 'process', target: decisionNodeId });
        
        addSubNodes(decisionNodeId, decisions, 'summary', 'decision');
      }
      
      // Open Questions - connected to Process node
      if (openQuestions.size > 0) {
        const questionNodeId = 'questions';
        mindMap.nodes.push({
          id: questionNodeId,
          name: 'Open Questions',
          type: 'redFlags',
          details: 'Unresolved questions'
        });
        mindMap.links.push({ source: 'process', target: questionNodeId });
        
        addSubNodes(questionNodeId, openQuestions, 'redFlags', 'question');
      }
      
      // Topic discussions
      Object.keys(discussedTopics).forEach(category => {
        if (discussedTopics[category].size > 0) {
          const topicNodeId = `${category}-topics`;
          
          // Add a topics container node
          mindMap.nodes.push({
            id: topicNodeId,
            name: `${category.charAt(0).toUpperCase() + category.slice(1)} Topics`,
            type: 'summary',
            details: `Topics related to ${category} discussed in the meeting`
          });
          
          // Connect to the correct main node
          const categoryMap = {
            backend: 'backend',
            frontend: 'frontend',
            infrastructure: 'infra',
            process: 'process',
            product: 'product'
          };
          
          mindMap.links.push({ source: categoryMap[category], target: topicNodeId });
          
          // Add topic nodes
          addSubNodes(topicNodeId, discussedTopics[category], 'summary', `${category}-topic`);
        }
      });
      
      // Engineers, Managers, Stakeholders - only if any were detected
      if (roleInfo.engineers.size > 0 || roleInfo.managers.size > 0 || roleInfo.stakeholders.size > 0) {
        const peopleNodeId = 'people';
        mindMap.nodes.push({
          id: peopleNodeId,
          name: 'Key People',
          type: 'users',
          details: 'People mentioned in the meeting with their roles'
        });
        
        // Connect to the main node that makes most sense
        mindMap.links.push({ source: 'process', target: peopleNodeId });
        
        // Add role-specific nodes
        if (roleInfo.engineers.size > 0) {
          const engineersNodeId = 'engineers';
          mindMap.nodes.push({
            id: engineersNodeId,
            name: 'Engineers',
            type: 'users',
            details: 'Engineering team members mentioned'
          });
          mindMap.links.push({ source: peopleNodeId, target: engineersNodeId });
          
          addSubNodes(engineersNodeId, roleInfo.engineers, 'users', 'engineer');
        }
        
        if (roleInfo.managers.size > 0) {
          const managersNodeId = 'managers';
          mindMap.nodes.push({
            id: managersNodeId,
            name: 'Managers',
            type: 'users',
            details: 'Managers and leads mentioned'
          });
          mindMap.links.push({ source: peopleNodeId, target: managersNodeId });
          
          addSubNodes(managersNodeId, roleInfo.managers, 'users', 'manager');
        }
        
        if (roleInfo.stakeholders.size > 0) {
          const stakeholdersNodeId = 'stakeholders';
          mindMap.nodes.push({
            id: stakeholdersNodeId,
            name: 'Stakeholders',
            type: 'users',
            details: 'Stakeholders and clients mentioned'
          });
          mindMap.links.push({ source: peopleNodeId, target: stakeholdersNodeId });
          
          addSubNodes(stakeholdersNodeId, roleInfo.stakeholders, 'users', 'stakeholder');
        }
      }

      // Add PM insights from the latest frame
      const latestFrame = contextData.context[contextData.context.length - 1];
      if (latestFrame && latestFrame.pm_insights) {
        const pmInsights = latestFrame.pm_insights;
        
        // Sprint Goals & Metrics
        if (pmInsights.sprint_goals?.length > 0 || pmInsights.key_metrics?.length > 0) {
          const goalsNodeId = 'sprint-goals-metrics';
          mindMap.nodes.push({
            id: goalsNodeId,
            name: 'Sprint Goals & Metrics',
            type: 'summary',
            details: 'Current sprint objectives and key metrics'
          });
          mindMap.links.push({ source: 'process', target: goalsNodeId });
          
          // Add sprint goals
          if (pmInsights.sprint_goals?.length > 0) {
            const sprintGoalsId = 'sprint-goals';
            mindMap.nodes.push({
              id: sprintGoalsId,
              name: 'Sprint Goals',
              type: 'tasks',
              details: 'Objectives for the current sprint'
            });
            mindMap.links.push({ source: goalsNodeId, target: sprintGoalsId });
            
            pmInsights.sprint_goals.forEach((goal, idx) => {
              const id = `goal-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Goal ${idx + 1}`,
                type: 'tasks',
                details: goal
              });
              mindMap.links.push({ source: sprintGoalsId, target: id });
            });
          }
          
          // Add metrics
          if (pmInsights.key_metrics?.length > 0) {
            const metricsId = 'key-metrics';
            mindMap.nodes.push({
              id: metricsId,
              name: 'Key Metrics',
              type: 'summary',
              details: 'Important metrics and KPIs'
            });
            mindMap.links.push({ source: goalsNodeId, target: metricsId });
            
            pmInsights.key_metrics.forEach((metric, idx) => {
              const id = `metric-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Metric ${idx + 1}`,
                type: 'summary',
                details: metric
              });
              mindMap.links.push({ source: metricsId, target: id });
            });
          }
        }
        
        // Feature Status
        if (pmInsights.feature_status?.length > 0) {
          const featureStatusId = 'feature-status';
          mindMap.nodes.push({
            id: featureStatusId,
            name: 'Feature Status',
            type: 'summary',
            details: 'Current status of features in development'
          });
          mindMap.links.push({ source: 'process', target: featureStatusId });
          
          pmInsights.feature_status.forEach((feature, idx) => {
            const id = `feature-${idCounter++}`;
            mindMap.nodes.push({
              id,
              name: `Feature ${idx + 1}`,
              type: 'summary',
              details: feature
            });
            mindMap.links.push({ source: featureStatusId, target: id });
          });
        }
        
        // Dependencies & Technical Constraints
        if (pmInsights.dependencies?.length > 0 || pmInsights.technical_constraints?.length > 0) {
          const constraintsNodeId = 'dependencies-constraints';
          mindMap.nodes.push({
            id: constraintsNodeId,
            name: 'Dependencies & Constraints',
            type: 'redFlags',
            details: 'Technical dependencies and limitations'
          });
          mindMap.links.push({ source: 'process', target: constraintsNodeId });
          
          // Add dependencies
          if (pmInsights.dependencies?.length > 0) {
            const dependenciesId = 'dependencies';
            mindMap.nodes.push({
              id: dependenciesId,
              name: 'Dependencies',
              type: 'redFlags',
              details: 'Cross-team and system dependencies'
            });
            mindMap.links.push({ source: constraintsNodeId, target: dependenciesId });
            
            pmInsights.dependencies.forEach((dep, idx) => {
              const id = `dep-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Dependency ${idx + 1}`,
                type: 'redFlags',
                details: dep
              });
              mindMap.links.push({ source: dependenciesId, target: id });
            });
          }
          
          // Add technical constraints
          if (pmInsights.technical_constraints?.length > 0) {
            const constraintsId = 'tech-constraints';
            mindMap.nodes.push({
              id: constraintsId,
              name: 'Technical Constraints',
              type: 'redFlags',
              details: 'Technical limitations and constraints'
            });
            mindMap.links.push({ source: constraintsNodeId, target: constraintsId });
            
            pmInsights.technical_constraints.forEach((constraint, idx) => {
              const id = `constraint-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Constraint ${idx + 1}`,
                type: 'redFlags',
                details: constraint
              });
              mindMap.links.push({ source: constraintsId, target: id });
            });
          }
        }
        
        // Risks & Resource Needs
        if (pmInsights.risks?.length > 0 || pmInsights.resource_needs?.length > 0) {
          const risksNodeId = 'risks-resources';
          mindMap.nodes.push({
            id: risksNodeId,
            name: 'Risks & Resources',
            type: 'redFlags',
            details: 'Project risks and resource requirements'
          });
          mindMap.links.push({ source: 'process', target: risksNodeId });
          
          // Add risks
          if (pmInsights.risks?.length > 0) {
            const risksId = 'risks';
            mindMap.nodes.push({
              id: risksId,
              name: 'Risks',
              type: 'redFlags',
              details: 'Potential risks and concerns'
            });
            mindMap.links.push({ source: risksNodeId, target: risksId });
            
            pmInsights.risks.forEach((risk, idx) => {
              const id = `risk-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Risk ${idx + 1}`,
                type: 'redFlags',
                details: risk
              });
              mindMap.links.push({ source: risksId, target: id });
            });
          }
          
          // Add resource needs
          if (pmInsights.resource_needs?.length > 0) {
            const resourcesId = 'resources';
            mindMap.nodes.push({
              id: resourcesId,
              name: 'Resource Needs',
              type: 'tasks',
              details: 'Required resources and constraints'
            });
            mindMap.links.push({ source: risksNodeId, target: resourcesId });
            
            pmInsights.resource_needs.forEach((need, idx) => {
              const id = `resource-${idCounter++}`;
              mindMap.nodes.push({
                id,
                name: `Need ${idx + 1}`,
                type: 'tasks',
                details: need
              });
              mindMap.links.push({ source: resourcesId, target: id });
            });
          }
        }
        
        // Stakeholder Requests
        if (pmInsights.stakeholder_requests?.length > 0) {
          const stakeholderId = 'stakeholder-requests';
          mindMap.nodes.push({
            id: stakeholderId,
            name: 'Stakeholder Requests',
            type: 'users',
            details: 'Requirements and requests from stakeholders'
          });
          mindMap.links.push({ source: 'process', target: stakeholderId });
          
          pmInsights.stakeholder_requests.forEach((request, idx) => {
            const id = `request-${idCounter++}`;
            mindMap.nodes.push({
              id,
              name: `Request ${idx + 1}`,
              type: 'users',
              details: request
            });
            mindMap.links.push({ source: stakeholderId, target: id });
          });
        }
      }

      return mindMap;
    } catch (error) {
      console.error('Error creating enhanced mind map:', error);
      return createDefaultMindMap();
    }
  };

  // Enhance the executive summary with more comprehensive analysis
  const createComprehensiveWarnings = (contextData) => {
    if (!contextData || !contextData.context || contextData.context.length === 0) {
      return createDefaultWarnings();
    }

    try {
      const warnings = { redFlags: [], warnings: [] };
      
      // Helper function to extract relevant text around a match
      const extractRelevantText = (text, regex) => {
        const match = regex.exec(text);
        if (!match) return text;
        
        const matchIndex = match.index;
        const contextStart = Math.max(0, matchIndex - 30);
        const contextEnd = Math.min(text.length, matchIndex + match[0].length + 30);
        
        return text.substring(contextStart, contextEnd);
      };

      // Format timestamp consistently across the application
      const formatTimestamp = (seconds) => {
        if (seconds === undefined || seconds === null) return "unknown time";
        
        const minutes = Math.floor(seconds / 60);
        const remainingSecs = Math.floor(seconds % 60);
        return `${minutes}:${remainingSecs.toString().padStart(2, '0')}`;
      };
      
      // Track open action items and questions
      const actionItems = new Map(); // key: action description, value: {assignee, dueDate, timestamp}
      const openQuestions = new Map(); // key: question, value: {asker, timestamp, lastResponse}
      
      // Process frames to identify action items and questions
      contextData.context.forEach((frame, index) => {
        if (!frame.description) return;
        
        const description = frame.description.toLowerCase();
        const timestamp = frame.metadata?.timestamp || 0;
        const nextFrame = contextData.context[index + 1];
        
        // Action Item Detection
        const actionMatch = description.match(/(?:can you|could you|please)?\s*([^,.!?]+)\s*by\s*([^,.!?]+)/i);
        if (actionMatch) {
          const action = actionMatch[1].trim();
          const dueDate = actionMatch[2].trim();
          const assigneeMatch = description.match(/@(\w+)/);
          const assignee = assigneeMatch ? assigneeMatch[1] : null;
          
          actionItems.set(action, {
            assignee,
            dueDate,
            timestamp,
            description: frame.description
          });
        }
        
        // Question Detection
        const questionMatch = description.match(/(?:can|could|would|should|is|are|will|how|what|when|where|why|who)\s+([^,.!?]+)\??/i);
        if (questionMatch) {
          const question = questionMatch[0].trim();
          const askerMatch = frame.description.match(/@(\w+)/);
          const asker = askerMatch ? askerMatch[1] : "Someone";
          
          // Check if question was answered in next frame
          const wasAnswered = nextFrame && 
            (nextFrame.description.toLowerCase().includes('yes') ||
             nextFrame.description.toLowerCase().includes('no') ||
             nextFrame.description.length > 50); // Assume long responses are answers
          
          if (!wasAnswered) {
            openQuestions.set(question, {
              asker,
              timestamp,
              description: frame.description
            });
          }
        }
      });

      // Add open action items to warnings
      actionItems.forEach((details, action) => {
        warnings.warnings.push({
          title: "Open Action Item",
          time: formatTimestamp(details.timestamp),
          details: `${details.assignee ? '@' + details.assignee : 'Someone'} needs to ${action} by ${details.dueDate}.\n\nContext: "${details.description}"`
        });
      });

      // Add unanswered questions to warnings
      openQuestions.forEach((details, question) => {
        warnings.warnings.push({
          title: "Unanswered Question",
          time: formatTimestamp(details.timestamp),
          details: `${details.asker} asked: "${question}"\n\nContext: "${details.description}"`
        });
      });

      // Add existing warning patterns
      const warningPatterns = [
        { 
          regex: /unclear requirements|requirements changed|scope change|change request|specification issue/i, 
          title: "Requirements Clarity Issue",
          details: "Requirements need clarification"
        },
        { 
          regex: /technical debt|refactor needed|needs cleanup|architectural issue|code quality|maintenance/i, 
          title: "Technical Debt Concern",
          details: "Code quality issues identified"
        },
        { 
          regex: /test coverage|missing tests|quality assurance|QA concern|manual testing|automated testing/i, 
          title: "Testing Coverage Issue",
          details: "Insufficient test coverage"
        },
        { 
          regex: /dependency|waiting on|blocked by|external team|third party|vendor|integration/i, 
          title: "External Dependency",
          details: "Progress blocked by external factors"
        },
        { 
          regex: /communication issue|misunderstanding|unclear|confusion|not aligned|alignment/i, 
          title: "Communication Issue",
          details: "Team alignment problems identified"
        },
        {
          regex: /documentation|docs|missing information|need to document|knowledge transfer/i,
          title: "Documentation Needed",
          details: "Documentation is insufficient"
        }
      ];

      // Process each frame with warning patterns
      contextData.context.forEach((frame, index) => {
        if (!frame.description) return;
        
        const description = frame.description;
        const timestamp = frame.metadata?.timestamp 
          ? formatTimestamp(frame.metadata.timestamp)
          : new Date().toLocaleTimeString();
          
        // Check for warnings
        warningPatterns.forEach(pattern => {
          if (pattern.regex.test(description)) {
            // Check if we already have this warning (avoid duplicates)
            const existingWarning = warnings.warnings.find(w => w.title === pattern.title);
            if (!existingWarning) {
              warnings.warnings.push({
                title: pattern.title,
                time: timestamp,
                details: `${pattern.details}: "${extractRelevantText(description, pattern.regex)}"`
              });
            }
          }
        });
      });

      // Add summary stats
      const totalFrames = contextData.context.length;
      const startTime = contextData.context[0]?.metadata?.timestamp || 0;
      const endTime = contextData.context[totalFrames-1]?.metadata?.timestamp || 0;
      const meetingDuration = endTime - startTime;
      
      if (warnings.warnings.length === 0) {
        warnings.warnings.push({
          title: "No issues detected",
          time: new Date().toLocaleTimeString(),
          details: `Analysis complete. No action items or concerns identified in the meeting (${Math.floor(meetingDuration/60)}m ${Math.floor(meetingDuration%60)}s).`
        });
      }

      return warnings;
    } catch (error) {
      console.error('Error creating comprehensive warnings:', error);
      return createDefaultWarnings();
    }
  };

  // Function to manually refresh the mind map and warning data
  const handleManualRefresh = async () => {
    if (!context?.context?.length) return;
    
    // Set loading state
    setLoading(true);
    
    try {
      // Refresh frames and context
      await fetchFrames();
      await fetchContext();
      
      // Create new data
      const newMindMapData = createEnhancedMindMap(context);
      const newWarningsData = createComprehensiveWarnings(context);
      const newHash = hashContext(context);
      
      // Update state
      setMindMapData(newMindMapData);
      setLastContextHash(newHash);
    } catch (error) {
      console.error('Error refreshing data:', error);
    } finally {
      // Clear loading state
      setLoading(false);
    }
  };

  // Hash the context to detect meaningful changes
  const hashContext = (context) => {
    if (!context || !context.context) return '';
    return context.context.map(frame => frame.frame_index).join(',');
  };

  // Update data only when meaningful changes occur, with throttling
  useEffect(() => {
    if (!context?.context?.length) {
      setMindMapData(createDefaultMindMap());
      return;
    }

    // Only auto-update if manual refresh is not enabled
    if (manualRefreshEnabled) return;
    
    const currentHash = hashContext(context);
    
    // Skip update if context hasn't meaningfully changed
    if (currentHash === lastContextHash) return;
    
    // Clear any pending timeout
    if (updateTimeout) {
      clearTimeout(updateTimeout);
    }
    
    // Set a new timeout for updates (throttle to once per second)
    const timeoutId = setTimeout(() => {
      setMindMapData(createEnhancedMindMap(context));
      setLastContextHash(currentHash);
    }, 1000);
    
    setUpdateTimeout(timeoutId);
    
    return () => {
      if (updateTimeout) {
        clearTimeout(updateTimeout);
      }
    };
  }, [context, lastContextHash, manualRefreshEnabled]);

  // Initial data load
  useEffect(() => {
    if (context?.context?.length > 0 && !mindMapData) {
      handleManualRefresh();
    }
  }, [context]);

  // Add this to your existing useEffect for polling
  useEffect(() => {
    const checkTextProcessing = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/text_processing_status`);
        setIsTextProcessing(response.data.is_processing);
      } catch (error) {
        console.error('Error checking text processing status:', error);
      }
    };

    const interval = setInterval(() => {
      checkTextProcessing();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

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
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Grid container spacing={3}>
        {/* Existing UI Components */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
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
              <>
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
                
                <ChatInterface 
                  context={context}
                  currentFrame={currentFrame}
                />

                {mindMapData && (
                  <Box sx={{ mt: 3 }}>
                    <MindMap 
                      data={mindMapData} 
                      onRefresh={handleManualRefresh}
                    />
                  </Box>
                )}
              </>
            )}
          </Paper>
        </Grid>
      </Grid>
      <TextStreamInput onDataCleared={() => setIsTextProcessing(false)} />
      <LiveTextSummary isProcessing={isTextProcessing} />
      {/* Recording status indicator */}
      {isRecording && (
        <Box sx={{ 
          position: 'fixed', 
          top: 16, 
          right: 16, 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '4px 12px',
          borderRadius: '16px',
          color: 'white',
          zIndex: 1000
        }}>
          <Box sx={{ 
            width: 8, 
            height: 8, 
            borderRadius: '50%', 
            backgroundColor: 'error.main',
            animation: 'pulse 1.5s ease-in-out infinite'
          }} />
          Live
        </Box>
      )}
    </Container>
  );
}

export default App; 