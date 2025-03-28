# Screen Understanding Debug UI

A debugging interface for visualizing video frames, metadata, and context in the Screen Understanding system, with an interactive chat interface for querying screen content.

## Features

- Real-time visualization of video frames
- Display of frame metadata and timestamps
- Current context window visualization
- Auto-updating UI with frame buffer status
- Virtualized scrolling for efficient frame rendering
- Interactive chat interface with:
  - Natural language queries about screen content
  - Voice input (speech-to-text)
  - Text-to-speech for AI responses
  - Markdown-formatted responses
  - Real-time context awareness

## Prerequisites

- Node.js (v14 or later)
- Python 3.7+
- Redis server running locally
- Screen Understanding system set up and running
- Google Gemini API key
- Modern web browser with speech support (Chrome/Edge/Safari)

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Create a `.env` file in the debug-ui directory:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Debug UI

1. Start the FastAPI backend:
```bash
python -m uvicorn server:app --reload --port 8000
```

2. Start the React development server:
```bash
npm start
```

3. Open your browser and navigate to http://localhost:3000

## Usage

1. Make sure your Screen Understanding system is running and processing frames
2. The UI will automatically display:
   - Current frames in the Redis buffer
   - Metadata for each frame
   - Current context window
   - Frame timestamps and indices
   - Chat interface for querying content
3. The UI auto-updates every 500ms to show new frames
4. Use the scroll interface to navigate through frames

### Using the Chat Interface

1. Type your question or click the microphone icon to use voice input
2. The AI will respond with markdown-formatted text
3. Click the speaker icon on any AI response to hear it read aloud
4. The AI has access to:
   - Current frame context
   - Previous frames' history
   - Chat conversation history

### Voice Features

- **Speech-to-Text (Input)**:
  - Click the microphone icon to start recording
  - Speak your question clearly
  - Click again to stop recording
  - Edit the transcribed text if needed

- **Text-to-Speech (Output)**:
  - Click the speaker icon on any AI response
  - Click again to stop playback
  - Works with all markdown-formatted responses

## Browser Support

The chat interface's speech features require:
- Chrome 33+ (recommended)
- Edge 79+
- Safari 14.1+
- A working microphone for voice input
- System audio for text-to-speech

## Troubleshooting

1. If frames are not appearing:
   - Check if Redis server is running
   - Verify frames are being uploaded correctly
   - Check the Redis prefix matches your configuration

2. If context is not showing:
   - Ensure the context file exists at `data/frame_context.json`
   - Verify the Screen Understanding system is writing context properly

3. If chat features aren't working:
   - Verify your Gemini API key is set correctly in `.env`
   - Check browser console for errors
   - Ensure microphone permissions are granted for voice input
   - Try using a supported browser for speech features

4. For other issues:
   - Check the browser console for errors
   - Check the FastAPI server logs
   - Verify all dependencies are installed correctly 