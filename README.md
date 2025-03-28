# Perspectiv Screen Understanding

A real-time screen understanding system that processes video frames using Google's Gemini Vision API to provide contextual understanding and analysis of screen content.

## Features

- Real-time frame processing with Gemini Vision API
- Intelligent context management with token-based pruning
- Live debug UI with frame timeline and buffer statistics
- Redis-based frame buffer for efficient frame management
- Rate limiting support for API calls
- Atomic context file operations for data safety
- Interactive chat interface with Gemini for contextual queries
- Speech-to-text and text-to-speech capabilities
- Markdown formatting for AI responses

## Project Structure

```
.
├── screen_understanding/     # Core processing library
│   ├── __init__.py
│   ├── core.py             # Main processing logic
│   ├── context.py          # Context management
│   ├── api/                # API clients
│   │   ├── base.py
│   │   └── clients/
│   │       └── gemini.py   # Gemini Vision API client
│   └── sources/            # Frame sources
│       └── frame_buffer.py # Redis-based frame buffer
├── debug-ui/               # Debug interface
│   ├── server.py          # FastAPI server with chat endpoint
│   └── src/               # React frontend
│       └── App.js         # Main UI component with chat interface
└── examples/              # Example scripts
    ├── upload_frames.py   # Frame upload utility
    ├── cleanup.py        # Cleanup utility
    └── test_context_processing.py  # Test script
```

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
cd debug-ui
npm install
```

3. Create a `.env` file with your Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start Redis server:
```bash
redis-server
```

2. Start the debug UI server:
```bash
cd debug-ui
python -m uvicorn server:app --reload --port 8000
```

3. Start the React development server:
```bash
cd debug-ui
npm start
```

4. Upload frames:
```bash
python examples/upload_frames.py --video path/to/video.mp4 --frame-interval 1.0 --redis-prefix test:
```

5. Process frames:
```bash
python examples/test_context_processing.py --redis-prefix test: --context-file data/frame_context.json --poll-interval 5.0 --rate-limit-rpm 30
```

## Configuration

- `max_context_frames`: Maximum number of frames to keep in context (default: 5)
- `max_context_tokens`: Maximum number of tokens in context (default: 4000)
- `rate_limit_rpm`: Rate limit for Gemini API calls (default: 30)
- `poll_interval`: Interval between frame processing checks (default: 5.0s)

## Debug UI Features

- Real-time frame timeline with thumbnails
- Buffer health monitoring
- Frame processing statistics
- Context token usage tracking
- Frame interval measurements
- Interactive chat interface with:
  - Natural language queries about screen content
  - Voice input support (Chrome/Edge/Safari)
  - Text-to-speech for AI responses
  - Markdown-formatted responses
  - Real-time context awareness

## Browser Compatibility

The debug UI's speech features require:
- Chrome 33+ (recommended)
- Edge 79+
- Safari 14.1+
- A working microphone for voice input
- System audio for text-to-speech

## Development

### Adding New API Clients

1. Create a new client in `screen_understanding/api/clients/`
2. Implement the `BaseAPIClient` interface
3. Register the client in `APIClientFactory`

### Adding Frame Sources

1. Create a new source in `screen_understanding/sources/`
2. Implement the `DataSource` interface
3. Use in `ScreenUnderstanding` initialization

## Error Handling

- Automatic fallback to local storage if context directory is unavailable
- Graceful handling of API rate limits
- Atomic context file operations to prevent corruption
- Buffer overflow protection with intelligent pruning
- Speech recognition error handling with user feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
