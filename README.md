# Screen Understanding and VQA System

A modular system for performing visual question answering and understanding on live video feeds. This system is designed to process video frames in real-time, understand context, and answer questions about the content being displayed.

## Features

- Real-time video frame processing
- API-based model integration (no local GPU required)
- Support for multiple VQA models through API providers
- Flexible data source adapters for different video stream inputs
- Advanced temporal context management with:
  - Configurable context window size
  - Token-based context pruning
  - Automatic context persistence
  - Rate limiting support
- Background frame processing
- Redis-based frame buffering
- Split architecture for frame upload and Q&A

## Architecture

The system is built with modularity in mind and consists of the following key components:

1. **Frame Processors**: Handle incoming video frames from various sources
2. **API Clients**: Interface with different VQA model providers
3. **Context Manager**: Maintains temporal context with:
   - Configurable retention periods
   - Token-based pruning
   - Automatic saving/loading of context
   - Rate limit management
4. **VQA Engine**: Processes questions and generates answers based on visual context
5. **Data Source Adapters**: Abstract different video stream sources
6. **Redis Integration**: Enables distributed frame processing and Q&A

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Start Redis server (required for frame buffering)

## Supported Models

Currently supported models through API integration:

- **Gemini Pro Vision** (via Google AI API): Fast and efficient vision-language model with rate limiting support
- **LLaVA** (via Replicate API): A powerful vision-language model capable of understanding and answering questions about images
- **HuggingFace**: Support for various hosted models on HuggingFace

## Usage

### Basic Usage

```python
from screen_understanding import ScreenUnderstanding
from screen_understanding.api import APIClientFactory
from screen_understanding.sources import FrameBufferSource

# Initialize with Gemini model and rate limiting
system = ScreenUnderstanding(
    model=APIClientFactory.create(
        "gemini",
        model_name="models/gemini-2.0-flash-lite",
        rate_limit_rpm=30
    ),
    source=FrameBufferSource(),
    max_context_frames=10,
    max_context_tokens=4000
)

# Start processing
await system.start()

# Ask questions with temporal context
answer = await system.ask("What changed in the UI since the last frame?")
print(answer)
```

### Split Processing Mode

Run frame upload and Q&A in separate processes:

1. Upload frames:
```bash
python examples/upload_frames.py --video your_video.mp4 --redis-prefix demo:
```

2. Run Q&A interface:
```bash
python examples/realtime_demo.py --redis-prefix demo: --rate-limit-rpm 30
```

## Environment Variables

Required environment variables:

- `GOOGLE_API_KEY`: API key for Google AI (if using Gemini)
- `REPLICATE_API_TOKEN`: API token for Replicate (if using LLaVA)
- `GEMINI_RATE_LIMIT_RPM`: Rate limit for Gemini API (default: 30)
- `DEFAULT_MODEL_TYPE`: Default model to use (e.g., "gemini", "llava")
- Other configuration variables as specified in `.env.example`

## Data Storage

- Frame context is automatically saved to `data/frame_context.json`
- Redis is used for frame buffering with configurable key prefixes
- Context is automatically pruned based on:
  - Maximum number of frames (configurable)
  - Maximum token count (configurable)
  - Retention period (time-based cleanup)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
