# Screen Understanding Examples

This directory contains example scripts demonstrating the usage of the Screen Understanding system.

## Prerequisites

1. Make sure Redis is running locally (or update the configuration to point to your Redis instance)
2. Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

## Recording a Sample Video

Use the `record_screen.py` script to record your screen:

```bash
# Record for 30 seconds at 10 FPS
python record_screen.py --duration 30 --fps 10

# Record with custom output path
python record_screen.py --output my_recording.webm --duration 60
```

## Running the Real-time Demo

The `realtime_demo.py` script provides an interactive demo with real-time querying:

```bash
# Run with default settings
python realtime_demo.py --video path/to/your/recording.webm

# Run with custom frame interval (e.g., process every 0.5 seconds)
python realtime_demo.py --video recording.webm --frame-interval 0.5
```

### Demo Features

1. **Real-time Frame Processing**: Frames are processed as they come in
2. **Interactive Query Interface**: Ask questions about what you see
3. **Performance Metrics**: See processing time for each query
4. **Context Window**: System maintains context from recent frames

### Example Usage

1. Start recording a screen sharing session:
```bash
python record_screen.py --duration 60
```

2. Run the demo with the recorded video:
```bash
python realtime_demo.py --video screen_recording_*.webm
```

3. Ask questions about what you see, for example:
- "What's the current page being shown?"
- "What's the status of the JIRA ticket on screen?"
- "What was the last error message displayed?"

### Tips for Best Results

1. Set an appropriate frame interval (default is 1 second)
2. Keep screen content visible for a few seconds
3. Use clear, specific questions
4. Consider the context window when asking questions about past content 