# Screen Understanding Debug UI

A debugging interface for visualizing video frames, metadata, and context in the Screen Understanding system.

## Features

- Real-time visualization of video frames
- Display of frame metadata and timestamps
- Current context window visualization
- Auto-updating UI with frame buffer status
- Virtualized scrolling for efficient frame rendering

## Prerequisites

- Node.js (v14 or later)
- Python 3.7+
- Redis server running locally
- Screen Understanding system set up and running

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Debug UI

1. Start the FastAPI backend:
```bash
uvicorn server:app --reload --port 8000
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
3. The UI auto-updates every 5 seconds to show new frames
4. Use the scroll interface to navigate through frames

## Troubleshooting

1. If frames are not appearing:
   - Check if Redis server is running
   - Verify frames are being uploaded correctly
   - Check the Redis prefix matches your configuration

2. If context is not showing:
   - Ensure the context file exists at `data/frame_context.json`
   - Verify the Screen Understanding system is writing context properly

3. For other issues:
   - Check the browser console for errors
   - Check the FastAPI server logs
   - Verify all dependencies are installed correctly 