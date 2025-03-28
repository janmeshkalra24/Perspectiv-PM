from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
import redis
import json
from typing import Dict, List, Optional
import ast
import time
import logging
import os
from pathlib import Path
import subprocess
import sys
import signal
import threading
import queue

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get workspace root directory (one level up from debug-ui)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG_UI_DIR = os.path.dirname(os.path.abspath(__file__))

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context file path - look in workspace root first, then data directory
CONTEXT_FILE_PATHS = [
    os.path.join(WORKSPACE_ROOT, "data/frame_context.json"),
    os.path.join(DEBUG_UI_DIR, "data/frame_context.json"),
    "data/frame_context.json",
    "../data/frame_context.json",
    "frame_context.json"
]

def find_context_file() -> Optional[Path]:
    """Find the context file in possible locations."""
    for path in CONTEXT_FILE_PATHS:
        file_path = Path(path)
        if file_path.exists():
            logger.info(f"Found context file at: {file_path}")
            return file_path
    return None

@app.get("/frames")
async def get_frames(prefix: str = "demo:", start: int = 0, limit: int = 30) -> Dict:
    """Get frames and their metadata within a range."""
    try:
        # Get all frame keys
        all_keys = sorted([
            k.decode() for k in redis_client.keys(f"{prefix}*")
            if not k.decode().endswith(":meta")
        ])
        
        total_frames = len(all_keys)
        
        # Get requested range
        frame_keys = all_keys[start:start + limit]
        frames_data = []
        
        for key in frame_keys:
            # Get frame data and metadata
            frame_data = redis_client.get(key)
            meta_data = redis_client.get(f"{key}:meta")
            
            if frame_data and meta_data:
                try:
                    metadata = ast.literal_eval(meta_data.decode())
                except:
                    metadata = {"error": "Could not parse metadata"}
                    
                frames_data.append({
                    "key": key,
                    "frame_index": metadata.get("frame_number", -1),
                    "timestamp": metadata.get("timestamp", 0),
                    "metadata": metadata,
                    "frame_data": frame_data.decode('latin1')  # Send as base64 later
                })
        
        return {
            "total_frames": total_frames,
            "frames": frames_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/frame/{frame_key}")
async def get_frame(frame_key: str) -> Response:
    """Get a specific frame's data."""
    try:
        frame_data = redis_client.get(frame_key)
        if not frame_data:
            raise HTTPException(status_code=404, detail="Frame not found")
            
        return Response(content=frame_data, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_buffer_stats(context_data):
    """Calculate buffer statistics including health."""
    context = context_data.get("context", [])
    total_frames = len(context)
    
    if total_frames == 0:
        return {
            "total_frames_processed": 0,
            "frames_in_buffer": 0,
            "last_clear_time": time.time(),
            "token_usage": 0,
            "frame_interval": None,
            "buffer_health": 0.0,
            "next_clear_time": None
        }
    
    # Calculate frames in buffer (frames that have been uploaded but not yet processed)
    frames_in_buffer = sum(1 for entry in context if not entry.get("processing_metadata"))
    frames_processed = total_frames - frames_in_buffer
    
    # Calculate token usage
    token_usage = sum(len(str(c.get("description", ""))) // 4 for c in context)
    
    # Calculate frame interval if we have at least 2 frames
    frame_interval = None
    if len(context) > 1:
        timestamps = [entry["metadata"]["timestamp"] for entry in context if "metadata" in entry]
        if len(timestamps) > 1:
            frame_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    
    # Calculate buffer health (ratio of processed frames to total frames)
    buffer_health = frames_processed / total_frames if total_frames > 0 else 0.0
    
    return {
        "total_frames_processed": frames_processed,
        "frames_in_buffer": frames_in_buffer,
        "last_clear_time": time.time(),
        "token_usage": token_usage,
        "frame_interval": frame_interval,
        "buffer_health": buffer_health,
        "next_clear_time": None
    }

@app.get("/context")
async def get_context() -> Dict:
    """Get the current context from the context file."""
    try:
        context_path = find_context_file()
        if not context_path:
            logger.warning("Context file not found in any of the expected locations")
            return {
                "context": [],
                "timestamps": [],
                "buffer_stats": calculate_buffer_stats({"context": []})
            }

        # Check if file exists but is empty
        if context_path.stat().st_size == 0:
            logger.warning("Context file is empty")
            return {
                "context": [],
                "timestamps": [],
                "buffer_stats": calculate_buffer_stats({"context": []})
            }

        with open(context_path, 'r') as f:
            try:
                context = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in context file: {e}")
                return {
                    "context": [],
                    "timestamps": [],
                    "buffer_stats": calculate_buffer_stats({"context": []})
                }
            
        # Add additional metadata for UI
        if "context" in context:
            for entry in context["context"]:
                if "metadata" in entry:
                    # Add frame interval if not present
                    if "frame_interval" not in entry["metadata"]:
                        entry["metadata"]["frame_interval"] = entry["metadata"].get("timestamp", 0) - \
                            context["context"][context["context"].index(entry)-1]["metadata"].get("timestamp", 0) \
                            if context["context"].index(entry) > 0 else 0
                            
        # Calculate buffer stats
        context["buffer_stats"] = calculate_buffer_stats(context)
            
        logger.debug(f"Returning context with {len(context.get('context', []))} entries")
        return context
        
    except FileNotFoundError as e:
        logger.warning(f"Context file not found: {e}")
        return {
            "context": [],
            "timestamps": [],
            "buffer_stats": calculate_buffer_stats({"context": []})
        }
    except Exception as e:
        logger.error(f"Error reading context: {e}", exc_info=True)
        return {"error": str(e), "context": []}

# Global process tracking
running_processes = {}

# Process output queues
process_logs = {
    "upload": queue.Queue(),
    "process": queue.Queue()
}

def log_stream(stream, process_type):
    """Read from a process stream and log it."""
    try:
        for line in iter(stream.readline, ''):  # Changed from b'' to '' for text mode
            try:
                # No need to decode since we're using universal_newlines=True
                line_str = line.strip()
                logger.info(f"{process_type}: {line_str}")
                process_logs[process_type].put(line_str)
            except Exception as e:
                logger.error(f"Error processing {process_type} line: {e}")
    except Exception as e:
        logger.error(f"Error in log stream for {process_type}: {e}")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    frame_interval: float = Form(...),
    delay_interval: float = Form(...)
):
    """Upload a video file and start processing with given parameters."""
    try:
        # Save the uploaded file
        uploads_dir = os.path.join(DEBUG_UI_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_processing")
async def start_processing(
    file_path: str = Form(...),
    frame_interval: float = Form(...),
    delay_interval: float = Form(...),
    redis_prefix: str = Form("test:")
):
    """Start the frame upload and processing pipeline."""
    try:
        logger.info(f"Starting processing with params: file_path={file_path}, frame_interval={frame_interval}, delay_interval={delay_interval}, redis_prefix={redis_prefix}")
        
        # Clear existing logs
        for q in process_logs.values():
            while not q.empty():
                q.get()
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(WORKSPACE_ROOT, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert file path to absolute path if relative
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(os.path.join(DEBUG_UI_DIR, file_path))
        
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Kill any existing processes
        for process_type in running_processes:
            try:
                logger.info(f"Stopping existing {process_type} process")
                os.killpg(os.getpgid(running_processes[process_type].pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Error stopping {process_type} process: {e}")
        
        # Get absolute paths for scripts and ensure they exist
        upload_script = os.path.join(WORKSPACE_ROOT, "examples", "upload_frames.py")
        process_script = os.path.join(WORKSPACE_ROOT, "examples", "test_context_processing.py")
        context_file = os.path.join(data_dir, "frame_context.json")
        
        # Validate scripts exist
        if not os.path.exists(upload_script):
            raise HTTPException(status_code=500, detail=f"Upload script not found at: {upload_script}")
        if not os.path.exists(process_script):
            raise HTTPException(status_code=500, detail=f"Processing script not found at: {process_script}")
        
        logger.info(f"Using paths: upload_script={upload_script}, process_script={process_script}, context_file={context_file}")
        
        # Start frame upload process
        upload_cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            upload_script,
            "--video", file_path,
            "--frame-interval", str(frame_interval),
            "--delay-interval", str(delay_interval),
            "--redis-prefix", redis_prefix
        ]
        
        logger.info(f"Starting upload process with command: {' '.join(upload_cmd)}")
        upload_process = subprocess.Popen(
            upload_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=WORKSPACE_ROOT  # Set working directory to workspace root
        )
        running_processes["upload"] = upload_process
        
        # Start output monitoring threads
        threading.Thread(target=log_stream, args=(upload_process.stdout, "upload"), daemon=True).start()
        threading.Thread(target=log_stream, args=(upload_process.stderr, "upload"), daemon=True).start()
        
        # Start processing process
        process_cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            process_script,
            "--redis-prefix", redis_prefix,
            "--context-file", context_file,
            "--poll-interval", "5.0",
            "--rate-limit-rpm", "30"
        ]
        
        logger.info(f"Starting processing with command: {' '.join(process_cmd)}")
        process = subprocess.Popen(
            process_cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=WORKSPACE_ROOT  # Set working directory to workspace root
        )
        running_processes["process"] = process
        
        # Start output monitoring threads
        threading.Thread(target=log_stream, args=(process.stdout, "process"), daemon=True).start()
        threading.Thread(target=log_stream, args=(process.stderr, "process"), daemon=True).start()
        
        return {"status": "success", "message": "Processing started"}
        
    except Exception as e:
        logger.error(f"Error in start_processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_processing")
async def stop_processing():
    """Stop all running processes."""
    try:
        for process_type in running_processes:
            try:
                os.killpg(os.getpgid(running_processes[process_type].pid), signal.SIGTERM)
            except:
                pass
        running_processes.clear()
        return {"status": "success", "message": "Processing stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/process_logs")
async def get_process_logs():
    """Get the latest logs from the processing scripts."""
    logs = {
        "upload": [],
        "process": []
    }
    
    for process_type in process_logs:
        while not process_logs[process_type].empty():
            try:
                logs[process_type].append(process_logs[process_type].get_nowait())
            except queue.Empty:
                break
    
    return logs

@app.post("/clear_data")
async def clear_data():
    """Clear all Redis data and reset processing state."""
    try:
        # Stop any running processes first
        for process_type in running_processes:
            try:
                os.killpg(os.getpgid(running_processes[process_type].pid), signal.SIGTERM)
            except:
                pass
        running_processes.clear()
        
        # Clear Redis data
        redis_client.flushall()
        
        # Clear process logs
        for q in process_logs.values():
            while not q.empty():
                q.get()
        
        # Clear context file
        context_path = find_context_file()
        if context_path:
            # Write empty context to file
            with open(context_path, 'w') as f:
                json.dump({"context": [], "timestamps": []}, f)
            logger.info(f"Cleared context file at: {context_path}")
        else:
            # Create new empty context file in data directory
            data_dir = os.path.join(WORKSPACE_ROOT, "data")
            os.makedirs(data_dir, exist_ok=True)
            context_path = os.path.join(data_dir, "frame_context.json")
            with open(context_path, 'w') as f:
                json.dump({"context": [], "timestamps": []}, f)
            logger.info(f"Created new empty context file at: {context_path}")
                
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        logger.error(f"Error clearing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 