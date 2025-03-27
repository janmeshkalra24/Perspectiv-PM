from fastapi import FastAPI, HTTPException
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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

logger = logging.getLogger(__name__)

# Context file path - look in workspace root first, then data directory
CONTEXT_FILE_PATHS = [
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
                "buffer_stats": {
                    "total_frames_processed": 0,
                    "frames_in_buffer": 0,
                    "last_clear_time": time.time(),
                    "token_usage": 0,
                    "frame_interval": None,
                    "buffer_health": 1.0,
                    "next_clear_time": None
                }
            }

        # Check if file exists but is empty
        if context_path.stat().st_size == 0:
            logger.warning("Context file is empty")
            return {
                "context": [],
                "timestamps": [],
                "buffer_stats": {
                    "total_frames_processed": 0,
                    "frames_in_buffer": 0,
                    "last_clear_time": time.time(),
                    "token_usage": 0,
                    "frame_interval": None,
                    "buffer_health": 1.0,
                    "next_clear_time": None
                }
            }

        with open(context_path, 'r') as f:
            try:
                context = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in context file: {e}")
                return {
                    "context": [],
                    "timestamps": [],
                    "buffer_stats": {
                        "total_frames_processed": 0,
                        "frames_in_buffer": 0,
                        "last_clear_time": time.time(),
                        "token_usage": 0,
                        "frame_interval": None,
                        "buffer_health": 1.0,
                        "next_clear_time": None
                    }
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
                            
        # Use existing buffer stats if present, otherwise calculate them
        if "buffer_stats" not in context:
            context["buffer_stats"] = {
                "total_frames_processed": len(context.get("context", [])),
                "frames_in_buffer": len(context.get("context", [])),
                "last_clear_time": time.time(),
                "token_usage": sum(len(str(c.get("description", ""))) // 4 for c in context.get("context", [])),
                "frame_interval": context["context"][1]["metadata"]["timestamp"] - context["context"][0]["metadata"]["timestamp"] 
                    if len(context.get("context", [])) > 1 else None,
                "buffer_health": 1.0,
                "next_clear_time": None
            }
            
        logger.debug(f"Returning context with {len(context.get('context', []))} entries")
        return context
        
    except FileNotFoundError as e:
        logger.warning(f"Context file not found: {e}")
        return {
            "context": [],
            "timestamps": [],
            "buffer_stats": {
                "total_frames_processed": 0,
                "frames_in_buffer": 0,
                "last_clear_time": time.time(),
                "token_usage": 0,
                "frame_interval": None,
                "buffer_health": 1.0,
                "next_clear_time": None
            }
        }
    except Exception as e:
        logger.error(f"Error reading context: {e}", exc_info=True)
        return {"error": str(e), "context": []} 