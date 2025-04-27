from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
import redis
import json
from typing import Dict, List, Optional, Any
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
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv
from text_processor import TextProcessor
import base64

app = FastAPI()

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ProfileManager and UserProfile from screen_understanding.core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from screen_understanding.core import ProfileManager, UserProfile
    logger.info("Successfully imported ProfileManager from screen_understanding.core")
except ImportError as e:
    logger.error(f"Error importing ProfileManager: {e}")
    # Define stub classes if import fails
    class ProfileManager:
        def __init__(self, profiles_file=None):
            self.profiles = {}
        def create_test_users(self):
            pass
    class UserProfile:
        pass

# Load environment variables
load_dotenv()

# Add port configuration
PORT = int(os.getenv('PORT', 8000))  # Default to 8000 if not specified
logger.info(f"Server configured to run on port {PORT}")

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini with the same settings as test_context_processing.py
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-lite')

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

class ChatRequest(BaseModel):
    history: List[Dict[str, Any]]
    currentFrame: Optional[Dict[str, Any]]
    messages: List[Dict[str, str]]

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with context awareness."""
    try:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Google API key not configured. Please set GOOGLE_API_KEY environment variable."
            )

        # Prepare context for the model
        context_text = "Screen Recording Context:\n"
        
        # Add historical context
        if request.history:
            context_text += "\nPrevious frames:\n"
            for frame in request.history[-5:]:  # Last 5 frames for context
                timestamp = frame.get("metadata", {}).get("timestamp", 0)
                description = frame.get("description", "No description available")
                context_text += f"[{timestamp:.1f}s] {description}\n"
        
        # Add current frame context
        if request.currentFrame:
            context_text += "\nCurrent frame:\n"
            timestamp = request.currentFrame.get("metadata", {}).get("timestamp", 0)
            description = request.currentFrame.get("description", "No description available")
            context_text += f"[{timestamp:.1f}s] {description}\n"
        
        # Add chat history
        chat_history = "\nChat history:\n"
        for msg in request.messages[:-1]:  # Exclude the latest message
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history += f"{role}: {msg['content']}\n"
        
        # Current user question
        current_question = request.messages[-1]["content"]
        
        # Prepare the prompt
        prompt = f"""You are an AI assistant helping to understand a screen recording.
Based on the context below, answer the user's question about what's happening in the recording.
Be specific and reference timestamps when relevant.

Format your response using markdown:
- Use **bold** for emphasis
- Use timestamps in `code` format
- Use bullet points for lists
- Use > for important quotes or highlights
- Use ### for section headers if needed

{context_text}

{chat_history}
User's question: {current_question}

Answer:"""

        # Generate response using Gemini with the same configuration as test_context_processing.py
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 1024,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="Empty response from Gemini API"
            )
            
        return {"response": response.text}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Initialize text processor
text_processor = TextProcessor(redis_client, redis_prefix="text:")  # Match frontend prefix

@app.post("/start_text_processing")
async def start_text_processing(
    text: str = Form(...),
    chunk_size: int = Form(...),
    delay_interval: float = Form(...),
    redis_prefix: str = Form("text:")
):
    """Start text processing with given parameters."""
    try:
        logger.info(f"Starting text processing with chunk_size={chunk_size}, delay_interval={delay_interval}")
        await text_processor.stream_text(text, chunk_size, delay_interval)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error starting text processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_text_processing")
async def stop_text_processing():
    """Stop text processing."""
    try:
        text_processor.stop_processing()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_text_data")
async def clear_text_data():
    """Clear all text data."""
    try:
        text_processor.clear_data()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/text_processing_status")
async def get_text_processing_status():
    """Get current text processing status."""
    try:
        return text_processor.get_processing_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/text_summary")
async def get_text_summary():
    """Get the current text summary from Redis."""
    try:
        logger.info("Retrieving text summary from Redis")
        summary_key = f"{text_processor.prefix}summary"
        logger.info(f"Using Redis key: {summary_key}")
        
        # Get the summary from Redis
        summary = text_processor.redis.get(summary_key)
        if summary:
            summary = summary.decode('utf-8')
        
        return {
            "status": "success",
            "summary": summary or "",
            "is_processing": text_processor.is_processing
        }
    except Exception as e:
        logger.error(f"Error getting text summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# User profile handling classes
class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    workload: Optional[str] = None
    blockers: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    
# Path to user profiles file
USER_PROFILES_FILE = os.path.join(WORKSPACE_ROOT, "user_profiles.json")

# Create a profile manager instance for direct access to profile functionality
profile_manager = ProfileManager(profiles_file=USER_PROFILES_FILE)

def load_profiles():
    """Load user profiles directly from the profiles file."""
    try:
        if os.path.exists(USER_PROFILES_FILE):
            # Always read directly from disk
            with open(USER_PROFILES_FILE, "r") as f:
                profiles = json.load(f)
            logger.info(f"Loaded {len(profiles)} profiles from {USER_PROFILES_FILE}")
            return profiles
        return {}
    except Exception as e:
        logger.error(f"Error loading profiles: {e}", exc_info=True)
        return {}
        
def save_profiles(profiles):
    """Save user profiles to the profiles file."""
    try:
        with open(USER_PROFILES_FILE, "w") as f:
            json.dump(profiles, f, indent=2)
        logger.info(f"Saved {len(profiles)} profiles to {USER_PROFILES_FILE}")
    except Exception as e:
        logger.error(f"Error saving profiles: {e}", exc_info=True)

@app.get("/profiles")
async def get_profiles(refresh: bool = False):
    """Get all user profiles from the file system or profile manager.
    
    Args:
        refresh: If True, force reload profiles from disk
    """
    try:
        # If refresh flag is set, force profile manager to reload from disk
        if refresh:
            logger.info("Forced reload of profiles from disk")
            # Reinitialize the profile manager to force reload
            global profile_manager
            profile_manager = ProfileManager(profiles_file=USER_PROFILES_FILE)
        
        # Load profiles directly from profile manager for more up-to-date data
        profiles_dict = {}
        for user_id, profile in profile_manager.profiles.items():
            # Convert UserProfile objects to dictionaries
            if hasattr(profile, 'to_dict'):
                profiles_dict[user_id] = profile.to_dict()
            else:
                # Fallback for dictionary-based profiles
                profiles_dict[user_id] = profile
        
        # If still no profiles, load from file
        if not profiles_dict:
            logger.info("No profiles in profile_manager, loading from disk")
            profiles_dict = load_profiles()
            
        logger.info(f"Returning {len(profiles_dict)} profiles")
        return profiles_dict
    except Exception as e:
        logger.error(f"Error getting profiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/profiles/{user_id}")
async def get_profile(user_id: str):
    """Get a specific user profile."""
    try:
        profiles = load_profiles()
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
        return profiles[user_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/profiles")
async def create_or_update_profile(profile_data: Dict[str, Any]):
    """Create or update a user profile."""
    try:
        user_id = profile_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
            
        profiles = load_profiles()
        
        # Create new profile or update existing one
        if user_id in profiles:
            # Update existing profile
            profiles[user_id].update(profile_data)
            profiles[user_id]["last_updated"] = time.time()
        else:
            # Create new profile
            profiles[user_id] = profile_data
            profiles[user_id]["last_updated"] = time.time()
            if "last_seen" not in profiles[user_id]:
                profiles[user_id]["last_seen"] = time.time()
                
        save_profiles(profiles)
        
        return profiles[user_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating/updating profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.patch("/profiles/{user_id}")
async def update_profile(user_id: str, update_data: UserProfileUpdate):
    """Update fields in a user profile."""
    try:
        profiles = load_profiles()
        
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
            
        # Update only the fields specified in the request
        profile = profiles[user_id]
        update_dict = update_data.dict(exclude_unset=True)
        
        for field, value in update_dict.items():
            if value is not None:
                profile[field] = value
                
        profile["last_updated"] = time.time()
        save_profiles(profiles)
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/profiles/clear-all")
async def clear_all_profiles():
    """Delete all user profiles."""
    try:
        # Save an empty dictionary to the profiles file
        save_profiles({})
        
        # Also clear the profile_manager profiles
        profile_manager.profiles = {}
        
        logger.info("Cleared all user profiles")
        
        return {
            "status": "success", 
            "message": "All user profiles have been cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing all profiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{user_id}/blockers")
async def add_blocker(user_id: str, blocker_data: Dict[str, str]):
    """Add a blocker to a user profile."""
    try:
        blocker = blocker_data.get("blocker")
        if not blocker:
            raise HTTPException(status_code=400, detail="blocker is required")
            
        profiles = load_profiles()
        
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
            
        profile = profiles[user_id]
        
        # Initialize blockers list if it doesn't exist
        if "blockers" not in profile:
            profile["blockers"] = []
            
        # Add blocker if it doesn't already exist
        if blocker not in profile["blockers"]:
            profile["blockers"].append(blocker)
            
        profile["last_updated"] = time.time()
        save_profiles(profiles)
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding blocker for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.delete("/profiles/{user_id}/blockers/{blocker_index}")
async def remove_blocker(user_id: str, blocker_index: int):
    """Remove a blocker from a user profile."""
    try:
        profiles = load_profiles()
        
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
            
        profile = profiles[user_id]
        
        if "blockers" not in profile or blocker_index >= len(profile["blockers"]):
            raise HTTPException(status_code=404, detail=f"Blocker at index {blocker_index} not found")
            
        # Remove the blocker
        profile["blockers"].pop(blocker_index)
        profile["last_updated"] = time.time()
        save_profiles(profiles)
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing blocker for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/profiles/{user_id}/decisions")
async def add_decision(user_id: str, decision_data: Dict[str, Any]):
    """Add a decision to a user profile."""
    try:
        description = decision_data.get("description")
        if not description:
            raise HTTPException(status_code=400, detail="decision description is required")
            
        status = decision_data.get("status", "pending")
        
        profiles = load_profiles()
        
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
            
        profile = profiles[user_id]
        
        # Initialize decisions list if it doesn't exist
        if "decisions" not in profile:
            profile["decisions"] = []
            
        # Add the decision
        decision = {
            "description": description,
            "status": status,
            "timestamp": time.time()
        }
        profile["decisions"].append(decision)
        
        profile["last_updated"] = time.time()
        save_profiles(profiles)
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding decision for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.patch("/profiles/{user_id}/decisions/{decision_index}")
async def update_decision_status(user_id: str, decision_index: int, status_data: Dict[str, str]):
    """Update the status of a decision in a user profile."""
    try:
        new_status = status_data.get("status")
        if not new_status:
            raise HTTPException(status_code=400, detail="status is required")
            
        profiles = load_profiles()
        
        if user_id not in profiles:
            raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")
            
        profile = profiles[user_id]
        
        if "decisions" not in profile or decision_index >= len(profile["decisions"]):
            raise HTTPException(status_code=404, detail=f"Decision at index {decision_index} not found")
            
        # Update the decision status
        profile["decisions"][decision_index]["status"] = new_status
        profile["last_updated"] = time.time()
        save_profiles(profiles)
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating decision status for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-profiles-from-gemini")
async def update_profiles_from_gemini():
    """Force a manual update of user profiles using Gemini on the latest frame."""
    try:
        # Get the latest frame from Redis
        all_keys = sorted([
            k.decode() for k in redis_client.keys("test:*")
            if not k.decode().endswith(":meta")
        ])
        
        if not all_keys:
            raise HTTPException(status_code=404, detail="No frames found in Redis")
            
        latest_frame_key = all_keys[-1]
        frame_data = redis_client.get(latest_frame_key)
        meta_data = redis_client.get(f"{latest_frame_key}:meta")
        
        if not frame_data or not meta_data:
            raise HTTPException(status_code=404, detail="Frame data missing")
            
        # Parse metadata
        try:
            metadata = ast.literal_eval(meta_data.decode())
        except:
            metadata = {"error": "Could not parse metadata"}
        
        logger.info(f"Retrieved latest frame with key {latest_frame_key}, frame number: {metadata.get('frame_number')}")
        
        # Get existing profiles to match against
        existing_profiles = {}
        for user_id, profile in profile_manager.profiles.items():
            if hasattr(profile, 'to_dict'):
                existing_profiles[user_id] = profile.to_dict()
            else:
                existing_profiles[user_id] = profile
                
        if not existing_profiles:
            logger.warning("No existing user profiles found for matching. Please create user profiles manually first.")
            return {
                "status": "warning",
                "message": "No existing user profiles found for matching. Please create user profiles manually first."
            }
            
        # Create a list of names to match against
        profile_names = [
            {"user_id": user_id, 
             "name": profile.get("name", user_id), 
             "role": profile.get("role", "")} 
            for user_id, profile in existing_profiles.items()
        ]
        
        # Use Gemini to analyze the frame for user profiles with fuzzy matching
        logger.info("Querying Gemini to identify user profiles with fuzzy matching")
        prompt = f"""Analyze this screen image and identify ONLY the users from the provided list:

USER LIST FOR MATCHING:
{json.dumps(profile_names, indent=2)}

INSTRUCTIONS:
1. ONLY identify users from the above list.
2. Use fuzzy matching to identify users by name or role if exact matches aren't found.
3. For each identified user, extract:
   - Current workload (high/medium/low) if apparent
   - Any blockers they might have
   - Any decisions they need to make
   - Activities they're engaged in

Format your response as a structured JSON with this schema:
{{
    "users": [
        {{
            "user_id": "string", // MUST be one from the provided list
            "workload": "string", // high, medium, or low
            "blockers": ["string"],
            "decisions": [
                {{
                    "description": "string",
                    "status": "string" // "pending" or "made"
                }}
            ],
            "activities": ["string"],
            "skills": ["string"]
        }}
    ]
}}

IMPORTANT:
- ONLY include users from the provided list that appear in the image
- Match each user based on their name or role using fuzzy matching
- If none of the users in the list appear in the image, return an empty users array"""
        
        # Make the Gemini API call - using the proper content format for Gemini
        try:
            logger.info("Sending image to Gemini for analysis")
            
            # Create a proper prompt that includes both text and image
            content = [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(frame_data).decode('ascii')}}
                    ]
                }
            ]
            
            response = model.generate_content(content)
            user_profiles = response.text
            
            logger.info(f"Received response from Gemini: {user_profiles[:200]}...")
            
            # Try to parse the response as JSON
            try:
                # Clean up the response to remove any markdown code blocks or other formatting
                cleaned_response = user_profiles.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response.replace("```json", "", 1)
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                user_profiles_json = json.loads(cleaned_response)
                logger.info(f"Successfully parsed JSON response: {json.dumps(user_profiles_json, indent=2)[:200]}...")
                
                # Process user profiles
                if "users" in user_profiles_json and isinstance(user_profiles_json["users"], list):
                    logger.info(f"Found {len(user_profiles_json['users'])} users in the frame")
                    
                    for user_data in user_profiles_json["users"]:
                        user_id = user_data.get("user_id")
                        if not user_id:
                            logger.warning("User data missing user_id, skipping")
                            continue
                            
                        # Verify this user exists in our list
                        if user_id not in existing_profiles:
                            logger.warning(f"User {user_id} not found in existing profiles, skipping")
                            continue
                            
                        # Get the profile from the manager
                        profile = profile_manager.get_profile(user_id)
                        logger.info(f"Processing user profile for user_id: {user_id}")
                        
                        # Update workload
                        if "workload" in user_data and user_data["workload"]:
                            profile.update_workload(user_data["workload"])
                            
                        # Update blockers
                        if "blockers" in user_data and isinstance(user_data["blockers"], list):
                            for blocker in user_data["blockers"]:
                                profile.add_blocker(blocker)
                                
                        # Update decisions
                        if "decisions" in user_data and isinstance(user_data["decisions"], list):
                            for decision in user_data["decisions"]:
                                description = decision.get("description", "")
                                status = decision.get("status", "pending")
                                if description:
                                    profile.add_decision(description, status)
                                    
                        # Update activities
                        if "activities" in user_data and isinstance(user_data["activities"], list):
                            for activity in user_data["activities"]:
                                profile.add_activity(activity)
                                
                        # Update skills
                        if "skills" in user_data and isinstance(user_data["skills"], list):
                            for skill in user_data["skills"]:
                                if skill not in profile.skills:
                                    profile.skills.append(skill)
                                
                        # Update last seen timestamp
                        timestamp = metadata.get("timestamp")
                        profile.update_last_seen(timestamp)
                        
                        # Update the profile in the manager
                        profile_manager.update_profile(profile)
                        logger.info(f"Updated profile for user {user_id}")
                
                return {
                    "status": "success", 
                    "message": f"Updated user profiles from Gemini", 
                    "users": user_profiles_json.get("users", [])
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                cleaned_text = user_profiles.replace("\n", " ")[:200]
                logger.error(f"Raw response snippet: {cleaned_text}")
                return {
                    "status": "error",
                    "message": "Failed to parse Gemini response as JSON",
                    "raw_response": user_profiles
                }
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error calling Gemini API: {str(e)}",
                "details": str(e)
            }
        
    except Exception as e:
        logger.error(f"Error updating profiles from Gemini: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error updating profiles from Gemini: {str(e)}",
            "details": str(e)
        } 