from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
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
from knowledge_graph import ScreenKnowledgeGraph

app = FastAPI()

# Load environment variables
load_dotenv()

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
        logger.debug(f"Attempting to fetch frame: {frame_key}")
        
        # Validate frame key
        if not frame_key or ':' not in frame_key:
            logger.warning(f"Invalid frame key format: {frame_key}")
            raise HTTPException(status_code=400, detail="Invalid frame key format")
            
        # Decode URL encoded key if needed
        decoded_key = frame_key
        
        # Get frame data
        frame_data = redis_client.get(decoded_key)
        if not frame_data:
            logger.warning(f"Frame not found: {decoded_key}")
            
            # Try checking if Redis is connected
            try:
                redis_client.ping()
                logger.info("Redis connection is active")
            except Exception as redis_err:
                logger.error(f"Redis connection error: {redis_err}")
                raise HTTPException(status_code=500, detail=f"Redis connection error: {redis_err}")
                
            raise HTTPException(status_code=404, detail=f"Frame not found: {decoded_key}")
        
        logger.debug(f"Successfully retrieved frame: {decoded_key}, size: {len(frame_data)} bytes")
        return Response(content=frame_data, media_type="image/jpeg")
        
    except HTTPException:
        # Re-raise HTTP exceptions as they're already well-formed
        raise
    except Exception as e:
        logger.error(f"Error fetching frame {frame_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching frame: {str(e)}")

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
                logger.info(f"Stopping process: {process_type}")
                os.killpg(os.getpgid(running_processes[process_type].pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Error stopping process {process_type}: {e}")
        running_processes.clear()
        
        # Clear Redis data
        logger.info("Clearing Redis data")
        redis_client.flushall()
        
        # Clear process logs
        logger.info("Clearing process logs")
        for q in process_logs.values():
            while not q.empty():
                q.get()
        
        # Clear Neo4j database (knowledge graph)
        logger.info("Clearing Neo4j database")
        try:
            # Clear all nodes and relationships
            with knowledge_graph.driver.session() as session:
                # First remove all relationships
                session.run("MATCH ()-[r]->() DELETE r")
                # Then remove all nodes
                session.run("MATCH (n) DELETE n")
                logger.info("Neo4j database cleared successfully")
        except Exception as neo4j_err:
            logger.error(f"Error clearing Neo4j database: {neo4j_err}")
        
        # Clear context file
        logger.info("Clearing context file")
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
        
        # Also reinitialize the knowledge graph schema
        try:
            logger.info("Reinitializing knowledge graph schema")
            # Recreate schema constraints and indexes
            knowledge_graph._initialize_schema()
        except Exception as schema_err:
            logger.error(f"Error reinitializing knowledge graph schema: {schema_err}")
                
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        logger.error(f"Error clearing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    history: List[Dict[str, Any]]
    currentFrame: Optional[Dict[str, Any]]
    messages: List[Dict[str, str]]

# Initialize knowledge graph
knowledge_graph = ScreenKnowledgeGraph('models/gemini-2.0-flash-lite')

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
        context = {
            "history": request.history,
            "currentFrame": request.currentFrame,
            "messages": request.messages[:-1]  # Previous messages
        }
        
        # Current user question
        current_question = request.messages[-1]["content"]
        
        try:
            # Try to get answer from knowledge graph first
            answer = await knowledge_graph.query_knowledge_graph(current_question, context)
            
            # If knowledge graph fails or returns a fallback message, use traditional approach
            if "error" in answer.lower() or "falling back" in answer.lower():
                # Prepare context text as before
                context_text = "Screen Recording Context:\n"
                
                if request.history:
                    context_text += "\nPrevious frames:\n"
                    for frame in request.history[-5:]:
                        timestamp = frame.get("metadata", {}).get("timestamp", 0)
                        description = frame.get("description", "No description available")
                        context_text += f"[{timestamp:.1f}s] {description}\n"
                
                if request.currentFrame:
                    context_text += "\nCurrent frame:\n"
                    timestamp = request.currentFrame.get("metadata", {}).get("timestamp", 0)
                    description = request.currentFrame.get("description", "No description available")
                    context_text += f"[{timestamp:.1f}s] {description}\n"
                
                chat_history = "\nChat history:\n"
                for msg in request.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n"
                
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

                # Generate response using Gemini with the same configuration
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
                
                answer = response.text if response.text else "Sorry, I couldn't answer that question."
            
            return {"response": answer}
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_data")
async def get_graph_data():
    """Get the current state of the knowledge graph for visualization."""
    try:
        # Query nodes with safer property access
        nodes_query = """
        MATCH (n)
        WITH n, labels(n) as labels
        RETURN COLLECT({
            id: elementId(n),
            label: CASE 
                WHEN n.label IS NOT NULL THEN n.label
                WHEN n.frame_id IS NOT NULL THEN toString(n.frame_id)
                WHEN n.element_id IS NOT NULL THEN toString(n.element_id)
                ELSE toString(elementId(n))
            END,
            type: head(labels),
            timestamp: CASE WHEN n.timestamp IS NOT NULL THEN n.timestamp ELSE 0 END,
            description: CASE WHEN n.description IS NOT NULL THEN n.description ELSE '' END
            // Omitting content and category fields that cause warnings
        }) as nodes
        """
        
        # Query relationships with more details
        relationships_query = """
        MATCH (source)-[r]->(target)
        RETURN COLLECT({
            source: elementId(source),
            target: elementId(target),
            type: type(r),
            properties: {}  // Using empty object to avoid warnings
        }) as links
        """
        
        # Execute queries
        with knowledge_graph.driver.session() as session:
            nodes_result = session.run(nodes_query).single()
            links_result = session.run(relationships_query).single()
            
            # Filter out null properties and ensure data is present
            nodes = []
            if nodes_result and "nodes" in nodes_result:
                nodes = [
                    {k: v for k, v in node.items() if v is not None}
                    for node in nodes_result["nodes"]
                ]
            
            links = []
            if links_result and "links" in links_result:
                links = links_result["links"]
            
            return {
                "nodes": nodes,
                "links": links
            }
            
    except Exception as e:
        logger.error(f"Error fetching graph data: {e}")
        # Return empty structure instead of error
        return {
            "nodes": [],
            "links": []
        }

@app.get("/visualization_graph")
async def get_visualization_graph():
    """Get a UI-optimized version of the knowledge graph for visualization."""
    try:
        # Query core frame data with safer query that avoids property warnings
        frame_query = """
        MATCH (f:Frame)
        RETURN COLLECT({
            id: toString(elementId(f)),
            frame_id: f.frame_id,
            timestamp: CASE WHEN f.timestamp IS NOT NULL THEN f.timestamp ELSE 0 END,
            description: CASE WHEN f.description IS NOT NULL THEN f.description ELSE '' END
        }) as frames
        """
        
        # Query relationships between frames (temporal)
        frame_rel_query = """
        MATCH (f1:Frame)-[r]->(f2:Frame)
        RETURN COLLECT({
            source: toString(elementId(f1)),
            target: toString(elementId(f2)),
            type: type(r)
        }) as frame_links
        """
        
        with knowledge_graph.driver.session() as session:
            # Get frames
            frames_result = session.run(frame_query).single()
            frame_links_result = session.run(frame_rel_query).single()
            
            frames = frames_result["frames"] if frames_result and "frames" in frames_result else []
            frame_links = frame_links_result["frame_links"] if frame_links_result and "frame_links" in frame_links_result else []
            
            # Transform into visualization structure
            nodes = []
            links = []
            
            # Always add root node
            nodes.append({
                "id": "root",
                "label": "Screen Recording Analysis",
                "type": "root",
                "group": "root"
            })
            
            # If there are no frames, return minimal structure
            if not frames:
                logger.info("No frames found in Neo4j, returning minimal visualization")
                return {
                    "nodes": [
                        {
                            "id": "root",
                            "label": "Screen Recording Analysis (No Data)",
                            "type": "root",
                            "group": "root"
                        },
                        {
                            "id": "empty_state",
                            "label": "No recording data available",
                            "type": "container",
                            "group": "timeline"
                        }
                    ],
                    "links": [
                        {
                            "source": "root",
                            "target": "empty_state",
                            "type": "CONTAINS"
                        }
                    ]
                }
            
            # Add timeline container
            nodes.append({
                "id": "timeline",
                "label": f"Timeline ({len(frames)} frames)",
                "type": "container",
                "group": "timeline"
            })
            links.append({
                "source": "root",
                "target": "timeline",
                "type": "CONTAINS"
            })
            
            # Process frames
            for frame in frames:
                try:
                    # Check if frame_id is defined and valid
                    if "frame_id" not in frame or frame["frame_id"] is None:
                        logger.warning(f"Frame missing frame_id: {frame}")
                        continue
                        
                    frame_id = frame.get("frame_id", -1)
                    timestamp = float(frame.get("timestamp", 0))
                    node_id = f"frame_{frame_id}"
                    
                    # Create frame node with safe f-string formatting
                    nodes.append({
                        "id": node_id,
                        "label": f"Frame {frame_id} ({timestamp:.1f}s)",
                        "type": "frame",
                        "group": "frame",
                        "timestamp": timestamp,
                        "description": frame.get("description", "")
                    })
                    
                    # Link to timeline
                    links.append({
                        "source": "timeline",
                        "target": node_id,
                        "type": "CONTAINS"
                    })
                except Exception as e:
                    logger.warning(f"Error processing frame {frame.get('frame_id', 'unknown')}: {e}")
                    continue
            
            # Add temporal links between frames
            for link in frame_links:
                source_id = link.get("source")
                target_id = link.get("target")
                if source_id and target_id:
                    links.append({
                        "source": f"frame_{source_id}",
                        "target": f"frame_{target_id}",
                        "type": link.get("type", "NEXT")
                    })
            
            # Extract insights from frame descriptions
            insights = {}
            for frame in frames:
                if frame.get("description"):
                    try:
                        # Use the existing LLM to extract insights
                        prompt = """Analyze this frame description and extract key elements:
Description: {}

Extract and categorize elements into these types:
1. applications: List of application names or windows
2. technical_terms: List of technical terms or concepts
3. user_actions: List of user interactions or actions
4. tasks: List of tasks or todos
5. warnings: List of warnings or errors

Return a Python dictionary with these exact keys and list values.
Example:
{{"applications": ["Chrome", "VS Code"], "technical_terms": ["API"], "user_actions": ["clicked button"], "tasks": ["fix bug"], "warnings": ["error"]}}""".format(frame.get("description", ""))
                        
                        response = knowledge_graph.llm.invoke(prompt)
                        
                        # Create a default fallback structure
                        default_insights = {
                            "applications": [],
                            "technical_terms": [],
                            "user_actions": [],
                            "tasks": [],
                            "warnings": []
                        }
                        
                        # Extract text based on response type
                        try:
                            if hasattr(response, 'text'):
                                if isinstance(response.text, str):
                                    response_text = response.text.strip()
                                elif callable(response.text):
                                    response_text = str(response.text())
                                else:
                                    response_text = str(response)
                            elif isinstance(response, str):
                                response_text = response.strip()
                            elif hasattr(response, 'content'):
                                response_text = str(response.content)
                            else:
                                response_text = str(response)
                                
                            # Clean the response text to ensure it's valid Python literal
                            response_text = response_text.replace("'", '"').replace('\n', ' ')
                            
                            # Try to parse as Python literal
                            try:
                                frame_insights = ast.literal_eval(response_text)
                                
                                # Validate the response structure
                                expected_keys = {"applications", "technical_terms", "user_actions", "tasks", "warnings"}
                                if not isinstance(frame_insights, dict) or not all(k in frame_insights for k in expected_keys):
                                    logger.warning(f"Invalid insight format, using defaults")
                                    frame_insights = default_insights
                            except Exception as parsing_error:
                                logger.warning(f"Error parsing insight JSON: {parsing_error}")
                                frame_insights = default_insights
                        except Exception as text_error:
                            logger.warning(f"Error extracting response text: {text_error}")
                            frame_insights = default_insights
                            
                        # Add insights to categories
                        for category, items in frame_insights.items():
                            if not isinstance(items, list):
                                items = []
                            if category not in insights:
                                insights[category] = set()
                            insights[category].update(items)
                    except Exception as e:
                        logger.warning(f"Error extracting insights from frame {frame.get('frame_id', 'unknown')}: {e}")
                        # Add empty categories if needed
                        for category in ["applications", "technical_terms", "user_actions", "tasks", "warnings"]:
                            if category not in insights:
                                insights[category] = set()
                        continue
            
            # Add insight nodes
            for category, items in insights.items():
                try:
                    # Add category container
                    category_id = f"category_{category}"
                    nodes.append({
                        "id": category_id,
                        "label": category.replace("_", " ").title(),
                        "type": "container",
                        "group": category
                    })
                    links.append({
                        "source": "root",
                        "target": category_id,
                        "type": "CONTAINS"
                    })
                    
                    # Add items
                    for item in items:
                        try:
                            item_str = str(item)
                            item_id = f"{category}_{abs(hash(item_str))}"  # Use abs to avoid negative hashes
                            nodes.append({
                                "id": item_id,
                                "label": item_str,
                                "type": "insight",
                                "group": category
                            })
                            links.append({
                                "source": category_id,
                                "target": item_id,
                                "type": "CONTAINS"
                            })
                        except Exception as e:
                            logger.warning(f"Error processing insight item in category {category}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Error processing category {category}: {e}")
                    continue
            
            return {
                "nodes": nodes,
                "links": links
            }
            
    except Exception as e:
        logger.error(f"Error creating visualization graph: {e}")
        # Return a minimal valid structure instead of an error
        return {
            "nodes": [
                {
                    "id": "root",
                    "label": "Screen Recording Analysis (Error)",
                    "type": "root",
                    "group": "root"
                },
                {
                    "id": "error_state",
                    "label": f"Error: {str(e)}",
                    "type": "container",
                    "group": "warnings"
                }
            ],
            "links": [
                {
                    "source": "root",
                    "target": "error_state",
                    "type": "CONTAINS"
                }
            ]
        } 