import time
import asyncio
import json
from typing import Any, Dict, List, Optional
from collections import deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages temporal context from processed frames."""

    def __init__(
        self,
        max_context_frames: int = 10,
        max_context_tokens: int = 4000,
        rate_limit_rpm: int = 30,
        context_file: str = "data/frame_context.json"
    ):
        """Initialize the context manager.
        
        Args:
            max_context_frames: Maximum number of frames to keep in context
            max_context_tokens: Maximum number of tokens to keep in context
            rate_limit_rpm: Rate limit for background processing
            context_file: File to save/load context from
        """
        self.max_context_frames = max_context_frames
        self.max_context_tokens = max_context_tokens
        self.context_file = context_file
        self.buffer_stats = {
            "total_frames_processed": 0,
            "frames_in_buffer": 0,
            "last_clear_time": time.time(),
            "token_usage": 0,
            "frame_interval": None,  # Will be set from metadata
            "buffer_health": 1.0,  # 1.0 = healthy, 0.0 = critical
        }
        
        # Ensure data directory exists
        try:
            context_path = Path(context_file)
            context_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured context directory exists at: {context_path.parent}")
        except Exception as e:
            logger.error(f"Failed to create context directory: {e}")
            # Create a fallback path in the current directory
            self.context_file = "frame_context.json"
            logger.info(f"Using fallback context file: {self.context_file}")
        
        # Rate limiting
        self.request_delay = 60.0 / rate_limit_rpm if rate_limit_rpm else 0
        self.last_request_time = 0
        
        # Context storage with timestamps
        self._context_store = deque(maxlen=max_context_frames)
        self._timestamps = deque(maxlen=max_context_frames)
        
        # Try to load existing context
        self._load_context()

    def add_frame_context(self, context: Dict[str, Any]) -> None:
        """Add context from a processed frame."""
        current_time = time.time()
        
        # Update buffer stats
        self.buffer_stats["total_frames_processed"] += 1
        
        # Update frame interval from metadata
        if "metadata" in context and "timestamp" in context["metadata"]:
            if len(self._context_store) > 0:
                prev_timestamp = self._context_store[-1].get("metadata", {}).get("timestamp", 0)
                current_timestamp = context["metadata"]["timestamp"]
                self.buffer_stats["frame_interval"] = current_timestamp - prev_timestamp
        
        # Calculate token usage
        description_length = len(str(context.get("description", "")))
        self.buffer_stats["token_usage"] = description_length // 4
        
        # Add enhanced metadata
        context["processing_metadata"] = {
            "processed_at": current_time,
            "processing_duration": context.get("metadata", {}).get("processing_duration", 0),
            "token_usage": self.buffer_stats["token_usage"],
            "buffer_position": len(self._context_store),
            "total_frames_processed": self.buffer_stats["total_frames_processed"]
        }
        
        # Add new context
        self._context_store.append(context)
        self._timestamps.append(current_time)
        
        # Update buffer stats
        self.buffer_stats["frames_in_buffer"] = len(self._context_store)
        # Update buffer health based on actual max frames (5)
        self.buffer_stats["buffer_health"] = max(0.0, min(1.0, (self.max_context_frames - len(self._context_store)) / self.max_context_frames))
        
        # Save context after each frame
        self._save_context()
        
        # Prune if we exceed token limit or max frames
        total_tokens = sum(len(str(c.get("description", ""))) for c in self._context_store) // 4
        self.buffer_stats["token_usage"] = total_tokens
        
        # Remove oldest frames if we exceed max frames or token limit
        while (len(self._context_store) > self.max_context_frames or 
               total_tokens > self.max_context_tokens) and len(self._context_store) > 1:
            # Always keep at least one frame
            self._context_store.popleft()
            self._timestamps.popleft()
            total_tokens = sum(len(str(c.get("description", ""))) for c in self._context_store) // 4
            # Update buffer stats
            self.buffer_stats["frames_in_buffer"] = len(self._context_store)
            self.buffer_stats["buffer_health"] = max(0.0, min(1.0, (self.max_context_frames - len(self._context_store)) / self.max_context_frames))
            logger.info(f"Pruned buffer to {len(self._context_store)} frames (max: {self.max_context_frames}, tokens: {total_tokens})")
            # Save again if we pruned frames
            self._save_context()

    def get_current_context(self) -> Dict[str, Any]:
        """Get the most recent context.
        
        Returns:
            Most recent context information
        """
        if not self._context_store:
            return {}
        return self._context_store[-1]

    def get_temporal_context(self, window_size: int = 5) -> str:
        """Get a summary of recent temporal context.
        
        Args:
            window_size: Number of recent frames to include
            
        Returns:
            Formatted context summary
        """
        if not self._context_store:
            return "No historical context available."
            
        # Get recent context entries
        recent = list(self._context_store)[-window_size:]
        recent_times = list(self._timestamps)[-window_size:]
        
        # Format context summary
        summary = ["Previous context:"]
        for ctx, ts in zip(recent, recent_times):
            frame_index = ctx.get("frame_index", 0)
            timestamp = ctx.get("metadata", {}).get("timestamp", 0)
            description = ctx.get("description", "No description available")
            
            summary.append(
                f"\n[Frame {frame_index} at {timestamp:.1f}s]: {description}"
            )
        
        return "\n".join(summary)

    async def can_process(self) -> bool:
        """Check if we can process another frame based on rate limit."""
        if not self.request_delay:
            return True
            
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
            
        self.last_request_time = time.time()
        return True

    def _save_context(self) -> None:
        """Save context to file."""
        # Initialize data structure first
        data = {
            "context": list(self._context_store),
            "timestamps": list(self._timestamps),
            "buffer_stats": self.buffer_stats,
            "metadata": {
                "last_updated": time.time(),
                "max_context_frames": self.max_context_frames,
                "max_context_tokens": self.max_context_tokens
            }
        }
        
        try:
            # Ensure directory exists
            context_path = Path(self.context_file)
            context_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_file = str(context_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Rename temporary file to actual file (atomic operation)
            Path(temp_file).replace(context_path)
            
            logger.debug(f"Saved {len(self._context_store)} context entries to {self.context_file}")
            
        except Exception as e:
            logger.error(f"Error saving context to {self.context_file}: {e}", exc_info=True)
            
            # Try fallback location in current directory
            try:
                fallback_file = Path("frame_context.json")
                with open(fallback_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved context to fallback location: {fallback_file}")
            except Exception as e2:
                logger.error(f"Failed to save context to fallback location: {e2}", exc_info=True)

    def _load_context(self) -> None:
        """Load context from file."""
        try:
            context_path = Path(self.context_file)
            if context_path.exists() and context_path.stat().st_size > 0:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self._context_store = deque(
                        data.get("context", []),
                        maxlen=self.max_context_frames
                    )
                    self._timestamps = deque(
                        data.get("timestamps", []),
                        maxlen=self.max_context_frames
                    )
                    self.buffer_stats = data.get("buffer_stats", self._get_default_buffer_stats())
                    logger.info(f"Loaded {len(self._context_store)} context entries from {self.context_file}")
            else:
                # Initialize empty context file
                self._context_store.clear()
                self._timestamps.clear()
                self.buffer_stats = self._get_default_buffer_stats()
                self._save_context()  # This will create the file
                logger.info(f"Initialized empty context file: {self.context_file}")
                
        except Exception as e:
            logger.error(f"Error loading context from {self.context_file}: {e}", exc_info=True)
            self._context_store.clear()
            self._timestamps.clear()
            self.buffer_stats = self._get_default_buffer_stats()

    def _get_default_buffer_stats(self) -> dict:
        """Get default buffer statistics."""
        return {
            "total_frames_processed": 0,
            "frames_in_buffer": 0,
            "last_clear_time": time.time(),
            "token_usage": 0,
            "frame_interval": None,
            "buffer_health": 1.0
        } 