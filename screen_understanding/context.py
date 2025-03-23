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
        retention_period: int = 3600,
        max_context_frames: int = 10,
        max_context_tokens: int = 4000,
        rate_limit_rpm: int = 30,
        context_file: str = "data/frame_context.json"
    ):
        """Initialize the context manager.
        
        Args:
            retention_period: How long to retain context in seconds
            max_context_frames: Maximum number of frames to keep in context
            max_context_tokens: Maximum number of tokens to keep in context
            rate_limit_rpm: Rate limit for background processing
            context_file: File to save/load context from
        """
        self.retention_period = retention_period
        self.max_context_frames = max_context_frames
        self.max_context_tokens = max_context_tokens
        self.context_file = context_file
        
        # Ensure data directory exists
        Path(context_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.request_delay = 60.0 / rate_limit_rpm if rate_limit_rpm else 0
        self.last_request_time = 0
        
        # Context storage with timestamps
        self._context_store = deque(maxlen=max_context_frames)
        self._timestamps = deque(maxlen=max_context_frames)
        
        # Try to load existing context
        self._load_context()

    def add_frame_context(self, context: Dict[str, Any]) -> None:
        """Add context from a processed frame.
        
        Args:
            context: Context information from the frame
        """
        current_time = time.time()
        
        # Clean up old context
        self._cleanup_old_context(current_time)
        
        # Add new context
        self._context_store.append(context)
        self._timestamps.append(current_time)
        
        # Save context periodically (every 10 frames)
        if len(self._context_store) % 10 == 0:
            self._save_context()
        
        # Prune if we exceed token limit (rough estimate: 4 chars = 1 token)
        total_tokens = sum(
            len(str(c.get("description", ""))) 
            for c in self._context_store
        ) // 4
        
        while total_tokens > self.max_context_tokens and self._context_store:
            self._context_store.popleft()
            self._timestamps.popleft()

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

    def _cleanup_old_context(self, current_time: float) -> None:
        """Remove context older than retention period."""
        if not self._timestamps:
            return
            
        while (
            self._timestamps and 
            current_time - self._timestamps[0] > self.retention_period
        ):
            self._timestamps.popleft()
            self._context_store.popleft()

    def _save_context(self) -> None:
        """Save context to file."""
        try:
            data = {
                "context": list(self._context_store),
                "timestamps": list(self._timestamps)
            }
            with open(self.context_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving context: {e}")

    def _load_context(self) -> None:
        """Load context from file."""
        try:
            if Path(self.context_file).exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self._context_store = deque(
                        data["context"],
                        maxlen=self.max_context_frames
                    )
                    self._timestamps = deque(
                        data["timestamps"],
                        maxlen=self.max_context_frames
                    )
                    logger.info(f"Loaded {len(self._context_store)} context entries")
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            self._context_store.clear()
            self._timestamps.clear() 