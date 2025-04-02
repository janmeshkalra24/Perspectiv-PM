from typing import Any, Dict, Optional
import asyncio
import logging
from pathlib import Path
from collections import deque
from asyncio import Queue

from .api.base import BaseAPIClient
from .sources.base import DataSource
from .context import ContextManager

logger = logging.getLogger(__name__)

class ScreenUnderstanding:
    """Main class for screen understanding and VQA system."""

    def __init__(
        self,
        model: BaseAPIClient,
        source: DataSource,
        max_context_frames: int = 10,
        max_context_tokens: int = 4000,
        context_file: str = "frame_context.json",
        max_queue_size: int = 50
    ):
        """Initialize the screen understanding system.
        
        Args:
            model: API client for visual understanding
            source: Data source for video frames
            max_context_frames: Maximum number of frames to keep in context
            max_context_tokens: Maximum number of tokens in context
            context_file: File to save/load context from
            max_queue_size: Maximum number of frames to queue for processing
        """
        self.model = model
        self.source = source
        self.context_manager = ContextManager(
            max_context_frames=max_context_frames,
            max_context_tokens=max_context_tokens,
            rate_limit_rpm=getattr(model, 'rate_limit_rpm', 30),
            context_file=context_file
        )
        self._is_running = False
        self._current_frame = None
        self._frame_queue = Queue(maxsize=max_queue_size)
        self._processing_task = None
        self._processed_frames = set()  # Track processed frame indices

    async def _process_frames(self):
        """Process frames from the queue."""
        while self._is_running:
            try:
                frame_data = await self._frame_queue.get()
                frame_index = frame_data.get("frame_index")
                frame_key = frame_data.get("key", f"test:{frame_index}")  # Get key or construct it
                
                # Skip if already processed
                if frame_index in self._processed_frames:
                    logger.info(f"Skipping already processed frame {frame_index}")
                    self._frame_queue.task_done()
                    continue
                
                # Check rate limiting
                if not await self.context_manager.can_process():
                    logger.info("Rate limiting applied, waiting before processing next frame")
                    await asyncio.sleep(self.context_manager.request_delay)
                    # Put frame back in queue
                    await self._frame_queue.put(frame_data)
                    self._frame_queue.task_done()
                    continue
                
                logger.info(f"Processing frame {frame_index} with model...")
                
                # Process frame
                try:
                    # First get general description
                    result = await self.model.process_image(frame_data["image_data"])
                    
                    # Then extract PM-specific insights with structured prompts
                    pm_insights = await self.model.answer_question(
                        """Analyze this frame from a product management perspective. Provide ONLY the most critical insights in a concise format.

STRICT RULES:
1. Each item MUST be 100 characters or less
2. Each category MUST have at most 3 items
3. Use bullet points only for actual items
4. Skip any category that has no relevant items
5. NO explanatory text or filler words

Return ONLY this JSON format:
{
    "sprint_goals": [
        "Implement user auth by EOW",
        "Complete API docs"
    ],
    "key_metrics": [
        "API response time < 200ms",
        "Test coverage > 85%"
    ],
    "feature_status": [
        "Auth: 80% done, pending security review",
        "API docs: 20% complete"
    ],
    "dependencies": [
        "Auth service needs updated identity provider",
        "Mobile app blocked on API"
    ],
    "risks": [
        "Security review may delay auth release",
        "Limited backend capacity"
    ],
    "next_steps": [
        "Schedule security review",
        "Start API documentation"
    ],
    "decisions": [
        "Using OAuth2 for auth flow",
        "Postponing analytics to next sprint"
    ],
    "stakeholder_requests": [
        "Marketing needs user flows by Friday",
        "Support team requests better error messages"
    ]
}""",
                        {"image_data": frame_data["image_data"]}
                    )
                    
                    # Add PM insights to result
                    try:
                        result["pm_insights"] = pm_insights
                    except:
                        logger.warning(f"Could not parse PM insights for frame {frame_index}")
                        result["pm_insights"] = {}
                    
                    logger.info(f"Model returned result for frame {frame_index}: {result.get('description', '')[:100]}...")
                    
                    # Add metadata to result
                    result.update({
                        "frame_index": frame_index,
                        "frame_key": frame_key,  # Add frame key for UI
                        "metadata": frame_data["metadata"]
                    })
                    
                    # Add to context
                    self.context_manager.add_frame_context(result)
                    logger.info(f"Added frame {frame_index} to context")
                    
                    # Mark as processed
                    self._processed_frames.add(frame_index)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}", exc_info=True)
                
                finally:
                    self._frame_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight error loop

    async def start(self) -> None:
        """Start processing frames from the source."""
        if self._is_running:
            return

        self._is_running = True
        self._processed_frames.clear()  # Clear processed frames set
        logger.info("Starting frame processing...")
        
        # Start processing task
        self._processing_task = asyncio.create_task(self._process_frames())
        
        try:
            while self._is_running:
                frame_data = await self.source.get_frame()
                if frame_data is None:
                    await asyncio.sleep(0.1)
                    continue

                frame_index = frame_data.get("frame_index", -1)
                logger.info(f"Received frame {frame_index} from source")
                
                # Store current frame
                self._current_frame = frame_data
                
                # Skip if already queued
                if frame_index in self._processed_frames:
                    logger.info(f"Skipping already processed frame {frame_index}")
                    continue
                
                # Add to processing queue
                try:
                    # Try to add to queue with a timeout
                    await asyncio.wait_for(
                        self._frame_queue.put(frame_data),
                        timeout=0.1
                    )
                    logger.info(f"Queued frame {frame_index} for processing")
                except asyncio.TimeoutError:
                    logger.warning(f"Queue full, skipping frame {frame_index}")
                
        except Exception as e:
            logger.error(f"Error in frame processing: {e}", exc_info=True)
            raise
        finally:
            self._is_running = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

    async def ask(self, question: str) -> str:
        """Ask a question about the current frame.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer from the model
        """
        if not self._current_frame:
            raise ValueError("No frame available")
            
        # Get temporal context
        context_summary = self.context_manager.get_temporal_context()
        
        # Enhance question with context
        enhanced_question = f"""Question: {question}

Previous context from recent frames:
{context_summary}

Please answer the question based on the current frame, using the context from previous frames if relevant."""
        
        # Get answer
        return await self.model.answer_question(
            enhanced_question,
            {"image_data": self._current_frame["image_data"]}
        )

    async def stop(self) -> None:
        """Stop the system."""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass 