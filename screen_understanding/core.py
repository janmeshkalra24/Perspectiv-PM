from typing import Any, Dict, Optional
import asyncio
import logging
from pathlib import Path

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
        context_file: str = "frame_context.json"
    ):
        """Initialize the screen understanding system.
        
        Args:
            model: API client for visual understanding
            source: Data source for video frames
            max_context_frames: Maximum number of frames to keep in context
            max_context_tokens: Maximum number of tokens in context
            context_file: File to save/load context from
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
        self._background_task = None

    async def _process_frame_background(self, frame_data: Dict[str, Any]):
        """Process a frame in the background for context building.
        
        Args:
            frame_data: Frame data including image and metadata
        """
        try:
            # Check rate limiting
            if not await self.context_manager.can_process():
                return
                
            # Process frame
            result = await self.model.process_image(frame_data["image_data"])
            
            # Add metadata to result
            result.update({
                "frame_index": frame_data["frame_index"],
                "metadata": frame_data["metadata"]
            })
            
            # Add to context
            self.context_manager.add_frame_context(result)
            
        except Exception as e:
            logger.error(f"Error in background processing: {e}")

    async def start(self) -> None:
        """Start processing frames from the source."""
        if self._is_running:
            return

        self._is_running = True
        try:
            while self._is_running:
                frame_data = await self.source.get_frame()
                if frame_data is None:
                    await asyncio.sleep(0.1)
                    continue

                # Store current frame
                self._current_frame = frame_data
                
                # Process in background for context
                if not self._background_task or self._background_task.done():
                    self._background_task = asyncio.create_task(
                        self._process_frame_background(frame_data)
                    )
                
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            raise
        finally:
            self._is_running = False

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
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass 