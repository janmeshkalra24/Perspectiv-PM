#!/usr/bin/env python3

import argparse
import asyncio
import cv2
import redis
from pathlib import Path
import logging

from screen_understanding import ScreenUnderstanding
from screen_understanding.api import APIClientFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleFrameSource:
    """Simple source that provides a single frame."""
    
    def __init__(self, frame_data: bytes):
        self.frame_data = frame_data
        self.frame_sent = False
    
    async def get_frame(self):
        """Return the frame once."""
        if not self.frame_sent:
            self.frame_sent = True
            return {
                "image_data": self.frame_data,
                "metadata": {"frame_number": 0},
                "frame_index": 0
            }
        return None
    
    async def cleanup(self):
        pass

async def main():
    parser = argparse.ArgumentParser(description="Test single frame processing")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number to process (0-based)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["llava", "huggingface", "gemini"],
        help="Model to use for processing"
    )
    
    args = parser.parse_args()
    
    # Load the specific frame from video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.video}")
        return
        
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} is out of range. Video has {total_frames} frames.")
        return
    
    # Seek to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {args.frame}")
        return
    
    # Convert frame to bytes
    _, buffer = cv2.imencode(".jpg", frame)
    frame_bytes = buffer.tobytes()
    
    print(f"\nLoaded frame {args.frame} ({len(frame_bytes)} bytes)")
    
    try:
        # Initialize model and system with single frame source
        model = APIClientFactory.create(args.model)
        source = SingleFrameSource(frame_bytes)
        system = ScreenUnderstanding(model=model, source=source)
        
        # Start system
        system_task = asyncio.create_task(system.start())
        
        # Ask a test question
        print("\nAsking test question...")
        try:
            answer = await asyncio.wait_for(
                system.ask("What is shown in this frame?"),
                timeout=30.0
            )
            print(f"\nAnswer: {answer}")
            
        except asyncio.TimeoutError:
            print("\nError: Model response timed out after 30 seconds")
        except Exception as e:
            print(f"\nError getting answer: {e}")
            
    finally:
        if 'system' in locals():
            await system.stop()
            if 'system_task' in locals():
                system_task.cancel()
                try:
                    await system_task
                except asyncio.CancelledError:
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...") 