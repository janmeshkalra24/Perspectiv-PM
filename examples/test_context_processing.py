#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv
import redis
from tqdm import tqdm

from screen_understanding import ScreenUnderstanding
from screen_understanding.api import APIClientFactory
from screen_understanding.sources import FrameBufferSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextTester:
    """Tests context processing functionality."""
    
    def __init__(
        self,
        redis_prefix: str = "demo:",
        context_file: str = "data/frame_context.json",
        poll_interval: float = 5.0
    ):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.prefix = redis_prefix
        self.context_file = context_file
        self.poll_interval = poll_interval
        self.last_frame_count = 0
        self.last_context_size = 0
        self.last_context_data: Dict[str, Any] = {}
        self.processed_frames = set()
        self.start_time = time.time()
        
    def get_frame_count(self) -> int:
        """Get number of frames in Redis."""
        # Only count keys that contain frame data (not metadata)
        keys = [k.decode() for k in self.redis.keys(f"{self.prefix}*")]
        frame_keys = [k for k in keys if not k.endswith(":meta")]
        return len(frame_keys)
        
    def get_context_data(self) -> Dict[str, Any]:
        """Get full context data from file."""
        try:
            if Path(self.context_file).exists():
                with open(self.context_file, 'r') as f:
                    return json.load(f)
            return {"context": [], "timestamps": []}
        except Exception as e:
            logger.error(f"Error reading context file: {e}")
            return {"context": [], "timestamps": []}
            
    def get_new_context_entries(self, current_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get new context entries since last check."""
        current_entries = current_data.get("context", [])
        new_entries = []
        
        for entry in current_entries:
            frame_idx = entry.get("frame_index")
            if frame_idx is not None and frame_idx not in self.processed_frames:
                new_entries.append(entry)
                self.processed_frames.add(frame_idx)
                
        return new_entries
            
    async def monitor_processing(self, system: ScreenUnderstanding):
        """Monitor frame processing and context updates.
        
        Args:
            system: ScreenUnderstanding instance to monitor
        """
        logger.info("Starting context processing monitor...")
        logger.info(f"Watching Redis prefix: {self.prefix}")
        logger.info(f"Watching context file: {self.context_file}")
        
        # Create context file directory if it doesn't exist
        context_path = Path(self.context_file)
        context_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress bars with initial frame count
        initial_frame_count = self.get_frame_count()
        frame_pbar = tqdm(
            total=max(initial_frame_count, 1),  # Ensure at least 1 to avoid None
            desc="Frames processed",
            position=0,
            unit="frames"
        )
        context_pbar = tqdm(
            total=max(initial_frame_count, 1),
            desc="Context entries",
            position=1,
            unit="frames"
        )
        
        while True:
            try:
                # Get current counts and data
                frame_count = self.get_frame_count()
                context_data = self.get_context_data()
                context_size = len(context_data.get("context", []))
                
                # Check for new context entries
                new_entries = self.get_new_context_entries(context_data)
                
                # Update progress bar totals if needed
                if frame_count > frame_pbar.total:
                    frame_pbar.total = frame_count
                    context_pbar.total = frame_count
                
                # Update progress
                frame_pbar.n = frame_count
                context_pbar.n = context_size
                
                # Update descriptions
                frame_pbar.set_description(
                    f"Frames in Redis ({frame_count} total)"
                )
                context_pbar.set_description(
                    f"Frames processed ({context_size}/{frame_count})"
                )
                
                # Display processing rates
                elapsed = time.time() - self.start_time
                if context_size > 0:
                    process_rate = context_size / elapsed
                    frame_pbar.set_postfix({
                        "process_rate": f"{process_rate:.1f} fps",
                        "elapsed": f"{elapsed:.0f}s"
                    })
                
                # Log changes
                if frame_count > self.last_frame_count:
                    logger.info(f"New frames detected: {frame_count - self.last_frame_count}")
                    
                if new_entries:
                    logger.info(f"\nProcessed {len(new_entries)} new frames:")
                    for entry in new_entries:
                        frame_idx = entry.get("frame_index", "unknown")
                        timestamp = entry.get("metadata", {}).get("timestamp", 0)
                        description = entry.get("description", "No description")
                        
                        # Format timestamp as HH:MM:SS
                        time_str = str(datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")) if timestamp else "unknown"
                        
                        logger.info(f"\nFrame {frame_idx} at {time_str}:")
                        logger.info(f"Description: {description}")
                        
                        # Log any additional Gemini results
                        for key, value in entry.items():
                            if key not in ["frame_index", "metadata", "description"]:
                                logger.info(f"{key}: {value}")
                        
                        logger.info("-" * 80)
                    
                self.last_frame_count = frame_count
                self.last_context_size = context_size
                self.last_context_data = context_data
                
                # Check if context file exists and is writable
                if not context_path.exists():
                    logger.warning(f"Context file does not exist: {self.context_file}")
                elif not os.access(context_path, os.W_OK):
                    logger.warning(f"Context file is not writable: {self.context_file}")
                
                # Check if system is still processing
                if hasattr(system, '_background_task'):
                    if system._background_task and system._background_task.done():
                        exc = system._background_task.exception()
                        if exc:
                            logger.error(f"Background processing error: {exc}")
                            break
                        else:
                            # If task completed successfully and all frames are processed
                            if context_size >= frame_count:
                                logger.info("\nAll frames processed successfully!")
                                break
                
                # Update progress bars
                frame_pbar.refresh()
                context_pbar.refresh()
                
                await asyncio.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("\nMonitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
        
        # Clean up progress bars
        frame_pbar.close()
        context_pbar.close()

async def main():
    parser = argparse.ArgumentParser(description="Test context processing")
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="demo:",
        help="Redis key prefix for frames"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default="data/frame_context.json",
        help="Path to context file"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Interval between polling checks in seconds"
    )
    parser.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=30,
        help="Rate limit for Gemini API calls (requests per minute)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Ensure GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
        
    try:
        # Initialize components
        source = FrameBufferSource(key_prefix=args.redis_prefix)
        model = APIClientFactory.create(
            "gemini",
            model_name="models/gemini-2.0-flash-lite",
            rate_limit_rpm=args.rate_limit_rpm
        )
        
        system = ScreenUnderstanding(
            model=model,
            source=source,
            context_file=args.context_file
        )
        
        # Create tester
        tester = ContextTester(
            redis_prefix=args.redis_prefix,
            context_file=args.context_file,
            poll_interval=args.poll_interval
        )
        
        # Start system and monitoring
        logger.info("Starting ScreenUnderstanding system...")
        system_task = asyncio.create_task(system.start())
        
        try:
            await tester.monitor_processing(system)
        except Exception as e:
            logger.error(f"Error during testing: {e}")
        finally:
            # Clean up
            system_task.cancel()
            try:
                await system_task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...") 