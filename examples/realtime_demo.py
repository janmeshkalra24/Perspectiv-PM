#!/usr/bin/env python3

import argparse
import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional
from tqdm import tqdm

import cv2
import numpy as np
import redis

from screen_understanding import ScreenUnderstanding
from screen_understanding.api import APIClientFactory
from screen_understanding.sources import FrameBufferSource

class VideoStreamController:
    """Controls video streaming and frame synchronization."""
    
    def __init__(self, source: FrameBufferSource):
        self.source = source
        self.current_frame_index = 0
        self.is_streaming = False
        self.frame_available = asyncio.Event()
    
    def get_current_frame_key(self) -> str:
        """Get the current frame key."""
        return f"{self.source.key_prefix}{self.current_frame_index}"
    
    async def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """Wait for a new frame to become available."""
        try:
            await asyncio.wait_for(self.frame_available.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

async def process_video(
    video_path: str,
    frame_interval: float = 1.0,
    progress_bar = None
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """Process video file and yield frames with metadata.
    
    Args:
        video_path: Path to video file
        frame_interval: Interval between frames in seconds
        progress_bar: tqdm progress bar instance
    """
    cap = cv2.VideoCapture(video_path)
    last_frame_time = 0
    
    # Get video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    current_second = 0
    
    if progress_bar is not None:
        progress_bar.reset(total=int(duration))
        progress_bar.set_description(f"Video duration: {duration:.1f}s")
    
    try:
        while cap.isOpened():
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                await asyncio.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to bytes
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            
            last_frame_time = current_time
            current_second = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            
            if progress_bar is not None:
                progress_bar.update(frame_interval)
                progress_bar.set_description(f"Time: {current_second:.1f}s / {duration:.1f}s")
            
            metadata = {
                "timestamp": current_second,
                "frame_number": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "total_frames": total_frames,
                "duration": duration
            }
            
            yield frame_bytes, metadata
            
    finally:
        cap.release()

async def stream_to_redis(
    controller: VideoStreamController,
    video_path: str,
    progress_bar = None
):
    """Stream video frames to Redis with synchronization.
    
    Args:
        controller: VideoStreamController instance
        video_path: Path to video file
        progress_bar: tqdm progress bar instance
    """
    source = controller.source
    
    try:
        # Clear any existing frames
        keys = source.redis.keys(f"{source.key_prefix}*")
        if keys:
            source.redis.delete(*keys)
        
        print(f"Streaming frames to Redis with prefix: {source.key_prefix}")
        controller.is_streaming = True
        
        async for frame, metadata in process_video(video_path, progress_bar=progress_bar):
            try:
                # Store frame
                key = f"{source.key_prefix}{controller.current_frame_index}"
                source.redis.set(key, frame)
                
                # Store metadata
                meta_key = f"{key}:meta"
                source.redis.set(meta_key, str(metadata))
                
                # Cleanup old frames (keep last 10)
                if controller.current_frame_index > 10:
                    old_key = f"{source.key_prefix}{controller.current_frame_index-11}"
                    source.redis.delete(old_key)
                    source.redis.delete(f"{old_key}:meta")
                
                # Signal new frame is available
                controller.frame_available.set()
                controller.frame_available.clear()
                controller.current_frame_index += 1
                
                # Periodically verify frames and report status
                if controller.current_frame_index % 10 == 0:
                    keys = source.redis.keys(f"{source.key_prefix}*")
                    print(f"\nActive frames: {len(keys)//2}")  # Divide by 2 because of metadata keys
                    
            except redis.RedisError as e:
                print(f"\nRedis error while processing frame {controller.current_frame_index}: {str(e)}")
                continue
            except Exception as e:
                print(f"\nUnexpected error while processing frame {controller.current_frame_index}: {str(e)}")
                continue
                
    except asyncio.CancelledError:
        print("\nVideo streaming cancelled")
        raise
    except Exception as e:
        print(f"\nFatal error in video streaming: {str(e)}")
        raise
    finally:
        controller.is_streaming = False
        if progress_bar is not None:
            progress_bar.close()

async def handle_user_input(
    system: ScreenUnderstanding,
    controller: VideoStreamController
) -> None:
    """Handle user input for queries with frame synchronization.
    
    Args:
        system: ScreenUnderstanding instance
        controller: VideoStreamController instance
    """
    print("\nEnter your questions (type 'q' to quit):")
    print("Waiting for initial frames...")
    
    # Wait for initial frames with timeout
    wait_start = time.time()
    max_wait_time = 60  # Wait up to 60 seconds for frames
    
    while True:
        # Check if any frames exist in Redis
        keys = controller.source.redis.keys(f"{controller.source.key_prefix}*")
        if len(keys) > 0:
            print(f"Found {len(keys)//2} frames in Redis")  # Divide by 2 because of metadata keys
            break
            
        if time.time() - wait_start > max_wait_time:
            print(f"Timeout waiting for frames after {max_wait_time} seconds")
            return
            
        print("Waiting for frames... (press Ctrl+C to exit)")
        await asyncio.sleep(2)  # Check every 2 seconds
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ('q', 'quit', 'exit'):
                break
                
            if not question:
                continue
            
            # Check frame availability again
            keys = controller.source.redis.keys(f"{controller.source.key_prefix}*")
            if len(keys) == 0:
                print("No frames available - video stream may have ended")
                continue
                
            start_time = time.time()
            try:
                answer = await asyncio.wait_for(system.ask(question), timeout=30.0)
                latency = time.time() - start_time
                
                # Get metadata for current frame
                meta_key = f"{controller.get_current_frame_key()}:meta"
                metadata = controller.source.redis.get(meta_key)
                
                print(f"\nAnswer: {answer}")
                print(f"Latency: {latency:.2f}s")
                if metadata:
                    print(f"Frame info: {metadata.decode()}")
            except asyncio.TimeoutError:
                print("\nError: Model response timed out after 30 seconds")
            except Exception as e:
                print(f"\nError getting answer: {str(e)}")
                if hasattr(system.source, 'redis'):
                    try:
                        keys = system.source.redis.keys(f"{system.source.key_prefix}*")
                        print(f"Available frame keys: {keys}")
                    except Exception:
                        pass
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting question loop...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            break

async def main():
    parser = argparse.ArgumentParser(description="Screen Understanding Demo")
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="demo:",
        help="Redis key prefix for frames (must match upload_frames.py)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["llava", "huggingface", "gemini"],
        help="Model to use for processing"
    )
    parser.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=None,
        help="Rate limit in requests per minute (default: use value from GEMINI_RATE_LIMIT_RPM env var)"
    )
    
    args = parser.parse_args()
    
    # Set up rate limiting from environment or CLI args
    rate_limit_rpm = args.rate_limit_rpm
    if rate_limit_rpm is None:
        rate_limit_rpm = int(os.getenv("GEMINI_RATE_LIMIT_RPM", "30"))  # Default to 30 RPM for Flash-Lite
    
    print(f"\nUsing rate limit of {rate_limit_rpm} requests per minute")
    
    # Set up Redis connection
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    try:
        redis_client.ping()
    except redis.ConnectionError:
        print("Error: Could not connect to Redis. Make sure Redis server is running.")
        return
    
    try:
        # Initialize components
        source = FrameBufferSource(redis_client, args.redis_prefix)
        controller = VideoStreamController(source)
        
        # Initialize model with rate limiting configuration
        model = APIClientFactory.create(
            args.model,
            model_name="models/gemini-2.0-flash-lite",
            rate_limit_rpm=rate_limit_rpm
        )
        
        system = ScreenUnderstanding(model=model, source=source)
        
        # Start system task
        system_task = asyncio.create_task(system.start())
        
        print(f"\nConnected to Redis, waiting for frames with prefix: {args.redis_prefix}")
        print("Make sure to run upload_frames.py in another terminal to stream frames")
        
        # Handle user input
        try:
            await handle_user_input(system, controller)
        except Exception as e:
            print(f"\nError in user input handling: {str(e)}")
        finally:
            # Clean up tasks
            system_task.cancel()
            try:
                await system_task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...") 