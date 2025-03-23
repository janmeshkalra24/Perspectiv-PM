#!/usr/bin/env python3

import argparse
import asyncio
import redis
from pathlib import Path

from screen_understanding import ScreenUnderstanding
from screen_understanding.api import APIClientFactory
from screen_understanding.sources import FrameBufferSource

class FrameQuerier:
    def __init__(self, redis_prefix: str = "demo:"):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.prefix = redis_prefix
        self.current_frame = 0
        
    def get_available_frames(self):
        """Get number of available frames."""
        keys = self.redis.keys(f"{self.prefix}*")
        return len(keys) // 2  # Divide by 2 because of metadata keys
        
    def get_frame(self, index: int):
        """Get frame data and metadata at index."""
        frame_key = f"{self.prefix}{index}"
        meta_key = f"{frame_key}:meta"
        
        frame_data = self.redis.get(frame_key)
        metadata = self.redis.get(meta_key)
        
        if frame_data and metadata:
            return frame_data, metadata.decode()
        return None, None

async def handle_queries(system: ScreenUnderstanding, querier: FrameQuerier):
    """Handle user queries about frames.
    
    Args:
        system: ScreenUnderstanding instance
        querier: FrameQuerier instance
    """
    print("\nCommands:")
    print("- i: Show current frame info")
    print("- q: Quit")
    print("\nOr type any question to ask about the current frame.")
    print("\nFrames will automatically update as they become available.")
    
    while True:
        try:
            command = input("\nCommand/Question: ").strip()
            
            if not command:
                continue
                
            if command.lower() == 'q':
                break
                
            if command.lower() == 'i':
                if hasattr(system.source, 'get_last_processed_frame_index'):
                    frame_idx = system.source.get_last_processed_frame_index()
                    frame_data = system.source._get_frame_data(frame_idx)
                    if frame_data:
                        print(f"\nCurrent frame info:")
                        print(f"Frame index: {frame_idx}")
                        print(f"Metadata: {frame_data['metadata']}")
                continue
            
            # Handle question
            print("\nProcessing question...")
            try:
                start_time = asyncio.get_event_loop().time()
                answer = await asyncio.wait_for(system.ask(command), timeout=30.0)
                latency = asyncio.get_event_loop().time() - start_time
                
                # Get current frame info
                frame_idx = system.source.get_last_processed_frame_index()
                frame_data = system.source._get_frame_data(frame_idx)
                
                print(f"\nAnswer: {answer}")
                print(f"Latency: {latency:.2f}s")
                if frame_data:
                    print(f"Frame info: {frame_data['metadata']}")
                
            except asyncio.TimeoutError:
                print("\nError: Model response timed out after 30 seconds")
            except Exception as e:
                print(f"\nError getting answer: {e}")
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

async def main():
    parser = argparse.ArgumentParser(description="Query frames and ask questions")
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="demo:",
        help="Redis key prefix for frames"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    source = FrameBufferSource(key_prefix=args.redis_prefix)
    querier = FrameQuerier(redis_prefix=args.redis_prefix)
    
    # Check for available frames
    frame_count = querier.get_available_frames()
    if frame_count == 0:
        print(f"No frames found with prefix '{args.redis_prefix}'")
        print("Please run upload_frames.py first")
        return
        
    print(f"Found {frame_count} frames with prefix '{args.redis_prefix}'")
    
    try:
        # Initialize model and system
        model = APIClientFactory.create("llava")
        system = ScreenUnderstanding(model=model, source=source)
        
        # Start system in background
        system_task = asyncio.create_task(system.start())
        
        # Handle queries
        await handle_queries(system, querier)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        # Cleanup
        if 'system' in locals():
            await system.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...") 