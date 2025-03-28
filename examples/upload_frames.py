#!/usr/bin/env python3

import argparse
import asyncio
import cv2
import redis
from pathlib import Path
from tqdm import tqdm
import time

class VideoUploader:
    def __init__(self, redis_prefix: str = "demo:", delay_interval: float = 0.0):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.prefix = redis_prefix
        self.delay_interval = delay_interval
        # Always clean up on initialization
        self.cleanup_old_frames()
        
    def cleanup_old_frames(self):
        """Clean up any existing frames and their metadata."""
        # Get all keys matching the prefix pattern
        frame_keys = self.redis.keys(f"{self.prefix}*")
        meta_keys = self.redis.keys(f"{self.prefix}*:meta")
        
        # Combine all keys to delete
        all_keys = frame_keys + meta_keys
        
        if all_keys:
            self.redis.delete(*all_keys)
            print(f"Cleaned up {len(frame_keys)} frame keys and {len(meta_keys)} metadata keys")
        else:
            print("No existing frames to clean up")
    
    async def upload_video(self, video_path: str, frame_interval: float = 1.0):
        """Upload video frames to Redis.
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames in seconds
            delay_interval: Delay between frame uploads in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        frame_index = 0
        
        # Calculate expected number of frames based on interval
        expected_frames = int(duration / frame_interval)
        
        print(f"\nVideo info:")
        print(f"- Duration: {duration:.1f}s")
        print(f"- FPS: {fps}")
        print(f"- Total frames in video: {total_frames}")
        print(f"- Frame interval: {frame_interval}s")
        print(f"- Expected frames to upload: {expected_frames}")
        
        with tqdm(total=expected_frames, desc="Uploading frames") as pbar:
            try:
                # Calculate frame positions to capture
                frame_positions = []
                current_time = 0
                while current_time < duration:
                    frame_pos = int(current_time * fps)
                    frame_positions.append(frame_pos)
                    current_time += frame_interval
                
                # Capture and upload frames at calculated positions
                for frame_pos in frame_positions:
                    # Seek to desired frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert frame to bytes
                    _, buffer = cv2.imencode(".jpg", frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Store frame and metadata
                    frame_key = f"{self.prefix}{frame_index}"
                    meta_key = f"{frame_key}:meta"
                    current_second = frame_pos / fps
                    
                    metadata = {
                        "timestamp": current_second,
                        "frame_number": frame_index,
                        "total_frames": expected_frames,
                        "duration": duration,
                        "fps": fps
                    }
                    
                    self.redis.set(frame_key, frame_bytes)
                    self.redis.set(meta_key, str(metadata))
                    
                    # Update progress
                    frame_index += 1
                    pbar.update(1)
                    pbar.set_description(f"Frame {frame_index}/{expected_frames}")
                    
                    # Print frame info periodically
                    if frame_index % 10 == 0:
                        print(f"\nUploaded frame {frame_index} at {current_second:.1f}s")
                        print(f"Active frame keys: {len(self.redis.keys(f'{self.prefix}*'))//2}")
                    
                    # Add delay between frames if specified
                    if self.delay_interval > 0:
                        await asyncio.sleep(self.delay_interval)
                        
            except KeyboardInterrupt:
                print("\nUpload interrupted by user")
            finally:
                cap.release()
                
        print(f"\nUpload complete. Total frames uploaded: {frame_index}")
        print(f"Use prefix '{self.prefix}' to query frames")

async def main():
    parser = argparse.ArgumentParser(description="Upload video frames to Redis")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="Interval between frames in seconds"
    )
    parser.add_argument(
        "--delay-interval",
        type=float,
        default=0.0,
        help="Delay between frame uploads in seconds"
    )
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="demo:",
        help="Redis key prefix for frames"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up existing frames before upload"
    )
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
        
    uploader = VideoUploader(redis_prefix=args.redis_prefix, delay_interval=args.delay_interval)
    
    if args.clean:
        uploader.cleanup_old_frames()
    
    try:
        await uploader.upload_video(str(video_path), args.frame_interval)
    except Exception as e:
        print(f"\nError during upload: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...") 