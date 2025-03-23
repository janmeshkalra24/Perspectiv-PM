#!/usr/bin/env python3

import argparse
import asyncio
import cv2
import redis
from pathlib import Path
from tqdm import tqdm
import time

class VideoUploader:
    def __init__(self, redis_prefix: str = "demo:"):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.prefix = redis_prefix
        
    def cleanup_old_frames(self):
        """Clean up any existing frames."""
        keys = self.redis.keys(f"{self.prefix}*")
        if keys:
            self.redis.delete(*keys)
            print(f"Cleaned up {len(keys)} old frame keys")
    
    async def upload_video(self, video_path: str, frame_interval: float = 1.0):
        """Upload video frames to Redis.
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        frame_index = 0
        last_frame_time = 0
        
        print(f"\nVideo info:")
        print(f"- Duration: {duration:.1f}s")
        print(f"- FPS: {fps}")
        print(f"- Total frames: {total_frames}")
        print(f"- Frame interval: {frame_interval}s")
        
        with tqdm(total=int(duration), desc="Uploading frames") as pbar:
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
                    
                    # Store frame and metadata
                    frame_key = f"{self.prefix}{frame_index}"
                    meta_key = f"{frame_key}:meta"
                    current_second = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                    
                    metadata = {
                        "timestamp": current_second,
                        "frame_number": frame_index,
                        "total_frames": total_frames,
                        "duration": duration
                    }
                    
                    self.redis.set(frame_key, frame_bytes)
                    self.redis.set(meta_key, str(metadata))
                    
                    # Update progress
                    last_frame_time = current_time
                    frame_index += 1
                    pbar.update(frame_interval)
                    pbar.set_description(f"Time: {current_second:.1f}s / {duration:.1f}s")
                    
                    # Print frame info periodically
                    if frame_index % 10 == 0:
                        print(f"\nUploaded frame {frame_index} at {current_second:.1f}s")
                        print(f"Active frame keys: {len(self.redis.keys(f'{self.prefix}*'))//2}")
                        
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
        
    uploader = VideoUploader(redis_prefix=args.redis_prefix)
    
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