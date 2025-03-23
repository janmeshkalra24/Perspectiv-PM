import os
from typing import Optional, Dict, Any
import redis
from redis.connection import ConnectionPool
import logging
import json
import asyncio

from .base import DataSource

logger = logging.getLogger(__name__)

class FrameBufferSource(DataSource):
    """Redis-based frame buffer data source."""

    def __init__(
        self,
        redis_client: redis.Redis = None,
        key_prefix: str = "frame_buffer:",
        host: str = None,
        port: int = None,
        db: int = None
    ):
        """Initialize the frame buffer source.
        
        Args:
            redis_client: Existing Redis client (optional)
            key_prefix: Prefix for Redis keys
            host: Redis host (defaults to env var REDIS_HOST)
            port: Redis port (defaults to env var REDIS_PORT)
            db: Redis database (defaults to env var REDIS_DB)
        """
        if redis_client:
            self.redis = redis_client
        else:
            self.host = host or os.getenv("REDIS_HOST", "localhost")
            self.port = port or int(os.getenv("REDIS_PORT", "6379"))
            self.db = db or int(os.getenv("REDIS_DB", "0"))
            
            # Use connection pooling
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False  # We want raw bytes for images
            )
            self.redis = redis.Redis(connection_pool=self.pool)
        
        self.key_prefix = key_prefix
        self._current_index = 0
        self._last_processed_index = -1
        
        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _get_frame_data(self, index: int) -> Optional[Dict[str, Any]]:
        """Get frame data and metadata at index.
        
        Args:
            index: Frame index to get
            
        Returns:
            Dict containing frame data and metadata, or None if not found
        """
        frame_key = f"{self.key_prefix}{index}"
        meta_key = f"{frame_key}:meta"
        
        frame_data = self.redis.get(frame_key)
        metadata_str = self.redis.get(meta_key)
        
        if frame_data and metadata_str:
            try:
                metadata = eval(metadata_str.decode())  # Safe since we control the metadata format
                return {
                    "image_data": frame_data,
                    "metadata": metadata,
                    "frame_index": index
                }
            except Exception as e:
                logger.error(f"Error parsing metadata for frame {index}: {e}")
        
        return None

    async def get_frame(self) -> Optional[Dict[str, Any]]:
        """Get the next frame from the buffer.
        
        Returns:
            Dict containing frame data and metadata, or None if no frame is available
        """
        try:
            # Check for new frames
            keys = self.redis.keys(f"{self.key_prefix}*")
            available_frames = len(keys) // 2  # Divide by 2 because of metadata keys
            
            # If we've processed all available frames, wait for new ones
            if self._current_index >= available_frames:
                await asyncio.sleep(0.1)  # Don't hammer Redis
                return None
            
            # Get frame data
            frame_data = self._get_frame_data(self._current_index)
            if frame_data:
                self._last_processed_index = self._current_index
                self._current_index += 1
                return frame_data
                
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error while getting frame {self._current_index}: {e}")
            return None

    async def cleanup(self) -> None:
        """Clean up Redis connection."""
        try:
            # Clear frames with this prefix
            pattern = f"{self.key_prefix}*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleaned up {len(keys)} frame keys")
                
            # Close connections
            self.redis.close()
            self.pool.disconnect()
            logger.info("Redis connections closed")
            
        except redis.RedisError as e:
            logger.error(f"Error during cleanup: {e}")

    def get_current_frame_index(self) -> int:
        """Get the current frame index."""
        return self._current_index

    def get_last_processed_frame_index(self) -> int:
        """Get the last processed frame index."""
        return self._last_processed_index 