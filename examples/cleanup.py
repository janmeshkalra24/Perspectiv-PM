#!/usr/bin/env python3

import redis
import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def cleanup(redis_prefix: str = "test:", context_file: str = "data/frame_context.json"):
    """Clean up Redis and context data.
    
    Args:
        redis_prefix: Redis key prefix to clean
        context_file: Path to context file to remove
    """
    # Clean Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        keys = redis_client.keys(f"{redis_prefix}*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleaned up {len(keys)} Redis keys with prefix '{redis_prefix}'")
        else:
            logger.info(f"No Redis keys found with prefix '{redis_prefix}'")
    except Exception as e:
        logger.error(f"Error cleaning Redis: {e}")

    # Clean context file
    try:
        context_path = Path(context_file)
        if context_path.exists():
            context_path.unlink()
            logger.info(f"Removed context file: {context_file}")
        else:
            logger.info(f"Context file not found: {context_file}")
    except Exception as e:
        logger.error(f"Error removing context file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clean up Redis and context data")
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="test:",
        help="Redis key prefix to clean"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default="data/frame_context.json",
        help="Path to context file"
    )
    
    args = parser.parse_args()
    cleanup(args.redis_prefix, args.context_file)

if __name__ == "__main__":
    main() 