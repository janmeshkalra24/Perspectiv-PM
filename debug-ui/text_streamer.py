#!/usr/bin/env python3

import asyncio
import json
import logging
import re
from typing import List, Dict, Any
import redis

logger = logging.getLogger(__name__)

class TextStreamer:
    """Handles streaming text chunks to Redis."""
    
    def __init__(
        self,
        redis_prefix: str = "test:",
        delay_interval: float = 1.0
    ):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.prefix = redis_prefix
        self.delay_interval = delay_interval
        self.is_streaming = False
        self.stream_task = None
        
    def split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of sentences.
        
        Args:
            text: Text to split
            chunk_size: Number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences (handling common abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                
        # Add any remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    async def stream_text(self, text: str, chunk_size: int):
        """Stream text chunks to Redis.
        
        Args:
            text: Text to stream
            chunk_size: Number of sentences per chunk
        """
        chunks = self.split_into_chunks(text, chunk_size)
        chunk_index = 0
        
        logger.info(f"Starting text streaming with {len(chunks)} chunks")
        
        try:
            while self.is_streaming and chunk_index < len(chunks):
                chunk = chunks[chunk_index]
                
                # Store chunk in Redis
                chunk_key = f"{self.prefix}text:{chunk_index}"
                meta_key = f"{chunk_key}:meta"
                
                metadata = {
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "timestamp": chunk_index * self.delay_interval
                }
                
                self.redis.set(chunk_key, chunk)
                self.redis.set(meta_key, json.dumps(metadata))
                
                logger.info(f"Streamed chunk {chunk_index + 1}/{len(chunks)}")
                
                chunk_index += 1
                await asyncio.sleep(self.delay_interval)
                
        except Exception as e:
            logger.error(f"Error during text streaming: {e}")
            raise
            
        logger.info("Text streaming completed")
        
    def start_streaming(self, text: str, chunk_size: int):
        """Start text streaming in background.
        
        Args:
            text: Text to stream
            chunk_size: Number of sentences per chunk
        """
        if self.is_streaming:
            raise RuntimeError("Text streaming is already in progress")
            
        self.is_streaming = True
        self.stream_task = asyncio.create_task(
            self.stream_text(text, chunk_size)
        )
        
    def stop_streaming(self):
        """Stop text streaming."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        if self.stream_task:
            self.stream_task.cancel()
            self.stream_task = None
            
    def cleanup(self):
        """Clean up Redis keys."""
        keys = self.redis.keys(f"{self.prefix}text:*")
        if keys:
            self.redis.delete(*keys) 