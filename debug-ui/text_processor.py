import asyncio
import json
import re
from typing import List, Dict, Optional
import google.generativeai as genai
from redis import Redis
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, redis_client: Redis, redis_prefix: str = "text:", model_name='gemini-2.0-flash-lite'):
        self.redis = redis_client
        self.prefix = redis_prefix
        self.is_processing = False
        self.current_text = ""
        self.chunk_size = 1
        self.delay_interval = 1.0
        self.summary = ""
        self._processing_task = None
        
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized TextProcessor with model: {model_name}")

    def split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of sentences."""
        # Split by periods, keeping the period
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    async def process_chunk(self, chunk: str) -> Optional[str]:
        """Process a single chunk of text and store the summary in Redis."""
        try:
            logger.info(f"Processing chunk of size {len(chunk)} characters")
            logger.info(f"Chunk preview: {chunk[:100]}...")
            
            # Get previous summary for context
            previous_summary = self.get_summary()
            context = f"Previous summary: {previous_summary}\n\n" if previous_summary else ""
            
            # Generate summary using Gemini with historical context
            prompt = f"{context}New text to summarize:\n\n{chunk}\n\nProvide a comprehensive summary that incorporates both the previous context and the new text."
            logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")
            
            # Remove await since generate_content is synchronous
            response = self.model.generate_content(prompt)
            summary = response.text
            logger.info(f"Received summary from Gemini: {summary}")
            
            return summary
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            return None

    async def stream_text(self, text: str, chunk_size: int, delay_interval: float):
        """Stream text chunks to Redis and update summary."""
        if self.is_processing:
            logger.warning("Text processing already in progress")
            return
            
        logger.info(f"Starting text streaming with chunk_size={chunk_size}, delay_interval={delay_interval}")
        self.is_processing = True
        self.current_text = text
        self.chunk_size = chunk_size
        self.delay_interval = delay_interval
        
        chunks = self.split_into_chunks(text, chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        chunk_index = 0
        
        try:
            for chunk in chunks:
                if not self.is_processing:
                    logger.info("Processing stopped by user")
                    break
                
                # Store chunk in Redis
                chunk_key = f"{self.prefix}chunk:{chunk_index}"
                self.redis.set(chunk_key, chunk)
                logger.info(f"Stored chunk {chunk_index} in Redis with key: {chunk_key}")
                
                # Process chunk and update summary
                chunk_summary = await self.process_chunk(chunk)
                if chunk_summary:
                    self.summary = chunk_summary
                    # Store summary in Redis
                    summary_key = f"{self.prefix}summary"
                    self.redis.set(summary_key, self.summary)
                    logger.info(f"Updated summary in Redis: {self.summary[:50]}...")
                    # Store current chunk index
                    self.redis.set(f"{self.prefix}current_chunk", chunk_index)
                    logger.info(f"Updated current chunk index to {chunk_index}")
                else:
                    logger.warning(f"No summary generated for chunk {chunk_index}")
                
                chunk_index += 1
                logger.info(f"Waiting {delay_interval} seconds before next chunk")
                await asyncio.sleep(delay_interval)
                
        except Exception as e:
            logger.error(f"Error in stream_text: {e}", exc_info=True)
        finally:
            logger.info("Text streaming completed")
            self.is_processing = False

    def stop_processing(self):
        """Stop text processing."""
        self.is_processing = False
        logger.info("Text processing stopped")

    def clear_data(self):
        """Clear all text data from Redis."""
        keys = self.redis.keys(f"{self.prefix}*")
        if keys:
            self.redis.delete(*keys)
        self.summary = ""
        self.current_text = ""
        self.is_processing = False
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None

    def get_processing_status(self) -> Dict:
        """Get current processing status."""
        return {
            "is_processing": self.is_processing,
            "current_text": self.current_text,
            "chunk_size": self.chunk_size,
            "delay_interval": self.delay_interval,
            "summary": self.summary
        }

    def store_summary(self, summary):
        """Store the summary in Redis with proper error handling."""
        try:
            summary_key = f"{self.prefix}summary"
            logger.info(f"Storing summary in Redis with key: {summary_key}")
            self.redis.set(summary_key, summary)
            
            # Verify storage
            stored_summary = self.redis.get(summary_key)
            if stored_summary:
                logger.info("Successfully verified summary storage in Redis")
                return True
            else:
                logger.error("Failed to verify summary storage in Redis")
                return False
        except Exception as e:
            logger.error(f"Error storing summary in Redis: {e}", exc_info=True)
            return False
            
    def get_summary(self):
        """Retrieve the summary from Redis with proper error handling."""
        try:
            summary_key = f"{self.prefix}summary"
            logger.info(f"Retrieving summary from Redis with key: {summary_key}")
            summary = self.redis.get(summary_key)
            
            if summary is None:
                logger.warning("No summary found in Redis")
                return ""
                
            if isinstance(summary, bytes):
                summary = summary.decode('utf-8')
                logger.info("Decoded summary from bytes to string")
                
            return summary
        except Exception as e:
            logger.error(f"Error retrieving summary from Redis: {e}", exc_info=True)
            return ""
            
    def clear_summary(self):
        """Clear the summary from Redis."""
        try:
            summary_key = f"{self.prefix}summary"
            logger.info(f"Clearing summary from Redis with key: {summary_key}")
            self.redis.delete(summary_key)
            logger.info("Successfully cleared summary from Redis")
            return True
        except Exception as e:
            logger.error(f"Error clearing summary from Redis: {e}", exc_info=True)
            return False 