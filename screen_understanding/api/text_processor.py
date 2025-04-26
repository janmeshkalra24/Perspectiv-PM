import json
import asyncio
import redis
from typing import Dict, Any, List
import google.generativeai as genai
from datetime import datetime

class TextProcessor:
    def __init__(self, redis_url: str = "redis://localhost:6379", model_name: str = "gemini-pro"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        genai.configure(api_key="YOUR_API_KEY")  # Replace with actual API key
        self.model = genai.GenerativeModel(model_name)
        self.processing = False
        self.current_summary = ""
        
    async def start_processing(self, text: str, chunk_size: int, delay_interval: float) -> Dict[str, Any]:
        """Start processing text in chunks.
        
        Args:
            text: Full text to process
            chunk_size: Number of sentences per chunk
            delay_interval: Delay between chunks in seconds
            
        Returns:
            Status response
        """
        if self.processing:
            return {"status": "error", "message": "Already processing text"}
            
        self.processing = True
        
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Create chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = '. '.join(sentences[i:i + chunk_size]) + '.'
            chunks.append(chunk)
            
        # Start processing in background
        asyncio.create_task(self._process_chunks(chunks, delay_interval))
        
        return {"status": "success", "message": "Text processing started"}
        
    async def _process_chunks(self, chunks: List[str], delay_interval: float):
        """Process text chunks and stream to Redis.
        
        Args:
            chunks: List of text chunks
            delay_interval: Delay between chunks in seconds
        """
        try:
            for i, chunk in enumerate(chunks):
                if not self.processing:
                    break
                    
                # Process chunk with Gemini
                prompt = f"""Analyze this text chunk from a conversation or transcript:

{chunk}

Please provide:
1. A brief summary of the key points
2. Any important details or context
3. Any questions or uncertainties that need clarification

Format your response as a JSON object with these fields:
- summary: Brief summary of key points
- details: List of important details
- questions: List of questions/uncertainties
"""
                
                response = await self.model.generate_content(prompt)
                
                try:
                    result = json.loads(response.text)
                except json.JSONDecodeError:
                    result = {
                        "summary": response.text[:200],
                        "details": [],
                        "questions": []
                    }
                    
                # Add metadata
                result.update({
                    "chunk_index": i,
                    "timestamp": datetime.now().timestamp(),
                    "text": chunk
                })
                
                # Stream to Redis
                self.redis.xadd(
                    "text_stream",
                    {"data": json.dumps(result)}
                )
                
                # Update current summary
                self.current_summary = result["summary"]
                
                # Wait before next chunk
                await asyncio.sleep(delay_interval)
                
        except Exception as e:
            print(f"Error processing chunks: {e}")
        finally:
            self.processing = False
            
    def stop_processing(self) -> Dict[str, Any]:
        """Stop text processing.
        
        Returns:
            Status response
        """
        self.processing = False
        return {"status": "success", "message": "Text processing stopped"}
        
    def get_current_summary(self) -> Dict[str, Any]:
        """Get current text summary.
        
        Returns:
            Current summary
        """
        return {
            "status": "success",
            "summary": self.current_summary,
            "is_processing": self.processing
        } 