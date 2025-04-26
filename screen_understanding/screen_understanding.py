import json
from typing import Dict, Any, Optional

class ScreenUnderstanding:
    async def process_new_data(self):
        """Process new frames and text chunks from source."""
        try:
            # Get new data from source
            new_data = self.source.get_new_data()
            frames = new_data["frames"]
            text_chunks = new_data["text_chunks"]
            
            if not frames and not text_chunks:
                return
                
            # Process frames
            for frame in frames:
                # Process frame with Gemini
                frame_result = await self.process_frame(frame)
                if frame_result:
                    self.context.append(frame_result)
                    
            # Process text chunks
            for chunk in text_chunks:
                # Process text chunk with Gemini
                chunk_result = await this.process_text_chunk(chunk)
                if chunk_result:
                    self.context.append(chunk_result)
                    
            # Save context to file
            await this.save_context()
            
        except Exception as e:
            logger.error(f"Error processing new data: {e}")
            raise
            
    async def process_text_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a text chunk with Gemini.
        
        Args:
            chunk: Text chunk data
            
        Returns:
            Processed chunk data or None if processing failed
        """
        try:
            # Prepare prompt for text chunk
            prompt = f"""Analyze this text chunk from a conversation or transcript:

{chunk['text']}

Please provide:
1. A brief summary of the key points
2. Any important details or context
3. Any questions or uncertainties that need clarification

Format your response as a JSON object with these fields:
- summary: Brief summary of key points
- details: List of important details
- questions: List of questions/uncertainties
"""
            
            # Get response from Gemini
            response = await self.model.generate_content(prompt)
            
            # Parse response
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # If response is not valid JSON, create a basic structure
                result = {
                    "summary": response.text[:200],  # First 200 chars as summary
                    "details": [],
                    "questions": []
                }
                
            # Add chunk metadata to result
            result.update({
                "chunk_index": chunk["metadata"]["chunk_index"],
                "timestamp": chunk["metadata"]["timestamp"],
                "text": chunk["text"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text chunk: {e}")
            return None 