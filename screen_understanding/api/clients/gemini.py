import base64
import os
from typing import Any, Dict
from dotenv import load_dotenv
import logging
import google.generativeai as genai
import httpx
import time
import asyncio

from ..base import APIConfig, BaseAPIClient

logger = logging.getLogger(__name__)

class GeminiClient(BaseAPIClient):
    """Client for Google's Gemini Pro Vision API."""

    def __init__(self, model_name: str = 'models/gemini-2.0-flash', rate_limit_rpm: int = None):
        # Load environment variables from .env file
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            logger.info(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
        else:
            logger.warning(f"No .env file found at {env_path}")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please create a .env file with your API key from https://makersuite.google.com/app/apikey"
            )
        else:
            logger.info("Successfully loaded GOOGLE_API_KEY")
            
        # Initialize base client for cleanup purposes
        config = APIConfig(
            base_url="https://generativelanguage.googleapis.com",
            api_key=api_key
        )
        super().__init__(config)
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure rate limiting if specified
        if rate_limit_rpm:
            # Convert RPM to delay in seconds between requests
            self.request_delay = 60.0 / rate_limit_rpm
            self.last_request_time = 0
        else:
            self.request_delay = None
            
        logger.info(f"Initialized {model_name} model with rate limit {rate_limit_rpm if rate_limit_rpm else 'disabled'} RPM")

    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process an image through Gemini Vision.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict containing model's understanding of the image
        """
        try:
            # Apply rate limiting if configured
            if self.request_delay:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_delay:
                    await asyncio.sleep(self.request_delay - time_since_last)
                self.last_request_time = time.time()
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Process with Gemini
            response = await self.model.generate_content_async(
                [
                    "Analyze this screenshot or image in detail. "
                    "Focus on describing the key elements, text, and layout you see.",
                    {"mime_type": "image/jpeg", "data": image_b64}
                ],
                stream=False
            )
            
            # Extract and return the response
            if response and response.text:
                return {
                    "description": response.text,
                    "confidence": 1.0  # Gemini doesn't provide confidence scores
                }
            else:
                raise ValueError("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            raise

    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Get answer to a question using Gemini Vision.
        
        Args:
            question: Question to answer
            context: Context information including image data
            
        Returns:
            Answer from Gemini
        """
        if "image_data" not in context:
            raise ValueError("Image data required in context")
            
        try:
            # Create image parts for Gemini
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(context["image_data"]).decode()
                }
            ]
            
            # Generate answer with enhanced context
            enhanced_prompt = f"""Please analyze this image and answer the following question:
{question}

Focus on providing specific details about UI elements, text content, and visual information relevant to the question."""
            
            response = await self.model.generate_content_async([enhanced_prompt, image_parts[0]])
            return response.text if response else "Sorry, I couldn't answer that question."
        except Exception as e:
            logger.error(f"Error answering question with Gemini: {str(e)}")
            raise 