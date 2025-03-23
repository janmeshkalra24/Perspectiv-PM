import base64
import os
from typing import Any, Dict
from dotenv import load_dotenv
import logging

from ..base import APIConfig, BaseAPIClient

logger = logging.getLogger(__name__)

class LLaVAClient(BaseAPIClient):
    """Client for LLaVA model on Replicate."""

    def __init__(self):
        # Load environment variables from .env file
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            logger.info(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
        else:
            logger.warning(f"No .env file found at {env_path}")
        
        api_key = os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            logger.error("REPLICATE_API_TOKEN not found in environment variables")
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set. "
                "Please create a .env file with your token from https://replicate.com/account"
            )
        else:
            logger.info("Successfully loaded REPLICATE_API_TOKEN")
            
        config = APIConfig(
            base_url="https://api.replicate.com/v1",
            api_key=api_key
        )
        super().__init__(config)
        self.model_version = "yorickvp/llava-v1.5-7b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358"

    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process an image through LLaVA.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Processed results including image embeddings and initial analysis
        """
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode()
        image_data_url = f"data:image/jpeg;base64,{image_b64}"
        
        response = await self._make_request(
            method="POST",
            endpoint="/predictions",
            json_data={
                "version": self.model_version,
                "input": {
                    "image": image_data_url,
                    "prompt": "Analyze this image and describe what you see, focusing on any text, UI elements, or important information displayed.",
                }
            }
        )
        
        return {
            "analysis": response.get("output", ""),
            "prediction_id": response.get("id")
        }

    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Get answer to a question using LLaVA.
        
        Args:
            question: Question to answer
            context: Context information including image data
            
        Returns:
            Answer from LLaVA
        """
        if "image_data" not in context:
            raise ValueError("Image data required in context")
            
        image_b64 = base64.b64encode(context["image_data"]).decode()
        image_data_url = f"data:image/jpeg;base64,{image_b64}"
        
        response = await self._make_request(
            method="POST",
            endpoint="/predictions",
            json_data={
                "version": self.model_version,
                "input": {
                    "image": image_data_url,
                    "prompt": f"Based on this image, please answer the following question: {question}"
                }
            }
        )
        
        return response.get("output", "Sorry, I couldn't answer that question.") 