import base64
import os
from typing import Any, Dict
from dotenv import load_dotenv
import logging
import json

from ..base import APIConfig, BaseAPIClient

logger = logging.getLogger(__name__)

class HuggingFaceClient(BaseAPIClient):
    """Client for HuggingFace Inference API."""

    def __init__(self):
        # Load environment variables from .env file
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            logger.info(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
        else:
            logger.warning(f"No .env file found at {env_path}")
        
        api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_key:
            logger.error("HUGGINGFACE_API_TOKEN not found in environment variables")
            raise ValueError(
                "HUGGINGFACE_API_TOKEN environment variable not set. "
                "Please create a .env file with your token from https://huggingface.co/settings/tokens"
            )
        else:
            logger.info("Successfully loaded HUGGINGFACE_API_TOKEN")
            
        config = APIConfig(
            base_url="https://api-inference.huggingface.co/models",
            api_key=api_key
        )
        super().__init__(config)
        # Using Microsoft's CogVLM which has good performance and is free to use
        self.model_name = "THUDM/cogvlm-chat-hf"

    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process an image through the model.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Processed results including image analysis
        """
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode()
        
        response = await self._make_request(
            method="POST",
            endpoint=f"/{self.model_name}",
            json_data={
                "inputs": {
                    "image": image_b64,
                    "text": "Analyze this image and describe what you see, focusing on any text, UI elements, or important information displayed."
                }
            }
        )
        
        return {
            "analysis": response[0]["generated_text"] if response else "",
            "prediction_id": None  # HF doesn't provide prediction IDs
        }

    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Get answer to a question using the model.
        
        Args:
            question: Question to answer
            context: Context information including image data
            
        Returns:
            Answer from the model
        """
        if "image_data" not in context:
            raise ValueError("Image data required in context")
            
        image_b64 = base64.b64encode(context["image_data"]).decode()
        
        response = await self._make_request(
            method="POST",
            endpoint=f"/{self.model_name}",
            json_data={
                "inputs": {
                    "image": image_b64,
                    "text": question
                }
            }
        )
        
        return response[0]["generated_text"] if response else "Sorry, I couldn't answer that question." 