from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class APIConfig(BaseModel):
    """Configuration for API clients."""
    api_key: Optional[str] = None
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3


class BaseAPIClient(ABC):
    """Base class for API clients."""

    def __init__(self, config: APIConfig):
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=self._get_headers()
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            json_data: Request payload
            
        Returns:
            API response as dictionary
        """
        try:
            response = self.client.request(
                method=method,
                url=endpoint,
                json=json_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            # Log error details here
            raise

    @abstractmethod
    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process an image through the API.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Processed results from the API
        """
        pass

    @abstractmethod
    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Get answer to a question through the API.
        
        Args:
            question: Question to answer
            context: Context information
            
        Returns:
            Answer from the API
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        self.client.close() 