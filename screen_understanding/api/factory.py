from typing import Dict, Type

from .base import BaseAPIClient
from .clients.replicate import LLaVAClient
from .clients.huggingface import HuggingFaceClient
from .clients.gemini import GeminiClient


class APIClientFactory:
    """Factory for creating API clients."""
    
    _clients: Dict[str, Type[BaseAPIClient]] = {
        "llava": LLaVAClient,
        "huggingface": HuggingFaceClient,
        "gemini": GeminiClient
    }
    
    @classmethod
    def register_client(cls, name: str, client_class: Type[BaseAPIClient]) -> None:
        """Register a new API client.
        
        Args:
            name: Name of the client
            client_class: Client class to register
        """
        cls._clients[name] = client_class
    
    @classmethod
    def create(cls, client_type: str, **kwargs) -> BaseAPIClient:
        """Create an API client instance.
        
        Args:
            client_type: Type of client to create
            **kwargs: Additional configuration parameters to pass to the client
            
        Returns:
            API client instance
            
        Raises:
            ValueError: If client_type is not registered
        """
        if client_type not in cls._clients:
            raise ValueError(f"Unknown client type: {client_type}")
            
        return cls._clients[client_type](**kwargs) 