from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

class BaseModel(ABC):
    """Base interface for all visual understanding models."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and extract relevant information.
        
        Args:
            frame: A numpy array representing the frame (HxWxC format)
            
        Returns:
            Dict containing extracted information from the frame
        """
        pass
    
    @abstractmethod
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """Answer a question based on the visual context.
        
        Args:
            question: The question to answer
            context: Context information from processed frames
            
        Returns:
            Answer to the question as a string
        """
        pass
    
    @abstractmethod
    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update the model's internal context.
        
        Args:
            new_context: New context information to incorporate
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up any resources used by the model."""
        pass 