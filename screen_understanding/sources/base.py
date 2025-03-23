from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DataSource(ABC):
    """Base interface for video frame data sources."""

    @abstractmethod
    async def get_frame(self) -> Optional[bytes]:
        """Get the next frame from the source.
        
        Returns:
            Frame data as bytes, or None if no frame is available
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the source."""
        pass 