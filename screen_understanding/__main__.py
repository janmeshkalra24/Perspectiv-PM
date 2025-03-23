import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from .api import APIClientFactory
from .core import ScreenUnderstanding
from .sources import FrameBufferSource

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.getenv("LOG_FILE", "screen_understanding.log")
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point for the screen understanding system."""
    try:
        # Initialize components
        model_type = os.getenv("DEFAULT_MODEL_TYPE", "llava")
        model = APIClientFactory.create(model_type)
        source = FrameBufferSource()

        # Create and start system
        system = ScreenUnderstanding(
            model=model,
            source=source,
            context_retention_period=int(os.getenv("CONTEXT_RETENTION_PERIOD", "3600"))
        )

        logger.info(f"Starting screen understanding system with {model_type} model")
        await system.start()

    except Exception as e:
        logger.error(f"Error in screen understanding system: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main()) 