from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional, List
from .text_processor import TextProcessor
from ..core import ScreenUnderstanding

app = FastAPI()
text_processor = TextProcessor()
router = APIRouter()

class TextProcessingRequest(BaseModel):
    text: str
    chunk_size: int
    delay_interval: float

class TextSummaryRequest(BaseModel):
    text: str
    context: Optional[List[str]] = None

class TextSummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str

@app.post("/process_text")
async def process_text(request: TextProcessingRequest):
    """Start processing text in chunks."""
    try:
        result = await text_processor.start_processing(
            request.text,
            request.chunk_size,
            request.delay_interval
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_processing")
async def stop_processing():
    """Stop text processing."""
    try:
        result = text_processor.stop_processing()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/text_summary")
async def get_text_summary():
    """Get current text summary."""
    try:
        result = text_processor.get_current_summary()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text_summary", response_model=TextSummaryResponse)
async def process_text(request: TextSummaryRequest):
    try:
        # Initialize the screen understanding system
        system = ScreenUnderstanding()
        
        # Process the text chunk
        result = await system.process_text_chunk(request.text, request.context)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to process text")
            
        return TextSummaryResponse(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            sentiment=result.get("sentiment", "neutral")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 