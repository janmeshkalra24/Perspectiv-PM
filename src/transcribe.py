"""
Module for converting audio/video files to text using OpenAI's Whisper model.
"""
import os
import whisper
import tempfile
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribe audio from an m4a or mp4 file using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio/video file (m4a or mp4)
        model_name (str): Name of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        str: Transcribed text
    """
    # Validate input file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    file_ext = Path(audio_path).suffix.lower()
    if file_ext not in ['.m4a', '.mp4']:
        raise ValueError(f"Unsupported file format: {file_ext}. Only m4a and mp4 files are supported.")
    
    # Load the Whisper model
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # If it's an mp4 file, extract audio first
        if file_ext == '.mp4':
            audio_path = _extract_audio_from_video(audio_path, temp_dir)
        
        # Transcribe the audio
        logger.info("Starting transcription...")
        result = model.transcribe(audio_path)
        
        return result["text"]

def _extract_audio_from_video(video_path, output_dir):
    """
    Extract audio from an mp4 file using ffmpeg.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted audio
        
    Returns:
        str: Path to the extracted audio file
    """
    output_path = os.path.join(output_dir, "extracted_audio.m4a")
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to process video files.")
    
    # Extract audio using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'aac',  # Use AAC codec
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio from video: {e.stderr.decode()}")

def main():
    """Command line interface for the transcription module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio from m4a or mp4 files")
    parser.add_argument("file_path", help="Path to the audio/video file (m4a or mp4)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: base)")
    parser.add_argument("--output", help="Path to save the transcription (optional)")
    
    args = parser.parse_args()
    
    try:
        # Transcribe the audio
        text = transcribe_audio(args.file_path, args.model)
        
        # Print or save the result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Transcription saved to: {args.output}")
        else:
            print("\nTranscription:")
            print("-" * 50)
            print(text)
            
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

if __name__ == "__main__":
    main() 