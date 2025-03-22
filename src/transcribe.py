"""
Module for converting audio/video files to text using OpenAI's Whisper model.
"""
import whisper
import argparse

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribe audio from an audio file using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        str: Transcribed text
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]

def main():
    """Command line interface for the transcription module."""
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper.")
    parser.add_argument("audio_file", help="Path to the audio file (MP3, WAV, etc.)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: base)")
    parser.add_argument("--output_file", default="transcription.txt", 
                      help="File to save the transcription (default: transcription.txt)")
    
    args = parser.parse_args()
    
    transcript = transcribe_audio(args.audio_file, args.model)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcription saved to {args.output_file}")

if __name__ == "__main__":
    main() 