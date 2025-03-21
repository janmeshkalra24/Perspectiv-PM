import whisper
import argparse

def transcribe_audio(audio_path, model_name="base", output_file="transcription.txt"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper.")
    parser.add_argument("--audio_file", type=str, help="Path to the audio file (MP3, WAV, etc.)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (default: base)")
    parser.add_argument("--output_file", type=str, default="transcription.txt", help="File to save the transcription (default: transcription.txt)")
    args = parser.parse_args()
    transcribe_audio(args.audio_file, args.model, args.output_file)
