import os
import time
import subprocess
from datetime import datetime
import random
def ensure_directory_exists(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def convert_pcm_to_wav(pcm_file, wav_file):
    """Convert PCM file to WAV with specified parameters"""
    ensure_directory_exists(wav_file)
    
    cmd = [
        'ffmpeg',
        '-f', 's16le',
        '-ar', '32000',
        '-ac', '1',
        '-i', pcm_file,
        '-ar', '16000',
        '-ac', '1',
        wav_file
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[{datetime.now()}] Successfully converted {pcm_file} to {wav_file}")
        return wav_file 
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error converting file: {e}")
        return None

def wait_for_file(file_path, check_interval=1):
    """Wait for a file to exist"""
    print(f"Waiting for file to exist: {file_path}")
    while not os.path.exists(file_path):
        time.sleep(check_interval)
    print(f"File found: {file_path}")

def get_largest_pcm_file(directory):
    """Find the largest PCM file in the given directory, waiting if necessary"""
    print(f"Waiting for PCM files in {directory}...")
    while True:
        pcm_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pcm")]
        if pcm_files:
            return max(pcm_files, key=os.path.getsize)
        time.sleep(2)  # Check again after 2 seconds

def monitor_and_convert(pcm_file, output_dir, base_name, extension=".wav"):
    """
    Monitor PCM file and convert to WAV periodically.
    Writes to a new WAV file each time by appending a timestamp.
    """
    wait_for_file(pcm_file)
    
    print(f"Monitoring {pcm_file} for changes...")
    last_size = 0
    
    while True:
        try:
            current_size = os.path.getsize(pcm_file)
            if current_size != last_size:
                print(f"[{datetime.now()}] File size changed from {last_size} to {current_size} bytes")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_file = os.path.join(output_dir, f"{base_name}_{timestamp}{extension}")
                wav_file = convert_pcm_to_wav(pcm_file, wav_file)
                if wav_file: 
                    transcribe_to_text(wav_file)
                last_size = current_size
            interval = 5
            print(f"[{datetime.now()}] Waiting for {interval} seconds before next check...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping conversion...")
            break
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = random.randint(1000, 9999)
    output_dir = os.path.expanduser(f"~/Desktop/zoom_audio_output_{timestamp}_{random_id}")
    os.makedirs(output_dir, exist_ok=True)
    pcm_directory = "/home/jkalra/Desktop/Perspectiv-PM/zoom-meeting-bot/sample_program/out"
    PCM_FILE = get_largest_pcm_file(pcm_directory)
    BASE_NAME = os.path.splitext(os.path.basename(PCM_FILE))[0]
    
    monitor_and_convert(PCM_FILE, output_dir, BASE_NAME)