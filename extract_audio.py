import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def extract_audio(video_path, output_path="extracted_audio.wav", timeout=300):
    """Extract audio from video file using FFmpeg with enhanced error handling.
    
    Args:
        video_path: Path to the input video file
        output_path: Path where the extracted audio will be saved
        timeout: Maximum time (in seconds) to wait for the extraction to complete
        
    Returns:
        str: Path to the extracted audio file if successful, None otherwise
    """
    start_time = time.time()
    
    try:
        # Check if FFmpeg is installed
        if not check_ffmpeg_installed():
            logging.error("FFmpeg is not installed or not in system PATH")
            return None
            
        # Validate input file
        if not os.path.exists(video_path):
            logging.error(f"Input video file not found: {video_path}")
            return None
            
        if os.path.getsize(video_path) == 0:
            logging.error(f"Input video file is empty: {video_path}")
            return None
            
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) or '.'
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
            return None
            
        # Clean up any existing output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logging.warning(f"Could not remove existing output file: {str(e)}"
                              " - Will attempt to overwrite.")
        
        # Build FFmpeg command with error handling
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',  # Overwrite output file without asking
            '-i', str(video_path),
            '-vn',  # Disable video recording
            '-acodec', 'pcm_s16le',  # Use 16-bit PCM encoding
            '-ar', '44100',  # Set audio sample rate to 44.1kHz
            '-ac', '1',  # Convert to mono
            '-af', 'aformat=sample_fmts=s16',  # Ensure 16-bit output
            str(output_path)
        ]
        
        logging.info(f"Extracting audio from {video_path}...")
        
        # Run FFmpeg with timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Check if the output file was created and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = time.time() - start_time
                logging.info(f"Successfully extracted audio to {output_path} "
                           f"(took {duration:.2f} seconds)")
                return output_path
            else:
                error_msg = (
                    f"Audio extraction failed - no output file created or file is empty. "
                    f"FFmpeg stderr: {result.stderr}"
                )
                logging.error(error_msg)
                return None
                
        except subprocess.TimeoutExpired:
            logging.error(f"Audio extraction timed out after {timeout} seconds")
            return None
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error (return code {e.returncode}): {e.stderr}")
            return None
            
    except Exception as e:
        logging.error(f"Unexpected error during audio extraction: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <video_path> [output_path]")
        sys.exit(1)
        
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "extracted_audio.wav"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    extract_audio(video_path, output_path)