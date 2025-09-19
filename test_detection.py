"""
Simplified test script for deepfake detection using moviepy for audio extraction.
"""

import os
import sys
import cv2
import torch
from pathlib import Path
from moviepy.editor import VideoFileClip
import tempfile
import numpy as np

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

def extract_audio_moviepy(video_path, output_path=None, sample_rate=16000):
    """Extract audio from video using moviepy."""
    try:
        # Create a temporary file if no output path is provided
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write audio to file
        audio.write_audiofile(output_path, fps=sample_rate, verbose=False, logger=None)
        
        # Load the audio file for processing
        import librosa
        y, sr = librosa.load(output_path, sr=sample_rate, mono=True)
        
        # Clean up temporary file if we created one
        if 'temp_file' in locals():
            os.unlink(output_path)
            
        return y, sr
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None

def test_detection(video_path):
    """Test deepfake detection on a video file."""
    print(f"Testing video: {video_path}")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Extract audio using moviepy
    print("Extracting audio...")
    audio, sr = extract_audio_moviepy(video_path)
    
    if audio is None or sr is None:
        print("Failed to extract audio from video")
        return
    
    print(f"Extracted audio: {len(audio)} samples at {sr} Hz")
    
    # Initialize the detector
    print("Initializing detector...")
    try:
        from lip_sync_detector.deepfake_detector import DeepfakeDetector
        detector = DeepfakeDetector()
        print("DeepfakeDetector initialized successfully")
        
        # Make a prediction
        print("Running detection...")
        result = detector.predict(video_path)
        
        print("\nDetection Results:")
        print(f"- Is Deepfake: {result.get('is_deepfake', 'N/A')}")
        print(f"- Confidence: {result.get('confidence', 0):.2f}")
        print(f"- Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the sample video we created
    video_path = os.path.join("data", "fake", "test_video.mp4")
    
    # If a video path is provided as a command-line argument, use that instead
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # If the video file doesn't exist, look for any video in the data directory
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        print("Looking for video files in the data directory...")
        
        # Search for video files in the data directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for root, _, files in os.walk("data"):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    print(f"Found video file: {video_path}")
                    break
            if video_path != os.path.join("data", "fake", "test_video.mp4"):
                break
    
    # If we found a video, test it
    if os.path.exists(video_path):
        test_detection(video_path)
    else:
        print("No video files found in the data directory.")
        print("Please provide a video file path as an argument.")
