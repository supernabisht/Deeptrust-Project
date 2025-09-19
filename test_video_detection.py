"""
Test script for deepfake detection on a video file.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the DeepfakeDetector
try:
    from lip_sync_detector.deepfake_detector import DeepfakeDetector
    print("Successfully imported DeepfakeDetector")
except ImportError as e:
    print(f"Error importing DeepfakeDetector: {e}")
    print("Trying alternative import...")
    from deepfake_detector import DeepfakeDetector

def test_video_detection(video_path: str):
    """Test deepfake detection on a video file.
    
    Args:
        video_path: Path to the input video file
    """
    print(f"Testing deepfake detection on video: {video_path}")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Initialize the detector
    print("Initializing DeepfakeDetector...")
    detector = DeepfakeDetector()
    
    # Make a prediction
    print("Running detection...")
    try:
        result = detector.predict(video_path)
        print("\nDetection Results:")
        print(f"- Is Deepfake: {result['is_deepfake']}")
        print(f"- Confidence: {result['confidence']:.2f}")
        print(f"- Processing Time: {result['processing_time']:.2f} seconds")
        print(f"- Frames Processed: {result['frames_processed']}")
        
        # Display detailed results if available
        if 'details' in result:
            print("\nDetailed Analysis:")
            for key, value in result['details'].items():
                print(f"- {key}: {value}")
                
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the sample video in the data/fake directory
    video_path = os.path.join("data", "fake", "video1.mp4")
    
    # If the sample video doesn't exist, try to find any video in the data directory
    if not os.path.exists(video_path):
        print(f"Sample video not found at {video_path}")
        print("Looking for other video files in the data directory...")
        
        # Search for video files in the data directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for root, _, files in os.walk("data"):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    print(f"Found video file: {video_path}")
                    break
            if video_path != os.path.join("data", "fake", "video1.mp4"):
                break
    
    # If we found a video, test it
    if os.path.exists(video_path):
        test_video_detection(video_path)
    else:
        print("No video files found in the data directory.")
        print("Please place a video file in the data/fake/ directory and try again.")
        print("You can also specify a video file path as a command-line argument.")
        
        # Check if a video file was provided as a command-line argument
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            if os.path.exists(video_path):
                test_video_detection(video_path)
            else:
                print(f"Error: Video file not found at {video_path}")
