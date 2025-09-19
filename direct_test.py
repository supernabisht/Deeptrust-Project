"""
Direct test of the LipSyncAnalyzer on a video file.
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# Add the current directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

# Import the LipSyncAnalyzer
try:
    from lip_sync_detector.lip_sync_analyzer import LipSyncAnalyzer
    print("Successfully imported LipSyncAnalyzer")
except ImportError as e:
    print(f"Error importing LipSyncAnalyzer: {e}")
    sys.exit(1)

def test_video(video_path: str):
    """Test the LipSyncAnalyzer on a video file.
    
    Args:
        video_path: Path to the input video file
    """
    print(f"Testing video: {video_path}")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Initialize the analyzer
    print("Initializing LipSyncAnalyzer...")
    analyzer = LipSyncAnalyzer()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"- FPS: {fps:.2f}")
    print(f"- Frame count: {frame_count}")
    print(f"- Duration: {duration:.2f} seconds")
    
    # Process each frame
    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame to speed up testing
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue
        
        # Convert to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract mouth ROI
        mouth_roi, landmarks = analyzer.extract_mouth_roi(frame_rgb)
        
        if mouth_roi is not None:
            processed_frames += 1
            
            # Display the frame with landmarks
            display_frame = frame.copy()
            
            # Draw landmarks if available
            if landmarks and 'landmarks' in landmarks and landmarks['landmarks'] is not None:
                for (x, y) in landmarks['landmarks']:
                    cv2.circle(display_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            # Show the frame with landmarks
            cv2.imshow("Lip Sync Analysis", display_frame)
            
            # Show the mouth ROI
            cv2.imshow("Mouth ROI", mouth_roi)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
    
    # Calculate processing time
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / processed_frames if processed_frames > 0 else 0
    
    print("\nProcessing complete!")
    print(f"- Total frames processed: {processed_frames}")
    print(f"- Total processing time: {total_time:.2f} seconds")
    print(f"- Average time per frame: {avg_time_per_frame*1000:.2f} ms")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with the sample video we created
    video_path = os.path.join("data", "fake", "test_video.mp4")
    
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
            if video_path != os.path.join("data", "fake", "test_video.mp4"):
                break
    
    # If we found a video, test it
    if os.path.exists(video_path):
        test_video(video_path)
    else:
        print("No video files found in the data directory.")
        print("Please place a video file in the data/fake/ directory and try again.")
        print("You can also specify a video file path as a command-line argument.")
