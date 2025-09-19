"""
Create a simple test video with a talking face animation.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_talking_head_video(output_path, duration=5, fps=25, width=640, height=360):
    """Create a simple talking head video with a moving mouth.
    
    Args:
        output_path: Path to save the output video
        duration: Duration of the video in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"- Duration: {duration} seconds")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    
    # Create frames
    for i in range(int(fps * duration)):
        # Create a blank frame with blue background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # Blue background
        
        # Draw a white face
        face_center = (width // 2, height // 2)
        face_radius = min(width, height) // 3
        cv2.circle(frame, face_center, face_radius, (255, 255, 255), -1)  # White face
        
        # Draw eyes
        eye_y = height // 2 - face_radius // 3
        eye_radius = face_radius // 5
        cv2.circle(frame, (width // 2 - face_radius // 2, eye_y), eye_radius, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (width // 2 + face_radius // 2, eye_y), eye_radius, (0, 0, 0), -1)  # Right eye
        
        # Animate mouth (opens and closes)
        mouth_open = 0.5 + 0.5 * np.sin(i * 0.5)  # Varies between 0 and 1
        mouth_width = int(face_radius * 0.8)
        mouth_height = int(face_radius * 0.4 * mouth_open)
        mouth_top = height // 2 + face_radius // 2 - mouth_height // 2
        
        # Draw mouth (as a simple ellipse)
        cv2.ellipse(
            frame,
            (width // 2, mouth_top + mouth_height // 2),
            (mouth_width // 2, mouth_height // 2),
            0, 0, 180, (0, 0, 0), -1
        )
        
        # Add frame number
        cv2.putText(
            frame, f"Frame {i+1}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255, 255, 255), 2
        )
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Test video created: {output_path}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("data/fake", exist_ok=True)
    
    # Create a test video
    output_path = os.path.join("data", "fake", "test_video.mp4")
    create_talking_head_video(output_path)
