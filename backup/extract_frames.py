import cv2
import os
import shutil
from datetime import datetime

def extract_frames(video_path, output_dir="lip_frames", frame_interval=1):
    """
    Extract frames from video with cleanup and better organization
    """
    # Clean previous frames
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    frame_count = 0
    saved_count = 0
    
    print("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame (adjust frame_interval as needed)
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {saved_count} frames to '{output_dir}'")
    print(f"Sampling rate: 1 frame every {frame_interval} frames")
    
    return True

# Usage
if __name__ == "__main__":
    video_file = "my_video.mp4"  # Change to your video path
    extract_frames(video_file, frame_interval=2)  # Adjust interval as needed