# extract_frames.py
import cv2
import os
import sys
from pathlib import Path

def extract_frames(video_path, output_dir="frames", frame_interval=1):
    """Extract frames from video file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
        
        frame_count = 0
        saved_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Total frames in video: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_file = output_dir / f"frame_{frame_count:06d}.jpg"
                success = cv2.imwrite(str(frame_file), frame)
                if not success:
                    print(f"Warning: Failed to save frame {frame_count}")
                else:
                    saved_count += 1
                    if saved_count % 10 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames...")
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {saved_count} frames to {output_dir}")
        return saved_count > 0
        
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_file> [output_dir] [frame_interval]")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "frames"
    frame_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if extract_frames(video_file, output_dir, frame_interval):
        print("Frame extraction completed successfully")
        sys.exit(0)
    else:
        print("Frame extraction failed")
        sys.exit(1)