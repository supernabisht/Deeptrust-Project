import sys
import os

def main():
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print("Files in directory:")
    for f in os.listdir('.'):
        print(f"- {f}")
    
    video_path = "data/real/video1.mp4"
    print(f"\nChecking video file: {video_path}")
    if os.path.exists(video_path):
        print("Video file exists!")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    else:
        print("Video file not found!")

if __name__ == "__main__":
    main()
