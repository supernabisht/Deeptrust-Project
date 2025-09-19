import os
import sys
import subprocess

def get_video_info(video_path):
    """Get video information using ffprobe if available"""
    try:
        # Try using ffprobe to get video information
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            'success': True,
            'info': result.stdout.strip(),
            'error': None
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'info': None,
            'error': f"FFprobe error: {e.stderr}"
        }
    except FileNotFoundError:
        return {
            'success': False,
            'info': None,
            'error': "FFprobe not found. Please install FFmpeg and add it to your PATH."
        }

def main():
    video_path = "data/real/video1.mp4"
    
    print(f"Checking video file: {os.path.abspath(video_path)}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File does not exist at {os.path.abspath(video_path)}")
        return 1
    
    # Check file size
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # in MB
    print(f"File size: {file_size:.2f} MB")
    
    # Check file permissions
    print(f"Read permission: {os.access(video_path, os.R_OK)}")
    
    # Try to get video info
    print("\nAttempting to get video information...")
    video_info = get_video_info(video_path)
    
    if video_info['success']:
        print("\nVideo information:")
        print(video_info['info'])
    else:
        print(f"\nCould not get video information: {video_info['error']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
