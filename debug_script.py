import sys
import os
import platform
import subprocess

def get_system_info():
    info = {}
    info['platform'] = platform.system()
    info['platform_release'] = platform.release()
    info['platform_version'] = platform.version()
    info['architecture'] = platform.machine()
    info['python_version'] = platform.python_version()
    return info

def check_video_file(video_path):
    result = {'path': os.path.abspath(video_path), 'exists': False}
    try:
        if os.path.exists(video_path):
            result['exists'] = True
            result['size_mb'] = os.path.getsize(video_path) / (1024*1024)
            # Try to get video info using ffprobe if available
            try:
                ffprobe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                             'format=duration,size,bit_rate', '-of', 
                             'default=noprint_wrappers=1:nokey=1', video_path]
                output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT, text=True)
                result['ffprobe_output'] = output.strip()
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                result['ffprobe_error'] = str(e)
    except Exception as e:
        result['error'] = str(e)
    return result

def main():
    # Create output directory if it doesn't exist
    output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'debug_info.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write system information
        f.write("="*80 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("="*80 + "\n\n")
        for key, value in get_system_info().items():
            f.write(f"{key}: {value}\n")
        
        # Write environment variables
        f.write("\n" + "="*80 + "\n")
        f.write("ENVIRONMENT VARIABLES\n")
        f.write("="*80 + "\n\n")
        for key, value in os.environ.items():
            f.write(f"{key}={value}\n")
        
        # Check video file
        video_path = "data/real/video1.mp4"
        f.write("\n" + "="*80 + "\n")
        f.write(f"CHECKING VIDEO FILE: {video_path}\n")
        f.write("="*80 + "\n\n")
        
        video_info = check_video_file(video_path)
        for key, value in video_info.items():
            f.write(f"{key}: {value}\n")
        
        # List files in directory
        f.write("\n" + "="*80 + "\n")
        f.write("FILES IN CURRENT DIRECTORY\n")
        f.write("="*80 + "\n\n")
        for root, dirs, files in os.walk('.'):
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    
    print(f"Debug information written to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
