import os
import shutil
from tqdm import tqdm
import random

def organize_dataset(source_dir, real_dir, fake_dir, split_ratio=0.8):
    """
    Organize dataset into real and fake directories
    
    Args:
        source_dir: Directory containing all videos
        real_dir: Directory to store real videos
        fake_dir: Directory to store fake videos
        split_ratio: Ratio of real to fake videos (default: 0.8 means 80% real, 20% fake)
    """
    # Create directories if they don't exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {source_dir}")
        return
    
    # Shuffle files
    random.shuffle(video_files)
    
    # Split into real and fake
    split_idx = int(len(video_files) * split_ratio)
    real_videos = video_files[:split_idx]
    fake_videos = video_files[split_idx:]
    
    # Copy files to respective directories
    print(f"Copying {len(real_videos)} real videos...")
    for video in tqdm(real_videos):
        src = os.path.join(source_dir, video)
        dst = os.path.join(real_dir, video)
        if not os.path.exists(dst):  # Skip if already exists
            shutil.copy2(src, dst)
    
    print(f"\nCopying {len(fake_videos)} fake videos...")
    for video in tqdm(fake_videos):
        src = os.path.join(source_dir, video)
        dst = os.path.join(fake_dir, video)
        if not os.path.exists(dst):  # Skip if already exists
            shutil.copy2(src, dst)
    
    print("\nDataset organization complete!")
    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize deepfake training data')
    parser.add_argument('--source', type=str, default='data/raw',
                       help='Directory containing all videos')
    parser.add_argument('--real-dir', type=str, default='data/real',
                       help='Directory to store real videos')
    parser.add_argument('--fake-dir', type=str, default='data/fake',
                       help='Directory to store fake videos')
    parser.add_argument('--ratio', type=float, default=0.8,
                       help='Ratio of real to fake videos (default: 0.8)')
    
    args = parser.parse_args()
    
    organize_dataset(args.source, args.real_dir, args.fake_dir, args.ratio)
