#!/usr/bin/env python3
"""
MFCC Feature Extractor for Deepfake Detection

This script extracts MFCC features from audio that match the model's training data format.
"""
import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from pathlib import Path

# Audio parameters
SAMPLE_RATE = 22050  # Hz
N_MFCC = 40  # Number of MFCC coefficients

def extract_mfcc_features(audio_path, n_mfcc=40):
    """Extract MFCC features from an audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate statistics for each MFCC coefficient
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        
        return features
        
    except Exception as e:
        print(f"Error extracting MFCC features: {str(e)}")
        return None

def extract_from_video(video_path):
    """Extract audio from video and then extract MFCC features"""
    try:
        from moviepy.editor import VideoFileClip
        import tempfile
        
        # Create a temporary directory for the audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, 'temp_audio.wav')
            
            # Extract audio from video
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            
            # Extract MFCC features
            features = extract_mfcc_features(temp_audio)
            
            # Add video metadata
            if features:
                features.update({
                    'video_duration': video.duration,
                    'video_fps': video.fps,
                    'video_width': video.w,
                    'video_height': video.h
                })
            
            video.close()
            return features
            
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MFCC features from video/audio')
    parser.add_argument('input_file', help='Input video or audio file')
    parser.add_argument('--output', '-o', help='Output CSV file (default: features.csv)', 
                       default='features.csv')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Extract features based on file type
    if args.input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Extracting audio from video: {args.input_file}")
        features = extract_from_video(args.input_file)
    else:
        print(f"Processing audio file: {args.input_file}")
        features = extract_mfcc_features(args.input_file)
    
    if features:
        # Convert to DataFrame and save
        df = pd.DataFrame([features])
        df.to_csv(args.output, index=False)
        print(f"Features extracted successfully. Saved to {args.output}")
        print("\nExtracted features:")
        print(df.head().T)  # Transpose for better readability
        return 0
    else:
        print("Failed to extract features.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
