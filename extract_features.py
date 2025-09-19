#!/usr/bin/env python3
"""
Feature Extraction for Deepfake Detection

This script extracts audio-visual features from video files for deepfake detection.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts audio-visual features from video files"""
    
    def __init__(self, output_dir='extracted_features'):
        """Initialize the feature extractor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio parameters
        self.sr = 22050  # Sample rate
        self.n_mfcc = 40  # Number of MFCC coefficients
        
        # Video parameters
        self.target_fps = 25  # Target frames per second
        self.frame_size = (224, 224)  # Target frame size
    
    def extract_audio_features(self, audio_path):
        """Extract audio features from an audio file"""
        try:
            # Load audio file
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
            
            # Calculate statistics
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Create feature dictionary
            features = {}
            for i in range(self.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
                features[f'mfcc_{i+1}_std'] = mfcc_std[i]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return {}
    
    def extract_video_features(self, video_path, max_frames=300):
        """Extract visual features from a video file"""
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to achieve target FPS
            frame_interval = max(1, int(fps / self.target_fps))
            
            # Initialize face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Initialize feature lists
            face_areas = []
            face_positions = []
            
            frame_num = 0
            processed_frames = 0
            
            while cap.isOpened() and processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame at specified interval
                if frame_num % frame_interval == 0:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        # Get the largest face
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        face_areas.append(w * h)
                        face_positions.append((x, y, w, h))
                    
                    processed_frames += 1
                
                frame_num += 1
            
            cap.release()
            
            # Calculate face statistics
            features = {}
            if face_areas:
                features['face_area_mean'] = np.mean(face_areas)
                features['face_area_std'] = np.std(face_areas)
                features['face_count'] = len(face_areas)
            else:
                features.update({
                    'face_area_mean': 0,
                    'face_area_std': 0,
                    'face_count': 0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting video features: {str(e)}")
            return {}
    
    def extract_lip_sync_features(self, video_path):
        """Extract lip-sync related features (simplified)"""
        # This is a placeholder - in a real implementation, you would analyze
        # the correlation between mouth movements and audio
        return {
            'lip_sync_score': np.random.uniform(0.7, 1.0),
            'lip_movement_variance': np.random.uniform(0.1, 0.5)
        }
    
    def process_video(self, video_path):
        """Process a video file and extract all features"""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Create temporary directory for intermediate files
            temp_dir = self.output_dir / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            # Extract audio from video
            audio_path = temp_dir / 'temp_audio.wav'
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Extract features
            audio_features = self.extract_audio_features(str(audio_path))
            video_features = self.extract_video_features(video_path)
            sync_features = self.extract_lip_sync_features(video_path)
            
            # Combine all features
            all_features = {
                'video_path': str(video_path),
                'duration': video.duration,
                'fps': video.fps,
                'width': video.w,
                'height': video.h,
                **audio_features,
                **video_features,
                **sync_features
            }
            
            # Clean up
            video.close()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None

def main():
    """Main function to process videos and extract features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from videos for deepfake detection')
    parser.add_argument('videos', nargs='+', help='Video files to process')
    parser.add_argument('--output', type=str, default='extracted_features/features.csv',
                      help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Process each video
    all_features = []
    for video_path in tqdm(args.videos, desc="Processing videos"):
        features = extractor.process_video(video_path)
        if features:
            all_features.append(features)
    
    # Save features to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
    else:
        logger.warning("No features were extracted from the provided videos")

if __name__ == "__main__":
    main()
