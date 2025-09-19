import os
import cv2
import numpy as np
import pandas as pd
import librosa
import moviepy.editor as mp
from tqdm import tqdm
import dlib
import json
from typing import Dict, List, Tuple

class EnhancedFeatureExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def extract_audio_features(self, audio_path: str) -> Dict:
        """Extract comprehensive audio features including MFCCs"""
        try:
            y, sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return {
                'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
                'mfcc_std': np.std(mfcc, axis=1).tolist(),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            }
        except Exception as e:
            print(f"Audio feature extraction failed: {e}")
            return {}

    def extract_lip_movement(self, video_path: str) -> Dict:
        """Extract lip movement features using facial landmarks"""
        cap = cv2.VideoCapture(video_path)
        lip_distances = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0:
                landmarks = self.predictor(gray, faces[0])
                # Calculate mouth aspect ratio
                mouth_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                lip_width = np.linalg.norm(mouth_pts[6] - mouth_pts[0])
                lip_height = np.linalg.norm(mouth_pts[3] - mouth_pts[9])
                lip_distances.append(lip_height / lip_width)
        
        cap.release()
        return {
            'lip_movement_mean': float(np.mean(lip_distances)) if lip_distances else 0,
            'lip_movement_std': float(np.std(lip_distances)) if lip_distances else 0
        }

def process_video(video_path: str) -> Dict:
    """Process a video and extract all features"""
    extractor = EnhancedFeatureExtractor()
    
    # Extract audio
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, logger=None)
    
    # Extract features
    audio_features = extractor.extract_audio_features(audio_path)
    lip_features = extractor.extract_lip_movement(video_path)
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return {**audio_features, **lip_features}

if __name__ == "__main__":
    video_path = "my_video.mp4"  # Change this to your video path
    features = process_video(video_path)
    print(json.dumps(features, indent=2))