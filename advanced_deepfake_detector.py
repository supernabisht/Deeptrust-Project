"""
Advanced Deepfake Detection System
=================================
This script provides a comprehensive deepfake detection system that analyzes:
1. Audio features (MFCC)
2. Lip-sync accuracy
3. Facial expressions and landmarks
4. Temporal consistency
"""

import os
import cv2
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp
from moviepy.editor import VideoFileClip
import tempfile
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 16000
MFCC_FEATURES = 13
FRAME_RATE = 30
FACE_DETECTION_CONFIDENCE = 0.7
TEMP_DIR = "temp_analysis"
MODEL_PATH = "models/deepfake_detection_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

class FeatureExtractor:
    """Extracts features from video and audio for deepfake detection."""
    
    def __init__(self):
        """Initialize the feature extractor with required models."""
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define facial landmarks for lip and eye analysis
        self.LIPS = [
            61, 291, 39, 181,  # Outer lips
            0, 17, 269, 405,   # Inner lips
            13, 14, 312, 317,  # Mouth corners
            78, 308, 191, 80   # Additional mouth points
        ]
        
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    def extract_audio_features(self, video_path: str) -> Optional[Dict]:
        """Extract audio features from video using librosa."""
        try:
            # Extract audio from video
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Create temporary audio file
            temp_audio_path = os.path.join(TEMP_DIR, "temp_audio.wav")
            audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE, verbose=False, logger=None)
            
            # Load audio file
            y, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES, hop_length=512, n_fft=2048)
            
            # Calculate deltas
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Calculate statistics for each MFCC coefficient
            features = {}
            for i, (m, d, d2) in enumerate(zip(mfcc, delta, delta2)):
                features[f'mfcc_{i+1}_mean'] = np.mean(m)
                features[f'mfcc_{i+1}_std'] = np.std(m)
                features[f'delta_{i+1}_mean'] = np.mean(d)
                features[f'delta_{i+1}_std'] = np.std(d)
                features[f'delta2_{i+1}_mean'] = np.mean(d2)
                features[f'delta2_{i+1}_std'] = np.std(d2)
            
            # Add spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features.update({
                'spectral_centroid': np.mean(spectral_centroid),
                'spectral_bandwidth': np.mean(spectral_bandwidth),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'zero_crossing_rate': np.mean(zero_crossing_rate)
            })
            
            # Clean up
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def extract_visual_features(self, video_path: str, max_frames: int = 100) -> Dict:
        """Extract visual features from video frames."""
        features = {
            'lip_width_mean': [],
            'lip_height_mean': [],
            'lip_area_mean': [],
            'lip_movement_std': [],
            'eye_aspect_ratio': [],
            'mouth_aspect_ratio': [],
            'facial_landmark_std': []
        }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {k: 0 for k in features}
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // max_frames) if total_frames > max_frames else 1
        
        while cap.isOpened() and frame_count < max_frames * frame_skip:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract lip landmarks
                lip_points = np.array([
                    (int(face_landmarks.landmark[i].x * frame.shape[1]),
                     int(face_landmarks.landmark[i].y * frame.shape[0]))
                    for i in self.LIPS
                ])
                
                # Calculate lip dimensions
                if len(lip_points) > 0:
                    x_min, y_min = np.min(lip_points, axis=0)
                    x_max, y_max = np.max(lip_points, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    area = width * height
                    
                    features['lip_width_mean'].append(width)
                    features['lip_height_mean'].append(height)
                    features['lip_area_mean'].append(area)
                
                # Calculate eye aspect ratio (simplified)
                left_eye = np.array([
                    (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                    for i in self.LEFT_EYE
                ])
                
                right_eye = np.array([
                    (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                    for i in self.RIGHT_EYE
                ])
                
                if len(left_eye) > 0 and len(right_eye) > 0:
                    # Simple eye aspect ratio (distance between eye corners / height)
                    left_ear = (np.linalg.norm(left_eye[0] - left_eye[8]) + 
                               np.linalg.norm(left_eye[4] - left_eye[12])) / (2 * np.linalg.norm(left_eye[2] - left_eye[6]))
                    
                    right_ear = (np.linalg.norm(right_eye[0] - right_eye[8]) + 
                                np.linalg.norm(right_eye[4] - right_eye[12])) / (2 * np.linalg.norm(right_eye[2] - right_eye[6]))
                    
                    features['eye_aspect_ratio'].append((left_ear + right_ear) / 2)
                
                # Calculate mouth aspect ratio
                if len(lip_points) >= 8:  # Need at least 8 points for MAR
                    # Horizontal distance between mouth corners
                    mouth_width = np.linalg.norm(lip_points[0] - lip_points[6])
                    # Vertical distance between upper and lower lips
                    mouth_height = (np.linalg.norm(lip_points[2] - lip_points[10]) +
                                  np.linalg.norm(lip_points[4] - lip_points[8])) / 2
                    
                    mar = mouth_height / (mouth_width + 1e-6)  # Avoid division by zero
                    features['mouth_aspect_ratio'].append(mar)
                
                # Calculate facial landmark movement
                if frame_count > 0:
                    # Compare with previous frame
                    prev_points = np.array([
                        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                        for i in range(len(face_landmarks.landmark))
                    ])
                    
                    if 'prev_landmarks' in locals():
                        movement = np.mean(np.linalg.norm(prev_points - prev_landmarks, axis=1))
                        features['facial_landmark_std'].append(movement)
                    
                    prev_landmarks = prev_points
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics for each feature
        result = {}
        for key, values in features.items():
            if values:  # Only process if we have values
                result[f'{key}_mean'] = np.mean(values)
                result[f'{key}_std'] = np.std(values) if len(values) > 1 else 0
                result[f'{key}_min'] = np.min(values)
                result[f'{key}_max'] = np.max(values)
            else:
                result[f'{key}_mean'] = 0
                result[f'{key}_std'] = 0
                result[f'{key}_min'] = 0
                result[f'{key}_max'] = 0
        
        return result

class DeepfakeDetector:
    """Main class for deepfake detection."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """Initialize the detector with optional pre-trained model and scaler."""
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = None
        
        # Load model and scaler if paths are provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, scaler_path)
        else:
            # Initialize with default model if no model is loaded
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with a default model if no pre-trained model is available."""
        # Use a simple ensemble of classifiers
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft',
            n_jobs=-1
        )
        
        # Initialize standard scaler
        self.scaler = StandardScaler()
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load a pre-trained model and scaler."""
        try:
            self.model = joblib.load(model_path)
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self._initialize_default_model()
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """Save the current model and scaler to disk."""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            
            if scaler_path and self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
            
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def extract_features(self, video_path: str) -> Optional[Dict]:
        """Extract all features from a video file."""
        print("Extracting audio features...")
        audio_features = self.feature_extractor.extract_audio_features(video_path)
        
        print("Extracting visual features...")
        visual_features = self.feature_extractor.extract_visual_features(video_path)
        
        if audio_features is None or visual_features is None:
            print("Error: Failed to extract features from the video.")
            return None
        
        # Combine all features
        features = {**audio_features, **visual_features}
        return features
    
    def preprocess_features(self, features: Dict) -> np.ndarray:
        """Preprocess features for model prediction."""
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([features])
        
        # Handle missing values
        df = df.fillna(0)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            try:
                # Fit scaler if not already fitted
                if not hasattr(self.scaler, 'n_features_in_'):
                    self.scaler.fit(df)
                
                # Scale features
                scaled_features = self.scaler.transform(df)
                return scaled_features
            except Exception as e:
                print(f"Error scaling features: {e}")
                return df.values
        else:
            return df.values
    
    def predict(self, video_path: str) -> Dict:
        """
        Predict if a video is a deepfake.
        
        Args:
            video_path: Path to the video file to analyze.
            
        Returns:
            Dict containing prediction results and confidence scores.
        """
        if not os.path.exists(video_path):
            return {
                'success': False,
                'error': f'Video file not found: {video_path}',
                'is_deepfake': None,
                'confidence': 0.0,
                'features_extracted': False,
                'analysis_time': 0.0
            }
        
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_features(video_path)
            if features is None:
                return {
                    'success': False,
                    'error': 'Failed to extract features from the video.',
                    'is_deepfake': None,
                    'confidence': 0.0,
                    'features_extracted': False,
                    'analysis_time': time.time() - start_time
                }
            
            # Preprocess features
            X = self.preprocess_features(features)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = max(proba)
                is_deepfake = int(proba[1] > 0.5)  # Assuming class 1 is deepfake
            else:
                # Fallback for models without predict_proba
                prediction = self.model.predict(X)[0]
                is_deepfake = int(prediction)
                confidence = 0.8 if is_deepfake == prediction else 0.2
            
            # Calculate analysis time
            analysis_time = time.time() - start_time
            
            # Prepare result
            result = {
                'success': True,
                'is_deepfake': bool(is_deepfake),
                'confidence': float(confidence),
                'features_extracted': True,
                'analysis_time': analysis_time,
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'video_path': os.path.abspath(video_path)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'is_deepfake': None,
                'confidence': 0.0,
                'features_extracted': False,
                'analysis_time': time.time() - start_time
            }

def main():
    """Main function to run the deepfake detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Deepfake Detection System')
    parser.add_argument('video_path', type=str, help='Path to the video file to analyze')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                       help='Path to the pre-trained model file')
    parser.add_argument('--scaler', type=str, default=SCALER_PATH,
                       help='Path to the feature scaler file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the analysis results (JSON)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADVANCED DEEPFAKE DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = DeepfakeDetector(args.model, args.scaler)
    
    # Analyze video
    print(f"\nAnalyzing video: {args.video_path}")
    result = detector.predict(args.video_path)
    
    # Display results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    if not result['success']:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Display detection result
    if result['is_deepfake']:
        print("\n🔴 DETECTED: This video appears to be a DEEPFAKE!")
    else:
        print("\n🟢 VERIFIED: This video appears to be AUTHENTIC!")
    
    # Display confidence level
    confidence = result['confidence']
    confidence_percent = int(confidence * 100)
    print(f"\n📊 Confidence Level: {confidence_percent}%")
    
    # Display a confidence bar
    bar_length = 30
    filled_length = int(round(bar_length * confidence))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"   [{bar}] {confidence_percent}%")
    
    # Display analysis time
    print(f"\n⏱️  Analysis Time: {result['analysis_time']:.2f} seconds")
    
    # Display a final message
    print("\n" + "=" * 60)
    if result['is_deepfake']:
        print("⚠️  WARNING: This video shows signs of manipulation.")
        print("   Please verify the content from trusted sources.")
    else:
        print("✅ This video appears to be authentic.")
        print("   However, no system is 100% accurate. Always verify from multiple sources.")
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                 np.int16, np.int32, np.int64, np.uint8,
                                 np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy_types(result), f, indent=2)
        print(f"\n📄 Results saved to: {os.path.abspath(args.output)}")
    
    print("\nThank you for using the Advanced Deepfake Detection System!")

if __name__ == "__main__":
    main()
