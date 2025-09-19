import os
import sys
import joblib
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
import librosa
import cv2

class QuickDeepfakeDetector:
    def __init__(self, model_path):
        """Initialize with our trained model"""
        self.model = joblib.load(model_path)
        
    def extract_audio_features(self, audio_path):
        """Extract basic audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract basic features
            features = {
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                'mfcc_0': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[0]),
                'mfcc_1': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1]),
            }
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {}

    def extract_video_features(self, video_path):
        """Extract basic video features"""
        try:
            # Extract audio from video
            audio_path = 'temp_audio.wav'
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            
            # Get basic video properties
            duration = video.duration
            fps = video.fps
            width, height = video.size
            
            # Read first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            # Basic frame stats (if frame was read)
            frame_features = {}
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_features = {
                    'frame_mean': np.mean(gray),
                    'frame_std': np.std(gray),
                    'frame_width': width,
                    'frame_height': height,
                    'fps': fps,
                    'duration': duration
                }
            
            # Get audio features
            audio_features = self.extract_audio_features(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return {**audio_features, **frame_features}
            
        except Exception as e:
            print(f"Error extracting video features: {e}")
            return {}

    def predict(self, video_path):
        """Make prediction on a video"""
        try:
            # Extract features
            features = self.extract_video_features(video_path)
            if not features:
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': 'Failed to extract features'
                }
            
            # Convert to DataFrame row
            df = pd.DataFrame([features])
            
            # Make prediction (using predict_proba if available)
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df)[0]
                prediction = self.model.classes_[np.argmax(proba)]
                confidence = max(proba)
            else:
                prediction = self.model.predict(df)[0]
                confidence = 1.0  # Default confidence if no probabilities
            
            return {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': float(confidence),
                'features_extracted': list(features.keys())
            }
            
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Deepfake Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='models/advanced_model.pkl', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    try:
        detector = QuickDeepfakeDetector(args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print(f"\nAnalyzing video: {args.video}")
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
        
    result = detector.predict(args.video)
    
    print("\n=== Detection Results ===")
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nFeatures used: {', '.join(result.get('features_extracted', []))}")

if __name__ == "__main__":
    main()
