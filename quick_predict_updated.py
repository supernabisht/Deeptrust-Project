#!/usr/bin/env python3
"""
Updated Quick Deepfake Detection Script

This script loads a pre-trained model and makes predictions on a video file,
with proper feature extraction.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import librosa
import cv2
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_audio_features(audio_path):
    """Extract audio features including spectral centroid"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Calculate statistics
        features = {}
        
        # MFCC features
        for i in range(mfcc.shape[0]):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        
        # Spectral features
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return {}

def extract_video_features(video_path):
    """Extract basic video features"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize variables for face tracking
        face_areas = []
        face_positions = []
        
        # Process some frames
        frames_to_process = min(100, frame_count)  # Process up to 100 frames
        frame_indices = np.linspace(0, frame_count-1, frames_to_process, dtype=int)
        
        for i in range(frames_to_process):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_areas.append(w * h)
                face_positions.append((x, y, w, h))
        
        cap.release()
        
        # Calculate face statistics
        features = {
            'video_width': width,
            'video_height': height,
            'fps': fps,
            'duration': frame_count / fps if fps > 0 else 0,
            'face_count': len(face_areas)
        }
        
        if face_areas:
            features.update({
                'face_area_mean': np.mean(face_areas),
                'face_area_std': np.std(face_areas)
            })
        else:
            features.update({
                'face_area_mean': 0,
                'face_area_std': 0
            })
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting video features: {str(e)}")
        return {}

def analyze_video(video_path):
    """Analyze video and extract all required features"""
    try:
        # Create temporary directory for audio extraction
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        audio_path = temp_dir / "temp_audio.wav"
        
        # Extract audio from video
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
        
        # Extract features
        audio_features = extract_audio_features(str(audio_path))
        video_features = extract_video_features(video_path)
        
        # Combine features
        all_features = {**audio_features, **video_features}
        
        # Clean up
        video.close()
        if audio_path.exists():
            audio_path.unlink()
        
        return all_features
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return {}

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        # Look for any model in the trained_models directory
        models_dir = Path('enhanced_models/trained_models')
        model_files = list(models_dir.glob('*.joblib'))
        
        if not model_files:
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = str(model_files[0])
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Check if we have the expected model format
        if 'model' not in model_data:
            raise ValueError("Invalid model format: 'model' key not found")
            
        return model_data
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection')
    parser.add_argument('video', help='Video file to analyze')
    parser.add_argument('--model', help='Path to trained model (default: auto-detect)')
    parser.add_argument('--save-features', action='store_true', 
                       help='Save extracted features to CSV')
    
    args = parser.parse_args()
    
    try:
        # Check if video exists
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        
        # Load model
        model_data = load_model(args.model)
        model = model_data['model']
        
        # Analyze video
        logger.info(f"Analyzing video: {args.video}")
        features = analyze_video(args.video)
        
        if not features:
            raise ValueError("Failed to extract features from video")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure we have all required features
        if hasattr(model, 'feature_names_'):
            required_features = model.feature_names_
        else:
            # If feature names aren't available, try to infer from the model
            # This is a fallback and might not work for all models
            required_features = [f'f{i}' for i in range(model.n_features_in_)]
        
        # Add missing features with default values
        for feat in required_features:
            if feat not in features_df.columns:
                logger.warning(f"Missing feature: {feat} (using default value: 0)")
                features_df[feat] = 0
        
        # Reorder columns to match training data
        features_df = features_df[required_features]
        
        # Save features if requested
        if args.save_features:
            features_df.to_csv('extracted_features.csv', index=False)
            logger.info(f"Features saved to extracted_features.csv")
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Get class names
        classes = model_data.get('classes', ['REAL', 'FAKE'])
        
        # Print results
        print("\n=== Deepfake Detection Results ===")
        print(f"Video: {args.video}")
        print(f"Prediction: {classes[prediction]}")
        print("Probabilities:")
        for i, cls in enumerate(classes):
            print(f"  {cls}: {probabilities[i]:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
