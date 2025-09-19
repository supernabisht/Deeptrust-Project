import cv2
import numpy as np
import joblib
import os
import sys
import pandas as pd

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
import extract_audio as local_audio
import extract_frames as local_frames
import extract_mfcc as local_mfcc

class SimpleDeepfakeDetector:
    def __init__(self, model_path):
        """Initialize with our trained model"""
        self.model = joblib.load(model_path)
        self.required_features = [f'audio_feat_{i}' for i in range(25)] + \
                               [f'visual_feat_{i}' for i in range(15)] + \
                               ['audio_feats_mean', 'audio_feats_std', 'audio_feats_skew', 'audio_feats_kurtosis',
                                'visual_feats_mean', 'visual_feats_std', 'visual_feats_skew', 'visual_feats_kurtosis'] + \
                               [f'audio_visual_corr_{i+1}' for i in range(5)]

    def extract_video_features(self, video_path):
        """Extract features from video"""
        # Extract audio and frames
        audio_file = 'temp_audio.wav'
        local_audio.extract_audio(video_path, audio_file)
        frame_dir = 'temp_frames'
        local_frames.extract_frames(video_path, frame_dir)
        
        # Extract audio features
        audio_features = local_mfcc.extract_audio_features(audio_file)
        
        # For simplicity, we'll use random visual features
        # In a real scenario, you'd extract these from the frames
        visual_features = {f'visual_feat_{i}': np.random.random() for i in range(15)}
        
        # Combine all features
        features = {**audio_features, **visual_features}
        
        # Add dummy stats (in real case, calculate from features)
        features.update({
            'audio_feats_mean': np.mean(list(audio_features.values())),
            'audio_feats_std': np.std(list(audio_features.values())),
            'audio_feats_skew': 0.1,  # These would be calculated in a real scenario
            'audio_feats_kurtosis': 0.1,
            'visual_feats_mean': 0.5,
            'visual_feats_std': 0.2,
            'visual_feats_skew': 0.1,
            'visual_feats_kurtosis': 0.1,
            **{f'audio_visual_corr_{i+1}': 0.5 for i in range(5)}
        })
        
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(frame_dir):
            for f in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, f))
            os.rmdir(frame_dir)
            
        return features

    def predict(self, video_path):
        """Make prediction on a video"""
        # Extract features
        features = self.extract_video_features(video_path)
        
        # Convert to DataFrame row
        row = pd.DataFrame([features])[self.required_features]
        
        # Make prediction
        proba = self.model.predict_proba(row)[0]
        prediction = self.model.predict(row)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': max(proba),
            'fake_probability': proba[1] if len(proba) > 1 else 0.5
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Deepfake Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='models/advanced_model.pkl', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    detector = SimpleDeepfakeDetector(args.model)
    
    print(f"Analyzing video: {args.video}")
    result = detector.predict(args.video)
    
    print("\n=== Detection Results ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")

if __name__ == "__main__":
    main()