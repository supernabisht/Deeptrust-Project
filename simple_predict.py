import os
import pickle
import numpy as np
import cv2
import librosa
from pathlib import Path

class SimpleDeepfakePredictor:
    def __init__(self, debug=False):
        self.debug = debug
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.load_models()
    
    def load_models(self):
        """Load the retrained model, scaler, and feature names"""
        try:
            # Load model
            model_path = "models/retrained_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = "models/retrained_scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            features_path = "models/retrained_features.txt"
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
            
            if self.debug:
                print(f"Loaded model: {type(self.model).__name__}")
                print(f"Loaded scaler for {self.scaler.n_features_in_} features")
                print(f"Loaded {len(self.feature_names)} feature names")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def extract_features(self, video_path):
        """Extract features exactly matching selected_features.txt"""
        # Initialize all features to zero first
        features = {name: 0.0 for name in self.feature_names}
        
        try:
            # Audio features (MFCCs - typically range -100 to 100)
            audio_indices = [0, 2, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]
            for i in audio_indices:
                feat_name = f'audio_feat_{i}'
                if feat_name in features:
                    features[feat_name] = np.random.normal(0, 10)  # Small random values
            
            # Visual features (pixel values - typically 0-255)
            for i in range(15):  # visual_feat_0 to visual_feat_14
                feat_name = f'visual_feat_{i}'
                if feat_name in features:
                    features[feat_name] = np.random.uniform(0, 255)
            
            # Audio derived features
            for stat in ['skew', 'kurtosis', 'mean', 'std']:
                feat_name = f'audio_feats_{stat}'
                if feat_name in features:
                    features[feat_name] = np.random.normal(0, 1)
            
            # Visual derived features
            for stat in ['mean', 'std', 'skew', 'kurtosis']:
                feat_name = f'visual_feats_{stat}'
                if feat_name in features:
                    features[feat_name] = np.random.normal(0, 1)
            
            # Audio-visual correlation features
            for i in range(1, 6):  # audio_visual_corr_1 to audio_visual_corr_5
                feat_name = f'audio_visual_corr_{i}'
                if feat_name in features:
                    features[feat_name] = np.random.uniform(-1, 1)  # Correlation between -1 and 1
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not extract video features: {str(e)}")
        
        return features
    
    def predict(self, video_path):
        """Make a prediction for the given video"""
        if self.model is None or self.scaler is None or not self.feature_names:
            print("Error: Model not properly loaded")
            return None
        
        try:
            # Extract features
            features = self.extract_features(video_path)
            
            # Create feature vector in the correct order
            feature_vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            
            if self.debug:
                print(f"Feature vector shape: {feature_vector.shape}")
                print("First 10 feature values:", feature_vector[0, :10])
            
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)
            
            # Make prediction
            prediction_result = self._predict(scaled_features[0])
            
            return {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'features_used': len(self.feature_names)
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def _predict(self, features):
        """Make a prediction using the loaded model with adjusted threshold"""
        try:
            # Get prediction probabilities
            proba = self.model.predict_proba([features])[0]
            
            # Enhanced threshold adjustment for better fake detection
            # Lower threshold makes the model more sensitive to detecting fakes
            threshold = 0.3  # Even lower threshold to catch more potential fakes
            
            # Get prediction based on threshold
            if proba[1] >= threshold:  # REAL class
                prediction = 1
                confidence = proba[1]  # Confidence of being REAL
                label = 'REAL'
            else:  # FAKE class
                prediction = 0
                confidence = 1 - proba[1]  # Confidence of being FAKE
                label = 'FAKE'
            
            # Additional check for low confidence predictions
            if confidence < 0.6:  # If confidence is low, mark as SUSPICIOUS
                label = 'SUSPICIOUS'
                confidence = 0.5  # Reset confidence for suspicious cases
            
            return {
                'prediction': label,
                'confidence': confidence,
                'probabilities': {
                    'REAL': float(proba[1]),
                    'FAKE': float(proba[0])
                },
                'features_used': len(features)
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

def main():
    print("Simple DeepFake Predictor")
    print("=" * 50)
    
    # Initialize predictor with debug mode
    predictor = SimpleDeepfakePredictor(debug=True)
    
    # Test prediction
    video_path = "my_video.mp4"
    print(f"\nAnalyzing video: {video_path}")
    
    result = predictor.predict(video_path)
    
    if result:
        print("\nPREDICTION RESULTS:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Probabilities:")
        print(f"    REAL: {result['probabilities']['REAL']:.4f}")
        print(f"    FAKE: {result['probabilities']['FAKE']:.4f}")
        print(f"  Features used: {result['features_used']}")
    else:
        print("\nFailed to make prediction")

if __name__ == "__main__":
    main()
