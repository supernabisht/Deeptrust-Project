import joblib
import numpy as np
from enhanced_feature_extraction import process_video
import json

class EnhancedDeepfakeDetector:
    def __init__(self, model_path='models/enhanced_deepfake_model.pkl', 
                 scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def predict(self, video_path: str) -> dict:
        # Extract features
        features = process_video(video_path)
        
        # Prepare feature vector
        feature_vector = np.array(
            features['mfcc_mean'] + 
            features['mfcc_std'] + 
            [
                features['zero_crossing_rate'],
                features['spectral_centroid'],
                features['spectral_bandwidth'],
                features['lip_movement_mean'],
                features['lip_movement_std']
            ]
        ).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Make prediction
        prob = self.model.predict_proba(scaled_features)[0]
        prediction = self.model.predict(scaled_features)[0]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(max(prob)),
            'features': features
        }

if __name__ == "__main__":
    detector = EnhancedDeepfakeDetector()
    result = detector.predict("my_video.mp4")  # Change to your video path
    print("\nPrediction Results:")
    print("=" * 40)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nExtracted Features:")
    print(json.dumps(result['features'], indent=2))