import os
import numpy as np
import pickle
from final_deepfake_predictor import FinalDeepfakePredictor

def test_predictor():
    print("Testing DeepFake Predictor")
    print("=" * 50)
    
    # Initialize predictor with debug mode
    predictor = FinalDeepfakePredictor(debug=True)
    
    if predictor.model is None or predictor.scaler is None or not predictor.feature_names:
        print("\nERROR: Failed to load required model components")
        print("Please ensure you have run the model training first:")
        print("python optimized_model_trainer.py")
        return
    
    print(f"\nModel loaded successfully with {len(predictor.feature_names)} features")
    print(f"First 5 features: {predictor.feature_names[:5]}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = predictor.extract_real_video_features("my_video.mp4")
    
    if features:
        print("\nFeature extraction successful!")
        print(f"Extracted {len(features)} features")
        print("\nSample features:")
        for i, (k, v) in enumerate(features.items()):
            if i >= 10:  # Show first 10 features
                break
            print(f"  {k}: {v}")
        
        # Test prediction
        print("\nTesting prediction...")
        result = predictor.predict_video("my_video.mp4")
        
        if result:
            print("\nPREDICTION SUCCESSFUL!")
            print(f"Result: {result}")
        else:
            print("\nPrediction failed")
    else:
        print("\nFeature extraction failed")

if __name__ == "__main__":
    test_predictor()
