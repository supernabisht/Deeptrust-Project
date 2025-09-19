import os
import pickle
import numpy as np
from pathlib import Path

def verify_model():
    print("Verifying Model and Feature Compatibility")
    print("=" * 50)
    
    # Load the latest model and related files
    model_path = "models/optimized_deepfake_model_20250917_084729.pkl"
    scaler_path = "models/optimized_scaler.pkl"
    features_path = "models/selected_features.txt"
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print("ERROR: Required model files not found!")
        print("Please ensure you have the following files in the models/ directory:")
        print(f"- {model_path}")
        print(f"- {scaler_path}")
        print(f"- {features_path}")
        return
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"\n✅ Model loaded successfully: {model.__class__.__name__}")
    except Exception as e:
        print(f"\n❌ Failed to load model: {str(e)}")
        return
    
    # Load the scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded successfully: {scaler.__class__.__name__}")
        print(f"   - Features expected by scaler: {scaler.n_features_in_}")
    except Exception as e:
        print(f"\n❌ Failed to load scaler: {str(e)}")
        return
    
    # Load feature names
    try:
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
        print(f"✅ Feature names loaded: {len(feature_names)} features")
        print("   First 10 features:")
        for i, feat in enumerate(feature_names[:10]):
            print(f"   {i+1}. {feat}")
        if len(feature_names) > 10:
            print(f"   ... and {len(feature_names) - 10} more")
    except Exception as e:
        print(f"\n❌ Failed to load feature names: {str(e)}")
        return
    
    # Verify feature dimensions
    if hasattr(scaler, 'n_features_in_') and len(feature_names) != scaler.n_features_in_:
        print(f"\n⚠️ WARNING: Feature count mismatch!")
        print(f"- Features in scaler: {scaler.n_features_in_}")
        print(f"- Features in feature list: {len(feature_names)}")
    else:
        print("\n✅ Feature dimensions match between scaler and feature list")
    
    # Create a test feature vector
    test_features = np.zeros((1, len(feature_names)))
    
    try:
        # Test scaling
        scaled_features = scaler.transform(test_features)
        print("\n✅ Feature scaling successful")
        print(f"   Input shape: {test_features.shape}")
        print(f"   Output shape: {scaled_features.shape}")
    except Exception as e:
        print(f"\n❌ Feature scaling failed: {str(e)}")
        return
    
    try:
        # Test prediction
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features) if hasattr(model, 'predict_proba') else None
        print("\n✅ Model prediction successful")
        print(f"   Prediction: {prediction[0]}")
        if proba is not None:
            print(f"   Probabilities: {proba[0]}")
    except Exception as e:
        print(f"\n❌ Model prediction failed: {str(e)}")
        return
    
    print("\nVerification complete!")

if __name__ == "__main__":
    verify_model()
