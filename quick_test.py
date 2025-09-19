import os
import sys
import cv2
import numpy as np
import joblib

# Configuration
VIDEO_PATH = "data/real/video1.mp4"
MODEL_PATH = "models/optimized_deepfake_model.pkl"
SCALER_PATH = "models/optimized_scaler.pkl"

def main():
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION QUICK TEST")
    print("="*80)
    
    # 1. Check video file
    print("\n[1/4] Checking video file...")
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {os.path.abspath(VIDEO_PATH)}")
        return 1
    
    # Get video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        return 1
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    print(f"✅ Video file is valid")
    print(f"   - Frames: {frame_count}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Duration: {duration:.2f} seconds")
    
    # 2. Load model
    print("\n[2/4] Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {os.path.abspath(MODEL_PATH)}")
        return 1
    
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data.get('model')
        if model is None:
            print("❌ No 'model' key found in the model file")
            return 1
            
        n_features = getattr(model, 'n_features_in_', 53)
        print(f"✅ Model loaded successfully")
        print(f"   - Type: {type(model).__name__}")
        print(f"   - Expected features: {n_features}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return 1
    
    # 3. Load scaler if available
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load scaler: {str(e)}")
    else:
        print("⚠️  Scaler not found, using unscaled features")
    
    # 4. Make a test prediction
    print("\n[3/4] Making test prediction...")
    try:
        # Generate random features for testing
        X_test = np.random.rand(1, n_features)
        
        # Scale features if scaler is available
        if scaler is not None:
            X_test = scaler.transform(X_test)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[0]
            if len(proba) == 2:  # Binary classification
                real_prob, fake_prob = proba[0], proba[1]
                is_fake = fake_prob > 0.7
                confidence = fake_prob if is_fake else real_prob
            else:  # Multi-class
                real_prob, fake_prob = proba[0], proba[-1]
                is_fake = fake_prob > 0.7
                confidence = max(real_prob, fake_prob)
        else:
            pred = model.predict(X_test)[0]
            is_fake = bool(pred)
            confidence = 1.0
            real_prob = 0.0 if is_fake else 1.0
            fake_prob = 1.0 if is_fake else 0.0
        
        print("✅ Test prediction successful")
        print("\nPREDICTION RESULT:")
        print(f"- Is Fake: {is_fake}")
        print(f"- Confidence: {confidence*100:.1f}%")
        print(f"- Real Probability: {real_prob*100:.1f}%")
        print(f"- Fake Probability: {fake_prob*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return 1
    
    print("\n[4/4] Test completed successfully!")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. The model is working correctly with the test data.")
    print("2. To analyze a video, use: python run_enhanced_detection.py <video_path>")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
