import os
import sys
import joblib
import numpy as np

def test_model_loading():
    print("\nTesting model loading...")
    model_path = "models/optimized_deepfake_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {os.path.abspath(model_path)}")
        return False
    
    try:
        print(f"Loading model from: {os.path.abspath(model_path)}")
        model_data = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Check model components
        model = model_data.get('model')
        if model is None:
            print("❌ No 'model' key found in the model file")
            return False
            
        print("\nModel information:")
        print(f"- Model type: {type(model).__name__}")
        
        # Get number of features expected by the model
        n_features = getattr(model, 'n_features_in_', None)
        if n_features is not None:
            print(f"- Number of features expected: {n_features}")
        else:
            print("⚠️  Could not determine number of expected features")
        
        # Test prediction with dummy data
        print("\nTesting prediction with dummy data...")
        n_samples = 1
        n_features_actual = n_features if n_features is not None else 53  # Default to 53 if not available
        X_test = np.random.rand(n_samples, n_features_actual)
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                print(f"✅ Prediction successful with {X_test.shape[1]} features")
                print(f"   Prediction probabilities shape: {proba.shape}")
                print(f"   Probabilities: {proba}")
            elif hasattr(model, 'predict'):
                pred = model.predict(X_test)
                print(f"✅ Prediction successful with {X_test.shape[1]} features")
                print(f"   Predictions: {pred}")
            else:
                print("❌ Model doesn't support predict or predict_proba methods")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"\n{'='*80}")
    print(f"DEEPFAKE DETECTION MODEL TEST")
    print(f"{'='*80}\n")
    
    # Check if models directory exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found at {os.path.abspath(models_dir)}")
        print("\nPlease ensure you have the model files in the 'models' directory.")
        print("The following files are required:")
        print("- optimized_deepfake_model.pkl")
        print("- optimized_scaler.pkl (optional but recommended)\n")
        return 1
    
    # List files in models directory
    print("Files in models directory:")
    for f in os.listdir(models_dir):
        print(f"- {f} ({os.path.getsize(os.path.join(models_dir, f)) / 1024:.1f} KB)")
    
    # Test model loading
    success = test_model_loading()
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY:")
    print(f"- Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*80}\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
