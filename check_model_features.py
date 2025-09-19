import pickle
import numpy as np

def check_model():
    print("Checking Model and Scaler Feature Dimensions")
    print("=" * 50)
    
    # Paths to model files
    model_path = "models/optimized_deepfake_model_20250917_084729.pkl"
    scaler_path = "models/optimized_scaler.pkl"
    features_path = "models/selected_features.txt"
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
        
        print(f"\nModel type: {type(model).__name__}")
        
        # Get number of features expected by the model
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
        elif hasattr(model, 'n_features'):
            print(f"Model expects {model.n_features} features")
        else:
            print("Could not determine number of features from model")
        
        # Get number of features in scaler
        if hasattr(scaler, 'n_features_in_'):
            print(f"Scaler expects {scaler.n_features_in_} features")
        else:
            print("Could not determine number of features from scaler")
        
        print(f"Number of features in selected_features.txt: {len(feature_names)}")
        
        # Print first 10 feature names
        print("\nFirst 10 features:")
        for i, feat in enumerate(feature_names[:10]):
            print(f"  {i+1}. {feat}")
        
        # Create a test feature vector
        test_features = np.zeros((1, len(feature_names)))
        
        # Try to scale the features
        try:
            scaled_features = scaler.transform(test_features)
            print(f"\n✅ Successfully scaled features to shape: {scaled_features.shape}")
        except Exception as e:
            print(f"\n❌ Error scaling features: {str(e)}")
        
        # Try to make a prediction
        try:
            prediction = model.predict(test_features)
            print(f"✅ Successfully made prediction: {prediction}")
        except Exception as e:
            print(f"\n❌ Error making prediction: {str(e)}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    check_model()
