import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def check_model_features(model_path):
    """Check the features expected by the model."""
    print(f"\nChecking model: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model type: {type(model).__name__}")
        
        # Handle different model formats
        if isinstance(model, dict):
            print("Model is stored in a dictionary.")
            if 'model' in model:
                model = model['model']
                print(f"Actual model type: {type(model).__name__}")
        
        # Check for feature information
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
        
        if hasattr(model, 'feature_names_in_'):
            print("Feature names in model:", model.feature_names_in_)
        
        # Try to get feature names from model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'feature_names' in params:
                print(f"Feature names from params: {params['feature_names']}")
        
        return model
    
    except Exception as e:
        print(f"Error checking model: {e}")
        return None

def check_data_features(data_path):
    """Check the features in the data file."""
    print(f"\nChecking data: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Features ({len(df.columns)}): {list(df.columns[:5])}...")
        if len(df.columns) > 5:
            print(f"... and {len(df.columns) - 5} more features")
        return df.columns.tolist()
    except Exception as e:
        print(f"Error checking data: {e}")
        return []

def check_feature_alignment(model_path, data_path):
    """Check if the model's expected features match the data features."""
    model = check_model_features(model_path)
    if model is None:
        return
    
    data_features = check_data_features(data_path)
    
    # Try to get expected features from the model
    expected_features = None
    
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, 'get_params'):
        params = model.get_params()
        if 'feature_names' in params:
            expected_features = params['feature_names']
    
    if expected_features is not None:
        print("\nFeature Alignment:")
        print(f"- Expected features: {len(expected_features)}")
        print(f"- Actual features: {len(data_features)}")
        
        # Check for missing features
        missing = set(expected_features) - set(data_features)
        extra = set(data_features) - set(expected_features)
        
        if missing:
            print(f"\nMissing features ({len(missing)}): {list(missing)[:5]}")
            if len(missing) > 5:
                print(f"... and {len(missing) - 5} more missing features")
        
        if extra:
            print(f"\nExtra features in data ({len(extra)}): {list(extra)[:5]}")
            if len(extra) > 5:
                print(f"... and {len(extra) - 5} more extra features")
        
        if not missing and not extra:
            print("All features match!")
        elif not missing and extra:
            print("Warning: Data has extra features that the model doesn't expect")
        else:
            print("Error: Feature mismatch detected")
    else:
        print("\nCould not determine expected features from the model")

if __name__ == "__main__":
    model_path = "models/optimized_deepfake_model.pkl"
    data_path = "data/processed/X_test_cleaned.csv"
    
    check_feature_alignment(model_path, data_path)
