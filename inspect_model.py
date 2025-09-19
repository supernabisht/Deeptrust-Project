import joblib
import numpy as np
import pandas as pd
import os

def inspect_model(model_path):
    """Inspect the model to understand its structure and feature requirements."""
    print(f"\nInspecting model: {model_path}")
    
    try:
        # Try to load the model
        model = joblib.load(model_path)
        print(f"\nModel type: {type(model)}")
        
        # Check if it's a scikit-learn model
        if hasattr(model, 'n_features_in_'):
            print(f"Expected number of features: {model.n_features_in_}")
            
        if hasattr(model, 'feature_names_in_'):
            print("Feature names in model:", model.feature_names_in_)
        
        # If it's a dictionary with model metadata
        if isinstance(model, dict):
            print("\nModel dictionary keys:", list(model.keys()))
            
            if 'feature_names' in model:
                print(f"\nFeature names ({len(model['feature_names'])}):")
                print(model['feature_names'])
                
            if 'model' in model:
                print("\nModel object type:", type(model['model']))
                if hasattr(model['model'], 'n_features_in_'):
                    print(f"Model expects {model['model'].n_features_in_} features")
        
        # Try to make a prediction with dummy data
        try:
            dummy_data = np.ones((1, model.n_features_in_ if hasattr(model, 'n_features_in_') else 100))
            pred = model.predict(dummy_data)
            print(f"\nSuccessfully made a prediction with dummy data. Output shape: {pred.shape}")
        except Exception as e:
            print(f"\nError making prediction with dummy data: {str(e)}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def find_model_files(directory='models'):
    """Find all model files in the specified directory."""
    model_extensions = ('.pkl', '.joblib', '.pkl.gz', '.joblib.gz')
    model_files = []
    
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(model_extensions):
                    model_files.append(os.path.join(root, file))
    
    return model_files

if __name__ == "__main__":
    # Find all model files
    model_files = find_model_files()
    
    if not model_files:
        print("No model files found in the 'models' directory.")
    else:
        print(f"Found {len(model_files)} model files:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        
        # Inspect the first model by default
        if model_files:
            print("\n" + "="*50)
            inspect_model(model_files[0])
