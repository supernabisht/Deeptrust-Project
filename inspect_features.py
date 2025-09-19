import os
import joblib
import pandas as pd
import numpy as np

def inspect_model(model_path):
    """Inspect the model to understand its structure and feature requirements."""
    print(f"\nInspecting model: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_names = model_data.get('feature_names')
            model_metadata = {k: v for k, v in model_data.items() 
                           if k not in ['model', 'feature_names']}
        else:
            model = model_data
            feature_names = None
            model_metadata = {}
        
        if model is None:
            print("Error: Model not found in the loaded file.")
            return
            
        # Print model information
        print(f"\nModel type: {type(model).__name__}")
        
        # Get expected number of features
        if hasattr(model, 'n_features_in_'):
            print(f"Expected number of features: {model.n_features_in_}")
        
        # Get feature names if available
        if feature_names is not None:
            print(f"\nModel has {len(feature_names)} feature names:")
            print(feature_names)
        elif hasattr(model, 'feature_names_in_'):
            print(f"\nModel has {len(model.feature_names_in_)} feature names from training:")
            print(model.feature_names_in_)
        else:
            print("\nNo feature names found in the model.")
        
        # Get model parameters
        if hasattr(model, 'get_params'):
            print("\nModel parameters:")
            for param, value in model.get_params().items():
                print(f"{param}: {value}")
        
        return model, feature_names, model_metadata
        
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        return None, None, None

def inspect_dataset(x_path, y_path):
    """Inspect the dataset to understand its features."""
    print(f"\nInspecting dataset: {x_path} and {y_path}")
    
    try:
        # Try to load aligned features if they exist
        aligned_x_path = x_path.replace('.csv', '_aligned.csv')
        if os.path.exists(aligned_x_path):
            print(f"Using aligned features from {aligned_x_path}")
            X = pd.read_csv(aligned_x_path)
        else:
            X = pd.read_csv(x_path)
        
        y = pd.read_csv(y_path, header=None, names=['label']).squeeze()
        
        # Print dataset information
        print(f"\nDataset shape: {X.shape}")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns)}")
        
        # Print class distribution
        print("\nClass distribution:")
        print(y.value_counts())
        
        return X, y
        
    except Exception as e:
        print(f"Error inspecting dataset: {str(e)}")
        return None, None

def main():
    # Find the latest model
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))
                  and not f.startswith('tmp')]
    
    if not model_files:
        print("No model files found in the models directory.")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), 
                    reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    
    # Inspect the model
    model, feature_names, model_metadata = inspect_model(model_path)
    
    # Inspect the dataset
    X_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'
    
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test, y_test = inspect_dataset(X_test_path, y_test_path)
        
        # If we have both model and dataset, check for feature alignment
        if model is not None and X_test is not None:
            expected_features = None
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
            elif feature_names is not None:
                expected_features = len(feature_names)
            
            if expected_features is not None:
                print(f"\nFEATURE ALIGNMENT:")
                print(f"Model expects {expected_features} features")
                print(f"Dataset has {X_test.shape[1]} features")
                
                if expected_features != X_test.shape[1]:
                    print("\nWARNING: Feature count mismatch!")
                    
                    # Try to find common features if feature names are available
                    if feature_names is not None and hasattr(X_test, 'columns'):
                        common_features = set(feature_names) & set(X_test.columns)
                        print(f"\nNumber of common features: {len(common_features)}")
                        
                        missing_in_data = set(feature_names) - set(X_test.columns)
                        if missing_in_data:
                            print(f"\nFeatures in model but not in data ({len(missing_in_data)}):")
                            print(list(missing_in_data)[:10])  # Print first 10
                            if len(missing_in_data) > 10:
                                print(f"... and {len(missing_in_data) - 10} more")
                        
                        extra_in_data = set(X_test.columns) - set(feature_names)
                        if extra_in_data:
                            print(f"\nFeatures in data but not in model ({len(extra_in_data)}):")
                            print(list(extra_in_data)[:10])  # Print first 10
                            if len(extra_in_data) > 10:
                                print(f"... and {len(extra_in_data) - 10} more")

if __name__ == "__main__":
    main()
