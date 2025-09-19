import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

def inspect_model(model_path):
    """Inspect the model to understand its structure and feature requirements."""
    print(f"\nInspecting model: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model type: {type(model).__name__}")
        
        # Check for common model attributes
        if hasattr(model, 'n_features_in_'):
            print(f"Expected number of features: {model.n_features_in_}")
        
        if hasattr(model, 'feature_names_in_'):
            print("Feature names in model:", model.feature_names_in_)
        
        if hasattr(model, 'feature_importances_'):
            print(f"Number of feature importances: {len(model.feature_importances_)}")
        
        # If it's a dictionary with model and metadata
        if isinstance(model, dict):
            print("\nModel dictionary keys:", list(model.keys()))
            if 'model' in model:
                print("\nModel object type:", type(model['model']))
                if hasattr(model['model'], 'n_features_in_'):
                    print(f"Model expects {model['model'].n_features_in_} features")
            if 'feature_names' in model:
                print(f"\nFeature names in model ({len(model['feature_names'])}):")
                print(model['feature_names'])
        
        return model
    
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        return None

def inspect_dataset(csv_path):
    """Inspect the dataset to understand its structure."""
    print(f"\nInspecting dataset: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 feature names:")
        print(df.columns.tolist()[:5])
        print("\nLast 5 feature names:")
        print(df.columns.tolist()[-5:])
        
        # Basic statistics
        print("\nBasic statistics:")
        print(df.describe().T[['min', 'max', 'mean', 'std']].head())
        
        return df
    
    except Exception as e:
        print(f"Error inspecting dataset: {str(e)}")
        return None

def fix_feature_mismatch(model, X):
    """Fix feature mismatch between model and dataset."""
    if not hasattr(model, 'n_features_in_'):
        print("\nModel doesn't specify expected number of features. Cannot fix mismatch.")
        return X
    
    expected_features = model.n_features_in_
    actual_features = X.shape[1]
    
    print(f"\nFixing feature mismatch:")
    print(f"- Expected features: {expected_features}")
    print(f"- Actual features: {actual_features}")
    
    if expected_features == actual_features:
        print("No mismatch detected.")
        return X
    
    # If we have feature names, try to align them
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    elif hasattr(model, 'feature_names'):
        feature_names = model.feature_names
    
    if feature_names is not None and len(feature_names) == expected_features:
        print(f"Aligning using {len(feature_names)} feature names")
        # Create a new DataFrame with expected features
        aligned_X = pd.DataFrame(columns=feature_names)
        
        # Fill in matching columns
        for col in feature_names:
            if col in X.columns:
                aligned_X[col] = X[col].values
            else:
                print(f"  - Warning: Feature '{col}' not found in dataset. Filling with zeros.")
                aligned_X[col] = 0.0
        
        return aligned_X
    
    # If no feature names, try to match by position
    print("No feature names available. Aligning by position with zero padding/trimming.")
    
    if actual_features < expected_features:
        # Pad with zeros
        padding = np.zeros((X.shape[0], expected_features - actual_features))
        X_aligned = np.hstack([X.values, padding])
        print(f"  - Added {expected_features - actual_features} zero columns")
    else:
        # Trim extra features
        X_aligned = X.values[:, :expected_features]
        print(f"  - Trimmed {actual_features - expected_features} columns")
    
    return pd.DataFrame(X_aligned, columns=[f'feature_{i}' for i in range(expected_features)])

def main():
    # Find the most recent model
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))
                  and not f.startswith('tmp')]
    
    if not model_files:
        print("No model files found in the 'models' directory.")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), 
                    reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    
    # Inspect the model
    model = inspect_model(model_path)
    if model is None:
        return
    
    # Get the actual model object if it's inside a dictionary
    model_obj = model['model'] if isinstance(model, dict) and 'model' in model else model
    
    # Find and inspect the test dataset
    X_test_path = 'data/processed/X_test.csv'
    if not os.path.exists(X_test_path):
        print(f"\nTest data not found at {X_test_path}")
        return
    
    X_test = inspect_dataset(X_test_path)
    if X_test is None:
        return
    
    # Fix feature mismatch
    X_test_fixed = fix_feature_mismatch(model_obj, X_test)
    
    # Save the fixed dataset
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'X_test_fixed.csv')
    X_test_fixed.to_csv(output_path, index=False)
    
    print(f"\nFixed test data saved to: {output_path}")
    print(f"New shape: {X_test_fixed.shape}")
    
    # Also save a copy of the feature names if available
    feature_names = None
    if hasattr(model_obj, 'feature_names_in_'):
        feature_names = model_obj.feature_names_in_
    elif hasattr(model_obj, 'feature_names'):
        feature_names = model_obj.feature_names
    elif hasattr(model, 'feature_names'):  # Check the outer dictionary
        feature_names = model['feature_names']
    
    if feature_names is not None:
        with open(os.path.join(output_dir, 'model_feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
        print(f"Feature names saved to: {output_dir}/model_feature_names.txt")

if __name__ == "__main__":
    main()
