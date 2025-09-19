import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_feature_names(model):
    """Extract feature names from the model if available."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif isinstance(model, dict) and 'feature_names' in model:
        return model['feature_names']
    elif hasattr(model, 'feature_importances_'):
        # If we have feature importances but no names, generate generic names
        return [f'feature_{i}' for i in range(len(model.feature_importances_))]
    elif hasattr(model, 'n_features_in_'):
        # If we know the number of features but not the names
        return [f'feature_{i}' for i in range(model.n_features_in_)]
    return None

def align_dataset_to_model(X, model):
    """Align dataset features to match model's expected features."""
    # Get model's expected feature names
    model_features = get_feature_names(model)
    
    if model_features is None:
        print("Warning: Could not determine model's expected features. Using dataset features as-is.")
        return X
    
    print(f"Model expects {len(model_features)} features")
    print(f"Dataset has {X.shape[1]} features")
    
    # If the model is a dictionary with a 'model' key, use that for feature alignment
    if isinstance(model, dict) and 'model' in model:
        return align_dataset_to_model(X, model['model'])
    
    # Create a new DataFrame with the model's expected features
    aligned_X = pd.DataFrame(columns=model_features)
    
    # Fill in matching columns
    for col in model_features:
        if col in X.columns:
            aligned_X[col] = X[col].values
        else:
            print(f"Warning: Feature '{col}' not found in dataset. Filling with zeros.")
            aligned_X[col] = 0.0
    
    return aligned_X

def main():
    # Find the most recent model
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) 
                  if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))]
    
    if not model_files:
        print("No model files found in the 'models' directory.")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"Using model: {model_path}")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Load the test data
        X_test_path = 'data/processed/X_test.csv'
        y_test_path = 'data/processed/y_test.csv'
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print("Test data not found. Please run prepare_dataset.py first.")
            return
        
        print("\nLoading test data...")
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path, header=None, names=['label']).squeeze()
        
        print(f"Original test data shape: {X_test.shape}")
        
        # Align features
        print("\nAligning features...")
        X_test_aligned = align_dataset_to_model(X_test, model)
        
        # Save aligned features
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        X_test_aligned.to_csv(os.path.join(output_dir, 'X_test_aligned.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test_aligned.csv'), index=False, header=False)
        
        print(f"\nAligned test data shape: {X_test_aligned.shape}")
        print(f"Aligned features saved to {output_dir}/X_test_aligned.csv")
        
        # If we have a model with feature names, save them for reference
        feature_names = get_feature_names(model)
        if feature_names:
            with open(os.path.join(output_dir, 'model_feature_names.txt'), 'w') as f:
                f.write('\n'.join(feature_names))
            print(f"Model feature names saved to {output_dir}/model_feature_names.txt")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
