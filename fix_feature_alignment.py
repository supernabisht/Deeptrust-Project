import os
import joblib
import pandas as pd
import numpy as np

def align_features(X, model_feature_names):
    """
    Align the features in X to match the model's expected feature names.
    
    Args:
        X: Input features (DataFrame or array-like)
        model_feature_names: List of feature names expected by the model
        
    Returns:
        DataFrame with aligned features
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        # Convert numpy array to DataFrame
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Create a new DataFrame with the expected feature names
    aligned_X = pd.DataFrame(columns=model_feature_names)
    
    # Fill in the values for features that exist in both
    for col in model_feature_names:
        if col in X_df.columns:
            aligned_X[col] = X_df[col].values
        else:
            print(f"Warning: Feature '{col}' not found in input data. Filling with zeros.")
            aligned_X[col] = 0.0
    
    return aligned_X

def main():
    # Load the model to get feature names
    model_path = 'models/advanced_model.pkl'
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model_feature_names = model_data.get('feature_names')
            if model_feature_names is None:
                print("Warning: No feature names found in the model. Using default names.")
                # If no feature names, we'll assume the features are in the same order
                model_feature_names = [f'feature_{i}' for i in range(85)]  # Adjust based on your model
        else:
            print("Warning: Model format not recognized. Using default feature names.")
            model_feature_names = [f'feature_{i}' for i in range(85)]  # Adjust based on your model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Process training data
    print("Processing training data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    aligned_X_train = align_features(X_train, model_feature_names)
    aligned_X_train.to_csv('data/processed/X_train_aligned.csv', index=False)
    
    # Process test data
    print("Processing test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    aligned_X_test = align_features(X_test, model_feature_names)
    aligned_X_test.to_csv('data/processed/X_test_aligned.csv', index=False)
    
    print("\nFeature alignment complete!")
    print(f"Original training features: {X_train.shape[1]}")
    print(f"Aligned training features: {aligned_X_train.shape[1]}")
    print(f"\nAligned feature names (first 10): {model_feature_names[:10]}...")

if __name__ == "__main__":
    main()
