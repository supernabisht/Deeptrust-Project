import os
import joblib
import pandas as pd
import numpy as np

# Paths
X_train_path = 'data/processed/X_train.csv'
X_test_path = 'data/processed/X_test.csv'
model_path = 'models/latest_model.pkl'

def load_model(model_path):
    """Load the model and return its feature names."""
    print(f"Loading model from: {model_path}")
    try:
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_names = model_data.get('feature_names')
        else:
            model = model_data
            feature_names = None
        
        if model is None:
            print("Error: Model not found in the loaded file.")
            return None, None
            
        # Get feature names from model if available
        if feature_names is None and hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        
        return model, feature_names
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def fix_dataset(x_path, feature_names, output_suffix='_fixed'):
    """Fix the dataset to match the model's expected features."""
    print(f"\nProcessing dataset: {x_path}")
    
    try:
        # Load the dataset
        X = pd.read_csv(x_path)
        print(f"Original shape: {X.shape}")
        
        # If no feature names are provided, use the ones from the dataset
        if feature_names is None:
            print("No feature names provided. Using dataset features as-is.")
            return X
        
        # Create a new DataFrame with the expected features
        X_fixed = pd.DataFrame(columns=feature_names)
        
        # Fill in matching columns
        missing_features = []
        for col in feature_names:
            if col in X.columns:
                X_fixed[col] = X[col].values
            else:
                missing_features.append(col)
                X_fixed[col] = 0.0  # Fill missing with zeros
        
        # Print information about the fix
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in dataset. Filled with zeros.")
            print("First 5 missing features:", missing_features[:5])
        
        # Add any extra features from the dataset (with warning)
        extra_features = set(X.columns) - set(feature_names)
        if extra_features:
            print(f"Warning: {len(extra_features)} extra features in dataset will be dropped.")
            print("First 5 extra features:", list(extra_features)[:5])
        
        print(f"Fixed shape: {X_fixed.shape}")
        
        # Save the fixed dataset
        output_path = x_path.replace('.csv', f'{output_suffix}.csv')
        X_fixed.to_csv(output_path, index=False)
        print(f"Saved fixed dataset to: {output_path}")
        
        return X_fixed
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None

def main():
    # Load the model to get feature names
    model, feature_names = load_model(model_path)
    if model is None:
        print("Failed to load model.")
        return
    
    print(f"\nModel expects {len(feature_names) if feature_names is not None else 'unknown'} features")
    if feature_names is not None:
        print("First 5 features:", feature_names[:5])
    
    # Process training and test datasets
    for x_path in [X_train_path, X_test_path]:
        if os.path.exists(x_path):
            fix_dataset(x_path, feature_names)
        else:
            print(f"\nDataset not found: {x_path}")
    
    print("\nFeature fixing complete!")

if __name__ == "__main__":
    main()
