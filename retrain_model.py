import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_and_prepare_data():
    """Load and prepare the dataset for training"""
    print("Loading and preparing data...")
    
    # Path to the balanced dataset
    dataset_path = os.path.join('backup', 'balanced_deepfake_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return None
    
    print(f"Using dataset: {dataset_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print("Columns:", df.columns.tolist())
        
        # Check if we have the required columns
        required_columns = ['label']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in dataset. Need: {required_columns}")
            return None
        
        # Separate features and target
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Ensure we have at least 2 classes
        if len(y.unique()) < 2:
            print("Dataset must contain at least 2 classes")
            return None
        
        return X, y, X.columns.tolist()
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def train_model(X, y, feature_names):
    """Train a new model with the given data"""
    print("\nTraining new model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights safely
    class_ratio = sum(y==0)/max(1, sum(y==1))  # Prevent division by zero
    
    # Train a robust RandomForest model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_features='sqrt',  # Better generalization
        bootstrap=True,       # Enable bootstrapping
        oob_score=True,      # Use out-of-bag samples for validation
        verbose=1            # Show training progress
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, feature_names

def save_model(model, scaler, feature_names):
    """Save the trained model and related files"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('models', 'retrained_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    scaler_path = os.path.join('models', 'retrained_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the feature names
    features_path = os.path.join('models', 'retrained_features.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature names saved to {features_path}")
    
    return model_path, scaler_path, features_path

def main():
    print("Retraining DeepFake Detection Model")
    print("=" * 50)
    
    # Load and prepare data
    result = load_and_prepare_data()
    if result is None:
        return
    
    X, y, feature_names = result
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("Feature names:", feature_names)
    
    # Train the model
    model, scaler, feature_names = train_model(X, y, feature_names)
    
    # Save the model and related files
    model_path, scaler_path, features_path = save_model(model, scaler, feature_names)
    
    print("\nRetraining complete!")

if __name__ == "__main__":
    main()
