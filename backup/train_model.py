import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(dataset_path):
    """Load and prepare the dataset"""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Convert labels to binary (0 for REAL, 1 for FAKE)
    y = (y == 'FAKE').astype(int)
    
    return X, y

def train_model(X, y):
    """Train a simple Random Forest model"""
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    return model

def save_model(model, model_path):
    """Save the trained model"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")

def main():
    # Configuration
    dataset_path = 'prepared_dataset.csv'
    model_path = 'models/advanced_model.pkl'
    
    try:
        # Load and prepare data
        X, y = load_data(dataset_path)
        
        # Train model
        model = train_model(X, y)
        
        # Save model
        save_model(model, model_path)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting model training...")
    if main():
        print("\n🎉 Model training completed successfully!")
    else:
        print("\n❌ Model training failed.")
