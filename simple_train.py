import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
try:
    # Load the dataset
    df = pd.read_csv('consolidated_dataset_20250917_091948.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label'].map({'REAL': 1, 'FAKE': 0})  # Convert to binary
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train a simple model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel trained successfully!")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save the model
    model_path = 'simple_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()
