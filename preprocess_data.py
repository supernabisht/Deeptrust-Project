import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Preprocess the dataset by handling missing values and scaling features.
    
    Args:
        data_path (str): Path to the dataset CSV file
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Clean the data
    print("Cleaning data...")
    
    # Convert target to numerical if it's categorical
    if df.iloc[:, -1].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df.iloc[:, -1])
        joblib.dump(le, 'models/label_encoder.joblib')
    else:
        y = df.iloc[:, -1]
    
    # Handle features
    X = df.iloc[:, :-1]  # All columns except last
    
    # Convert all feature columns to numeric, coercing errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values (using mean imputation)
    print("Handling missing values...")
    
    # First, drop columns with too many missing values (>50%)
    missing_cols = X.columns[X.isnull().mean() > 0.5]
    if len(missing_cols) > 0:
        print(f"Dropping columns with >50% missing values: {missing_cols.tolist()}")
        X = X.drop(columns=missing_cols)
    
    # Impute remaining missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Convert back to DataFrame to maintain column names
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Save the imputer for later use
    joblib.dump(imputer, 'models/imputer.joblib')
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Split into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("\n=== Data Preprocessing Complete ===")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Preprocess the data
    preprocess_data('consolidated_dataset_20250917_091948.csv')
