import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

def clean_test_data(x_path, y_path, output_dir='data/processed'):
    """
    Clean and preprocess test data by handling missing values.
    
    Args:
        x_path (str): Path to test features CSV file
        y_path (str): Path to test labels CSV file
        output_dir (str): Directory to save cleaned data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading test data from {x_path} and {y_path}...")
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path, header=None, names=['label']).squeeze()
    
    # Convert labels to binary (0/1)
    if y_test.dtype == 'object':
        y_test = y_test.map({'REAL': 0, 'FAKE': 1, 'real': 0, 'fake': 1, 0: 0, 1: 1})
    
    # Ensure X and y have the same number of samples
    if len(X_test) != len(y_test):
        min_len = min(len(X_test), len(y_test))
        X_test = X_test.iloc[:min_len]
        y_test = y_test.iloc[:min_len]
        print(f"Adjusted dataset size to {min_len} samples")
    
    # Convert all feature columns to numeric, coercing errors to NaN
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Handle missing values (using mean imputation)
    print("Handling missing values...")
    
    # First, drop columns with too many missing values (>50%)
    missing_cols = X_test.columns[X_test.isnull().mean() > 0.5]
    if len(missing_cols) > 0:
        print(f"Dropping columns with >50% missing values: {missing_cols.tolist()}")
        X_test = X_test.drop(columns=missing_cols)
    
    # Impute remaining missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_test_imputed = imputer.fit_transform(X_test)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    
    # Ensure we have exactly 84 features (to match the model's expectation)
    if X_test.shape[1] > 84:
        print(f"Reducing features from {X_test.shape[1]} to 84")
        X_test = X_test.iloc[:, :84]
    elif X_test.shape[1] < 84:
        print(f"Padding features from {X_test.shape[1]} to 84 with zeros")
        padding = np.zeros((len(X_test), 84 - X_test.shape[1]))
        padding_cols = [f'padding_{i}' for i in range(84 - X_test.shape[1])]
        padding_df = pd.DataFrame(padding, columns=padding_cols, index=X_test.index)
        X_test = pd.concat([X_test, padding_df], axis=1)
    
    # Save cleaned data
    x_output_path = os.path.join(output_dir, 'X_test_cleaned.csv')
    y_output_path = os.path.join(output_dir, 'y_test_cleaned.csv')
    
    X_test.to_csv(x_output_path, index=False)
    y_test.to_csv(y_output_path, index=False, header=False)
    
    print(f"\nCleaned test data saved to:")
    print(f"- Features: {x_output_path}")
    print(f"- Labels: {y_output_path}")
    print(f"\nFinal shape: {X_test.shape}")
    print(f"Missing values: {X_test.isnull().sum().sum()}")
    print(f"Class distribution:\n{y_test.value_counts()}")

if __name__ == "__main__":
    # Define paths
    x_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'
    
    # Clean the test data
    clean_test_data(x_test_path, y_test_path)
