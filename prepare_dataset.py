import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_dataset(csv_path: str, target_column: str = 'label', test_size: float = 0.2, random_state: int = 42):
    """
    Prepare dataset for training and evaluation.
    
    Args:
        csv_path: Path to the CSV file
        target_column: Name of the target column
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if target column exists
    if target_column not in df.columns:
        print("Target column not found. Available columns:", df.columns.tolist())
        return
    
    print("\nDataset Info:")
    print("------------")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Target distribution:\n{df[target_column].value_counts()}")
    
    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the splits
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=False)
    y_test.to_csv('data/processed/y_test.csv', index=False, header=False)
    
    print("\nDataset prepared successfully!")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Saved to 'data/processed/'")

if __name__ == "__main__":
    # Update this path if your CSV is in a different location
    csv_path = 'consolidated_dataset_20250917_091948.csv'
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        print("Please make sure the CSV file exists in the current directory.")
    else:
        prepare_dataset(csv_path, target_column='label')
