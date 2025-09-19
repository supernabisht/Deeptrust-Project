import pandas as pd
import sys

try:
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('consolidated_dataset_20250917_091948.csv')
    
    # Basic information
    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum().sum(), "total missing values")
    
    # Check class distribution (assuming last column is the target)
    target_col = df.columns[-1]
    print("\n=== Class Distribution ===")
    print(df[target_col].value_counts())
    
    # Display first few rows
    print("\n=== First 5 Rows ===")
    print(df.head())
    
except Exception as e:
    print(f"\nError: {str(e)}", file=sys.stderr)
