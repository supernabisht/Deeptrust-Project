import pandas as pd
import numpy as np

def load_dataset(filepath):
    print(f"\nLoading dataset from: {filepath}")
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Successfully loaded with {encoding}")
                return df
            except Exception as e:
                print(f"Failed with {encoding}: {str(e)[:100]}...")
        
        # If all encodings fail, try with low_memory=False
        print("Trying with low_memory=False")
        return pd.read_csv(filepath, low_memory=False)
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    filepath = 'consolidated_dataset_20250917_091948.csv'
    df = load_dataset(filepath)
    
    if df is not None:
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        print("\nColumn dtypes:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        
        if 'label' in df.columns:
            print("\nLabel distribution:")
            print(df['label'].value_counts())
