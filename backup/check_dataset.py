# check_dataset.py
import pandas as pd
import numpy as np
import sys

def analyze_dataset(file_path):
    try:
        print(f"🔍 Analyzing dataset: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"📊 Shape: {df.shape} (rows, columns)")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"❌ Missing values: {df.isnull().sum().sum()}")
        print(f"📈 Data types:\n{df.dtypes.value_counts()}")
        
        # Check for target variable (assuming it's 'label' or similar)
        target_cols = [col for col in df.columns if 'label' in col.lower() or 'target' in col.lower() or 'class' in col.lower()]
        if target_cols:
            print(f"🎯 Target variable found: {target_cols[0]}")
            print(f"📊 Class distribution:\n{df[target_cols[0]].value_counts()}")
        else:
            print("⚠ No obvious target variable found")
            
        # Check first few rows
        print("\n👀 First 3 rows:")
        print(df.head(3))
        
        # Basic stats
        print("\n📊 Basic statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_dataset(sys.argv[1])
    else:
        print("Please provide dataset filename as argument")
        print("Usage: python check_dataset.py your_dataset.csv")