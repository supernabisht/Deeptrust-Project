import pandas as pd
import numpy as np

def prepare_dataset(input_file, output_file):
    print(f"Preparing dataset from {input_file}...")
    
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Ensure all numeric columns are properly formatted
    for col in df.columns:
        if col != 'label':
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN values with column means
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Ensure labels are properly formatted
    df['label'] = df['label'].apply(lambda x: 'FAKE' if str(x).strip().upper() == 'FAKE' else 'REAL')
    
    # Save the prepared dataset
    df.to_csv(output_file, index=False)
    
    # Verify the dataset
    print("\nPrepared Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"NaN values: {df.isna().sum().sum()}")
    print(f"Infinite values: {df.isin([np.inf, -np.inf]).sum().sum()}")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    return df

if __name__ == "__main__":
    input_file = "simple_deepfake_dataset.csv"
    output_file = "prepared_dataset.csv"
    
    df = prepare_dataset(input_file, output_file)
    print(f"\n✅ Dataset prepared and saved to {output_file}")
    print("\nYou can now train the model with:")
    print(f"python run_advanced_detection.py --train --dataset {output_file} --model models/advanced_model.pkl")
