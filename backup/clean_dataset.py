import pandas as pd
import numpy as np

def clean_dataset(input_file, output_file):
    print(f"Reading dataset from {input_file}...")
    
    # Read the raw data with error handling
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Clean each line
    cleaned_lines = []
    for i, line in enumerate(lines):
        # Remove any extra commas and clean up the line
        line = line.strip()
        if not line:
            continue
            
        # Handle the last column (label)
        if line.endswith('REAL'):
            line = line[:-4] + 'REAL'
        elif line.endswith('FAKE'):
            line = line[:-4] + 'FAKE'
            
        cleaned_lines.append(line)
    
    # Write cleaned data to a temporary file
    temp_file = 'temp_cleaned.csv'
    with open(temp_file, 'w') as f:
        f.write('\n'.join(cleaned_lines))
    
    # Now try to read it with pandas
    try:
        df = pd.read_csv(temp_file)
        
        # Ensure label column is string type
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str).str.strip()
            
        # Convert all other columns to numeric
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any remaining NaN values with column means
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # Save the cleaned dataset
        df.to_csv(output_file, index=False)
        print(f"Successfully cleaned and saved dataset to {output_file}")
        
        # Print dataset info
        print("\nDataset Info:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    input_file = "balanced_deepfake_dataset.csv"
    output_file = "cleaned_balanced_deepfake_dataset.csv"
    
    if clean_dataset(input_file, output_file):
        print("\nYou can now use the cleaned dataset to train your model:")
        print(f"python run_advanced_detection.py --train --dataset {output_file} --model models/advanced_model.pkl")
    else:
        print("\nFailed to clean the dataset. Please check the input file format.")
