import pandas as pd
import numpy as np

def fix_dataset(input_file, output_file):
    print("Fixing dataset format...")
    
    # Read the raw file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split into lines and clean each line
    lines = content.strip().split('\n')
    cleaned_lines = []
    
    # Get header
    header = lines[0]
    cleaned_lines.append(header)
    
    # Process data rows
    for line in lines[1:]:
        # Remove any extra spaces and split by comma
        parts = [p.strip() for p in line.split(',')]
        
        # The last part should be either 'FAKE' or 'REAL'
        if len(parts) < 2:
            continue
            
        label = parts[-1].upper()
        if label not in ['FAKE', 'REAL']:
            # Try to extract label from the end of the string
            if 'FAKE' in label:
                label = 'FAKE'
            elif 'REAL' in label:
                label = 'REAL'
            else:
                label = 'UNKNOWN'
        
        # Rebuild the line with proper label
        cleaned_line = ','.join(parts[:-1] + [label])
        cleaned_lines.append(cleaned_line)
    
    # Save the cleaned data
    with open(output_file, 'w') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Saved fixed dataset to {output_file}")
    return output_file

def clean_numeric_data(input_file, output_file):
    print("\nCleaning numeric data...")
    
    # Read the fixed dataset
    df = pd.read_csv(input_file)
    
    # Convert all numeric columns
    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with column means
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    
    # Print dataset info
    print("\nCleaned Dataset Info:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return output_file

if __name__ == "__main__":
    input_file = "balanced_deepfake_dataset.csv"
    fixed_file = "fixed_balanced_deepfake_dataset.csv"
    output_file = "cleaned_balanced_deepfake_dataset.csv"
    
    # First fix the format
    fixed_file = fix_dataset(input_file, fixed_file)
    
    # Then clean the numeric data
    clean_numeric_data(fixed_file, output_file)
    
    print("\nYou can now use the cleaned dataset to train your model:")
    print(f"python run_advanced_detection.py --train --dataset {output_file} --model models/advanced_model.pkl")
