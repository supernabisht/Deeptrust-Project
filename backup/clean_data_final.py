import pandas as pd
import numpy as np

def clean_dataset(input_file, output_file):
    print("Starting dataset cleaning...")
    
    # Read the file line by line
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Process header
    header = lines[0].strip()
    num_columns = len(header.split(','))
    print(f"Expected number of columns: {num_columns}")
    
    # Process data rows
    cleaned_rows = []
    for i, line in enumerate(lines[1:], 1):  # Skip header
        line = line.strip()
        if not line:
            continue
            
        # Split the line
        parts = line.split(',')
        
        # Handle label (last column)
        if len(parts) >= num_columns:
            # If we have enough parts, ensure the last one is the label
            label = parts[-1].strip().upper()
            if 'FAKE' in label:
                label = 'FAKE'
            elif 'REAL' in label:
                label = 'REAL'
            else:
                label = 'UNKNOWN'
                
            # Rebuild the row with proper number of columns
            row = parts[:num_columns-1] + [label]
            cleaned_rows.append(','.join(row))
    
    # Write cleaned data to file
    with open(output_file, 'w') as f:
        f.write(header + '\n')  # Write header
        f.write('\n'.join(cleaned_rows))  # Write cleaned rows
    
    print(f"\nCleaned dataset saved to {output_file}")
    
    # Verify the cleaned dataset
    try:
        df = pd.read_csv(output_file)
        print("\nCleaned Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Check for any remaining issues
        print("\nChecking for remaining issues...")
        print(f"NaN values: {df.isna().sum().sum()}")
        print(f"Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        
        # If there are still issues, fix them
        if df.isna().sum().sum() > 0 or np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
            print("\nFixing remaining numeric issues...")
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].mean())
            
            # Save the final cleaned version
            df.to_csv(output_file, index=False)
            print("\nFinal cleaned dataset saved with all issues resolved.")
            
    except Exception as e:
        print(f"Error verifying dataset: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    input_file = "balanced_deepfake_dataset.csv"
    output_file = "final_cleaned_dataset.csv"
    
    if clean_dataset(input_file, output_file):
        print("\nDataset cleaning completed successfully!")
        print("\nYou can now use the cleaned dataset to train your model:")
        print(f"python run_advanced_detection.py --train --dataset {output_file} --model models/advanced_model.pkl")
    else:
        print("\nFailed to clean the dataset. Please check the input file format.")
