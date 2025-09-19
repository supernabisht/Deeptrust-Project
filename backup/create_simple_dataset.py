import pandas as pd
import numpy as np
import os

def create_simple_dataset():
    print("Creating a simple synthetic deepfake dataset...")
    
    # Number of samples
    n_samples = 1000
    
    # Create column names
    audio_features = [f'audio_feat_{i}' for i in range(25)]
    visual_features = [f'visual_feat_{i}' for i in range(15)]
    stats_features = [
        'audio_feats_mean', 'audio_feats_std', 'audio_feats_skew', 'audio_feats_kurtosis',
        'visual_feats_mean', 'visual_feats_std', 'visual_feats_skew', 'visual_feats_kurtosis'
    ]
    corr_features = [f'audio_visual_corr_{i+1}' for i in range(5)]
    
    all_columns = audio_features + visual_features + stats_features + corr_features
    
    # Create random data
    np.random.seed(42)
    data = np.random.randn(n_samples, len(all_columns)) * 0.1  # Small random values
    
    # Create labels (50% FAKE, 50% REAL)
    labels = ['FAKE' if i < n_samples//2 else 'REAL' for i in range(n_samples)]
    np.random.shuffle(labels)  # Shuffle the labels
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=all_columns)
    df['label'] = labels
    
    # Save to file
    output_file = 'simple_deepfake_dataset.csv'
    df.to_csv(output_file, index=False)
    
    # Verify file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024  # Size in KB
        print(f"✅ Successfully created {output_file} ({file_size:.1f} KB)")
        print(f"✅ Dataset shape: {df.shape}")
        print(f"✅ Class distribution:\n{df['label'].value_counts()}")
    else:
        print("❌ Failed to create the dataset file")
    
    return df

if __name__ == "__main__":
    create_simple_dataset()
