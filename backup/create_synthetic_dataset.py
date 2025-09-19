# create_synthetic_dataset.py - Enhanced version
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def create_enhanced_dataset():
    # Create more realistic synthetic data
    n_samples = 1000  # Reduced sample size for faster processing
    
    # Create column names to match the original dataset
    audio_features = [f'audio_feat_{i}' for i in range(25)]
    visual_features = [f'visual_feat_{i}' for i in range(15)]
    stats_features = [
        'audio_feats_mean', 'audio_feats_std', 'audio_feats_skew', 'audio_feats_kurtosis',
        'visual_feats_mean', 'visual_feats_std', 'visual_feats_skew', 'visual_feats_kurtosis'
    ]
    corr_features = [f'audio_visual_corr_{i+1}' for i in range(5)]
    
    all_columns = audio_features + visual_features + stats_features + corr_features
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(all_columns),
        n_informative=40,
        n_redundant=10,
        n_repeated=3,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],  # Balanced classes
        flip_y=0.05,  # Add some noise
        random_state=42
    )
    
    # Create DataFrame with proper column names
    df = pd.DataFrame(X, columns=all_columns)
    df['label'] = ['FAKE' if label == 1 else 'REAL' for label in y]
    
    # Add some realistic ranges to the data
    for col in audio_features + visual_features:
        df[col] = df[col] * 0.2  # Scale down the values
    
    # Ensure stats features have reasonable ranges
    df[stats_features] = df[stats_features].clip(-2, 2)
    
    # Ensure correlation features are between -1 and 1
    for col in corr_features:
        df[col] = df[col].clip(-1, 1)
    
    # Add some realistic variations
    df = add_variations(df)
    
    # Save enhanced dataset
    output_file = 'synthetic_deepfake_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Dataset created with {len(df)} samples and saved to {output_file}")
    
    # Verify the file was created
    if os.path.exists(output_file):
        print(f"✅ File verification: {output_file} exists and is {os.path.getsize(output_file)/1024:.1f} KB")
    else:
        print(f"❌ Error: Failed to create {output_file}")

    return df

def add_realistic_patterns(X, y):
    """Add realistic patterns to synthetic data"""
    # This function is no longer needed as we're handling patterns in the main function
    return X

def add_variations(df):
    """Add realistic variations to the dataset"""
    # Add some random noise to make the data more realistic
    for col in df.columns:
        if col != 'label':
            noise = np.random.normal(0, 0.01, len(df))
            df[col] = df[col] + noise
    return df