# Save this as create_proper_scaler.py
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_proper_scaler():
    print("=== CREATING PROPER SCALER ===")
    
    # Load cleaned dataset
    df = pd.read_csv('cleaned_deepfake_dataset.csv')
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Separate features and labels
    feature_columns = df.columns[:-1]  # All columns except the last (label)
    features = df[feature_columns].values
    
    print(f"📊 Using {len(feature_columns)} feature columns")
    print(f"📈 Feature matrix shape: {features.shape}")
    print(f"🔢 First sample features: {features[0][:5]}...")
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(features)
    
    # Save the scaler
    with open('proper_feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Proper scaler created for {scaler.n_features_in_} features")
    print("✅ Saved as 'proper_feature_scaler.pkl'")
    
    return scaler

if __name__ == "__main__":
    create_proper_scaler()