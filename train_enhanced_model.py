import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def load_and_preprocess_data(csv_path: str) -> tuple:
    """Load and preprocess the dataset"""
    df = pd.read_csv(csv_path)
    
    # Convert string representations of lists to actual lists
    for col in df.columns:
        if 'mfcc' in col or 'lip_movement' in col:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Flatten MFCC features
    mfcc_features = []
    for idx, row in df.iterrows():
        mfcc_mean = row['mfcc_mean'] if 'mfcc_mean' in row else [0]*13
        mfcc_std = row['mfcc_std'] if 'mfcc_std' in row else [0]*13
        features = mfcc_mean + mfcc_std
        features.extend([
            row.get('zero_crossing_rate', 0),
            row.get('spectral_centroid', 0),
            row.get('spectral_bandwidth', 0),
            row.get('lip_movement_mean', 0),
            row.get('lip_movement_std', 0)
        ])
        mfcc_features.append(features)
    
    X = np.array(mfcc_features)
    y = df['label'].values  # Assuming 'label' column exists with 0/1 for fake/real
    
    return X, y

def train_model():
    # Load and preprocess data
    X, y = load_and_preprocess_data('consolidated_dataset_20250917_091948.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, 'models/enhanced_deepfake_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model()