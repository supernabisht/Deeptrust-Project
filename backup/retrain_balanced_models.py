#!/usr/bin/env python3
"""
Retrain Models with Balanced Dataset
Uses the balanced dataset to train accurate REAL/FAKE classification models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import os
from datetime import datetime

def retrain_models():
    """Retrain models with balanced dataset"""
    print("Retraining models with balanced dataset...")
    
    # Load balanced dataset
    try:
        df = pd.read_csv('balanced_deepfake_dataset.csv')
        print(f"Loaded balanced dataset: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
    except Exception as e:
        print(f"Error loading balanced dataset: {e}")
        return False
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=200, depth=6, random_state=42, verbose=False)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Test Accuracy: {accuracy:.4f}")
            
            # Detailed classification report
            report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
            print(f"Precision - FAKE: {report['FAKE']['precision']:.4f}, REAL: {report['REAL']['precision']:.4f}")
            print(f"Recall - FAKE: {report['FAKE']['recall']:.4f}, REAL: {report['REAL']['recall']:.4f}")
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Save the best model and all components
    if best_model is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("models", exist_ok=True)
        
        # Save best model
        model_path = f"models/optimized_deepfake_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': best_model}, f)
        
        # Save scaler
        scaler_path = f"models/optimized_scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save label encoder
        encoder_path = f"models/optimized_label_encoder_{timestamp}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save feature names
        features_path = f"models/selected_features_{timestamp}.txt"
        with open(features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        # Update the main model files (for compatibility)
        main_model_path = "models/optimized_deepfake_model.pkl"
        with open(main_model_path, 'wb') as f:
            pickle.dump({'model': best_model}, f)
        
        main_scaler_path = "models/optimized_scaler.pkl"
        with open(main_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        main_encoder_path = "models/optimized_label_encoder.pkl"
        with open(main_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        main_features_path = "models/selected_features.txt"
        with open(main_features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        print(f"\nBest Model: {best_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Main model updated: {main_model_path}")
        
        # Test with a sample to verify
        print(f"\nTesting model prediction...")
        sample_real = X_test[y_test == 1][0].reshape(1, -1)  # Real sample
        sample_fake = X_test[y_test == 0][0].reshape(1, -1)  # Fake sample
        
        pred_real = best_model.predict(sample_real)[0]
        pred_fake = best_model.predict(sample_fake)[0]
        
        pred_real_label = label_encoder.inverse_transform([pred_real])[0]
        pred_fake_label = label_encoder.inverse_transform([pred_fake])[0]
        
        print(f"Real sample predicted as: {pred_real_label}")
        print(f"Fake sample predicted as: {pred_fake_label}")
        
        return True
    
    return False

def main():
    """Main function"""
    print("DeepTrust Model Retraining with Balanced Dataset")
    print("=" * 60)
    
    success = retrain_models()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Models retrained with balanced dataset!")
        print("The model should now correctly classify:")
        print("  - Real videos as REAL")
        print("  - Fake videos as FAKE")
        print("\nRun the detection pipeline again to test improved accuracy.")
    else:
        print("\nERROR: Model retraining failed")

if __name__ == "__main__":
    main()
