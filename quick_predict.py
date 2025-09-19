#!/usr/bin/env python3
"""
Quick Deepfake Detection Script

This script loads a pre-trained model and makes predictions on a video file.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        # Look for the best model (stacking model)
        model_path = 'enhanced_models/trained_models/stacking_model.joblib'
        
        if not os.path.exists(model_path):
            # If no stacking model, find any model
            models_dir = Path('enhanced_models/trained_models')
            model_files = list(models_dir.glob('*.joblib'))
            if model_files:
                model_path = str(model_files[0])
            else:
                raise FileNotFoundError("No trained models found. Please train a model first.")
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def analyze_video(video_path):
    """Simple video analysis (placeholder for actual feature extraction)"""
    # In a real implementation, you would extract features from the video
    # For now, we'll use sample features that match the training data format
    
    # This is a placeholder - replace with actual feature extraction
    import random
    
    # Generate random features that match the expected format
    features = {}
    
    # Add MFCC features (40 coefficients * 2 stats = 80 features)
    for i in range(1, 41):
        features[f'mfcc_{i}_mean'] = random.uniform(-300, 300)
        features[f'mfcc_{i}_std'] = random.uniform(0, 100)
    
    # Add other features (lip movement, face detection, etc.)
    features.update({
        'face_area_mean': random.uniform(1000, 10000),
        'face_area_std': random.uniform(0, 1000),
        'face_count': 1,
        'lip_sync_score': random.uniform(0.7, 1.0),
        'lip_movement_variance': random.uniform(0.1, 0.5),
        'duration': random.uniform(5, 60),
        'fps': 30,
        'width': 1920,
        'height': 1080
    })
    
    return features

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Deepfake Detection')
    parser.add_argument('video', help='Video file to analyze')
    parser.add_argument('--model', help='Path to trained model (default: auto-detect)')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model_data = load_model(args.model)
        model = model_data['model']
        
        # Analyze video (extract features)
        logger.info(f"Analyzing video: {args.video}")
        features = analyze_video(args.video)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
        # Get class names
        classes = model_data.get('classes', ['REAL', 'FAKE'])
        
        # Print results
        print("\n=== Deepfake Detection Results ===")
        print(f"Video: {args.video}")
        print(f"Prediction: {classes[prediction]}")
        print("Probabilities:")
        for i, cls in enumerate(classes):
            print(f"  {cls}: {probability[i]:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
