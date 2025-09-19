#!/usr/bin/env python3
"""
Simple Deepfake Predictor

This script loads a trained model and makes predictions using MFCC features.
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

class DeepfakePredictor:
    """Class for making predictions using a trained deepfake detection model"""
    
    def __init__(self, model_path=None):
        """Initialize the predictor with a trained model"""
        self.model = None
        self.feature_names = None
        self.classes = ['REAL', 'FAKE']  # Default class names
        self.model_path = model_path or self._find_latest_model()
        self.load_model()
    
    def _find_latest_model(self):
        """Find the latest trained model"""
        models_dir = Path('enhanced_models/trained_models')
        if not models_dir.exists():
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        # Look for model files
        model_files = list(models_dir.glob('*.joblib')) + list(models_dir.glob('*.pkl'))
        if not model_files:
            raise FileNotFoundError("No model files found in the models directory.")
        
        # Return the most recently modified model
        return str(sorted(model_files, key=os.path.getmtime, reverse=True)[0])
    
    def load_model(self):
        """Load the trained model and its metadata"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            model_data = joblib.load(self.model_path)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.classes = model_data.get('classes', self.classes)
            else:
                self.model = model_data
            
            if self.model is None:
                raise ValueError("No model found in the loaded data")
                
            logger.info(f"Model loaded successfully. Model type: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, features):
        """Make predictions on extracted features"""
        try:
            # Convert features to DataFrame if needed
            if not isinstance(features, pd.DataFrame):
                features_df = pd.DataFrame([features])
            else:
                features_df = features.copy()
            
            # Ensure we have the right features in the right order
            if self.feature_names is not None:
                # Add missing features with default values
                for feat in self.feature_names:
                    if feat not in features_df.columns:
                        logger.warning(f"Missing feature: {feat} (using default value: 0)")
                        features_df[feat] = 0
                
                # Reorder columns to match training data
                features_df = features_df[self.feature_names]
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_df)
                predictions = self.model.predict(features_df)
                
                # Convert numeric predictions to class labels
                if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                    # Multi-class classification
                    results = []
                    for i, pred in enumerate(predictions):
                        result = {
                            'prediction': self.classes[pred] if pred < len(self.classes) else str(pred),
                            'confidence': np.max(probabilities[i])
                        }
                        # Add probabilities for each class
                        for j, cls in enumerate(self.classes):
                            result[f'prob_{cls.lower()}'] = probabilities[i][j] if j < len(probabilities[i]) else 0.0
                        results.append(result)
                    
                    return pd.DataFrame(results)
                else:
                    # Binary classification
                    return pd.DataFrame({
                        'prediction': [self.classes[p] if p < len(self.classes) else str(p) 
                                     for p in predictions],
                        'confidence': probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities,
                        'prob_real': probabilities[:, 0] if len(probabilities.shape) > 1 else 1 - probabilities,
                        'prob_fake': probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
                    })
            else:
                # Model doesn't support probabilities
                predictions = self.model.predict(features_df)
                return pd.DataFrame({
                    'prediction': [self.classes[p] if p < len(self.classes) else str(p) 
                                 for p in predictions],
                    'confidence': 1.0  # Default confidence
                })
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

def main():
    """Main function for command-line usage"""
    import argparse
    from mfcc_extractor import extract_from_video, extract_mfcc_features
    
    parser = argparse.ArgumentParser(description='Deepfake Video Predictor')
    parser.add_argument('input_file', help='Input video or audio file')
    parser.add_argument('--model', help='Path to trained model (default: auto-detect)')
    parser.add_argument('--output', help='Output CSV file for results (optional)')
    
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Initialize predictor
        predictor = DeepfakePredictor(model_path=args.model)
        
        # Extract features based on file type
        logger.info(f"Processing file: {args.input_file}")
        if args.input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            features = extract_from_video(args.input_file)
        else:
            features = extract_mfcc_features(args.input_file)
        
        if not features:
            raise ValueError("Failed to extract features from the input file")
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = predictor.predict(features)
        
        # Print results
        print("\n=== Deepfake Detection Results ===")
        print(f"File: {args.input_file}")
        print("\nPrediction:")
        print(predictions.to_string(index=False))
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
