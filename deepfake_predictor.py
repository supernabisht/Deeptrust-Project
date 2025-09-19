#!/usr/bin/env python3
"""
Deepfake Video Predictor

This script loads a trained model and makes predictions on new video files.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
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
    
    def __init__(self, model_path='enhanced_models/trained_models/stacking_model.joblib'):
        """Initialize the predictor with a trained model"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.classes_ = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and its metadata"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names')
            self.classes_ = model_data.get('classes', ['REAL', 'FAKE'])
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_features(self, features_df):
        """Preprocess input features to match training data format"""
        # Ensure features match the training data
        missing_cols = set(self.feature_names) - set(features_df.columns)
        extra_cols = set(features_df.columns) - set(self.feature_names)
        
        # Add missing columns with default values
        for col in missing_cols:
            features_df[col] = 0
        
        # Remove extra columns
        features_df = features_df[self.feature_names]
        
        return features_df
    
    def predict(self, features_df):
        """Make predictions on input features"""
        try:
            # Preprocess features
            processed_features = self.preprocess_features(features_df)
            
            # Make predictions
            probabilities = self.model.predict_proba(processed_features)
            predictions = self.model.predict(processed_features)
            
            # Convert numeric predictions back to labels
            if self.classes_ is not None:
                predicted_labels = [self.classes_[p] for p in predictions]
            else:
                predicted_labels = ['FAKE' if p == 1 else 'REAL' for p in predictions]
            
            # Create results dataframe
            results = pd.DataFrame({
                'prediction': predicted_labels,
                'confidence': np.max(probabilities, axis=1),
                'probability_REAL': probabilities[:, 0] if self.classes_ is not None and len(self.classes_) > 0 else 1 - probabilities[:, 1],
                'probability_FAKE': probabilities[:, 1] if self.classes_ is not None and len(self.classes_) > 1 else probabilities[:, 1]
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

def main():
    """Main function to run predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Video Predictor')
    parser.add_argument('--features', type=str, required=True,
                      help='Path to CSV file containing features')
    parser.add_argument('--output', type=str, default='predictions.csv',
                      help='Output file path for predictions')
    parser.add_argument('--model', type=str, default='enhanced_models/trained_models/stacking_model.joblib',
                      help='Path to trained model')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = DeepfakePredictor(model_path=args.model)
        
        # Load features
        logger.info(f"Loading features from {args.features}")
        features_df = pd.read_csv(args.features)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predictor.predict(features_df)
        
        # Save results
        predictions.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(predictions['prediction'].value_counts().to_string())
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
