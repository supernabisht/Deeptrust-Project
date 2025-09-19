#!/usr/bin/env python3
"""
Deepfake Detection Pipeline

This script provides an end-to-end pipeline for deepfake detection on video files.
It handles feature extraction, model loading, and prediction in one go.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

class DeepfakeDetectionPipeline:
    """End-to-end pipeline for deepfake detection"""
    
    def __init__(self, model_path=None):
        """Initialize the pipeline"""
        self.model_path = model_path or self._find_latest_model()
        self.model = None
        self.feature_extractor = None
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_components()
    
    def _find_latest_model(self):
        """Find the latest trained model"""
        models_dir = Path('enhanced_models/trained_models')
        if not models_dir.exists():
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        # Look for the best model (stacking model)
        model_path = models_dir / 'stacking_model.joblib'
        if model_path.exists():
            return str(model_path)
        
        # If no stacking model, find any model
        model_files = list(models_dir.glob('*.joblib'))
        if not model_files:
            raise FileNotFoundError("No model files found in the models directory.")
        
        # Return the most recently modified model
        return str(sorted(model_files, key=os.path.getmtime, reverse=True)[0])
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # Import here to avoid circular imports
        from deepfake_predictor import DeepfakePredictor
        from extract_features import FeatureExtractor
        
        # Initialize predictor and feature extractor
        self.predictor = DeepfakePredictor(model_path=self.model_path)
        self.feature_extractor = FeatureExtractor()
        
        logger.info(f"Using model: {self.model_path}")
    
    def process_video(self, video_path):
        """Process a single video file"""
        try:
            logger.info(f"\nProcessing video: {video_path}")
            
            # Extract features
            features = self.feature_extractor.process_video(video_path)
            if not features:
                raise ValueError("Failed to extract features from video")
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Make prediction
            predictions = self.predictor.predict(features_df)
            
            # Format results
            result = {
                'video_path': str(video_path),
                'prediction': predictions['prediction'].iloc[0],
                'confidence': float(predictions['confidence'].iloc[0]),
                'probability_FAKE': float(predictions['probability_FAKE'].iloc[0]),
                'probability_REAL': float(predictions['probability_REAL'].iloc[0]),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def process_videos(self, video_paths, output_file=None):
        """Process multiple video files"""
        results = []
        
        for video_path in video_paths:
            result = self.process_video(video_path)
            if result:
                results.append(result)
        
        if not results:
            logger.warning("No videos were successfully processed")
            return None
        
        # Save results
        results_df = pd.DataFrame(results)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return results_df

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Pipeline')
    parser.add_argument('videos', nargs='+', help='Video files to analyze')
    parser.add_argument('--output', '-o', type=str, default='results/predictions.csv',
                      help='Output file path for results')
    parser.add_argument('--model', '-m', type=str, default=None,
                      help='Path to trained model (default: use latest)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DeepfakeDetectionPipeline(model_path=args.model)
        
        # Process videos
        results = pipeline.process_videos(args.videos, args.output)
        
        if results is not None:
            # Print summary
            print("\n=== Deepfake Detection Results ===")
            print(results[['video_path', 'prediction', 'confidence']].to_string())
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            detailed_output = Path('results') / f'detailed_results_{timestamp}.json'
            with open(detailed_output, 'w') as f:
                json.dump(results.to_dict('records'), f, indent=2)
            
            print(f"\nDetailed results saved to: {detailed_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
