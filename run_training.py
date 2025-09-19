#!/usr/bin/env python3
"""
Deepfake Detection Model Training and Evaluation Pipeline

This script provides an end-to-end pipeline for training and evaluating
ensemble models for deepfake detection.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from optimized_model_trainer_clean import EnhancedModelTrainer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate deepfake detection models')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV file containing the dataset')
    parser.add_argument('--label_col', type=str, default='label',
                      help='Name of the target column in the dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Fraction of data to use for testing')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--cv_folds', type=int, default=5,
                      help='Number of cross-validation folds')
    
    return parser.parse_args()

def main():
    """Main training and evaluation pipeline"""
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        trainer = EnhancedModelTrainer(
            model_dir=args.model_dir,
            random_state=args.random_state
        )
        
        # Load and preprocess data
        logger.info(f"Loading data from {args.data_path}...")
        X_train, X_test, y_train, y_test = trainer.load_data(
            data_path=args.data_path,
            label_col=args.label_col,
            test_size=args.test_size
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train_processed, X_test_processed = trainer.preprocess_data(X_train, X_test)
        
        # Split into training and validation sets
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_processed, y_train,
            test_size=0.2,
            random_state=args.random_state,
            stratify=y_train if len(np.unique(y_train)) > 1 else None
        )
        
        # Train models
        logger.info("Starting model training...")
        trained_models = trainer.train_models(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            cv_folds=args.cv_folds
        )
        
        # Evaluate on test set
        logger.info("Evaluating models on test set...")
        test_results = trainer.evaluate_models(trained_models, X_test_processed, y_test)
        
        # Save trained models
        logger.info("Saving trained models...")
        saved_paths = trainer.save_models(trained_models)
        
        logger.info("\nTraining and evaluation completed successfully!")
        logger.info(f"Models saved to: {args.model_dir}")
        
        # Print best model results
        if trainer.best_model is not None:
            best_model_name = max(trainer.cv_results, key=lambda k: trainer.cv_results[k]['mean_cv_score'])
            logger.info(f"\nBest model: {best_model_name}")
            logger.info(f"Mean CV ROC-AUC: {trainer.cv_results[best_model_name]['mean_cv_score']:.4f}")
            logger.info(f"Test ROC-AUC: {test_results[best_model_name]['roc_auc']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
