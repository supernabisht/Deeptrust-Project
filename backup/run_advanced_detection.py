import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from advanced_deepfake_detector import AdvancedDeepfakeDetector
from sklearn.model_selection import train_test_split

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Deepfake Detection System')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--dataset', type=str, default='balanced_deepfake_dataset.csv', 
                       help='Path to dataset CSV file')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model', type=str, default='advanced_model.pkl', 
                       help='Path to save/load model')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory for results')
    return parser.parse_args()

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def load_dataset(filepath):
    """Load and preprocess dataset"""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic preprocessing
    if 'label' in df.columns:
        df['label'] = df['label'].map({'real': 0, 'fake': 1, 0: 0, 1: 1})
    
    return df

def train_model(detector, X, y, output_dir):
    """Train and evaluate the model"""
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    detector.train_ensemble_model(X_train, y_train)
    
    # Evaluate on test set
    results = detector.evaluate(X_test, y_test)
    
    # Print metrics
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    # Plot confusion matrix
    y_pred, _ = detector.predict(X_test)
    detector.plot_confusion_matrix(y_test, y_pred, 
                                 title='Confusion Matrix (Test Set)')
    
    # Save confusion matrix
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
               dpi=300, bbox_inches='tight')
    
    # Plot feature importances
    if hasattr(detector, 'feature_importances_'):
        detector.plot_feature_importances(feature_names=X.columns.tolist())
        plt.savefig(os.path.join(output_dir, 'feature_importances.png'), 
                   dpi=300, bbox_inches='tight')
    
    return detector

def analyze_video(detector, video_path, output_dir):
    """Analyze a single video file"""
    print(f"\nAnalyzing video: {video_path}")
    
    # Extract features
    features_df = detector.extract_enhanced_features(video_path)
    
    if features_df.empty:
        print("Error: Could not extract features from video")
        return
    
    # Make prediction
    y_pred, y_proba = detector.predict(features_df)
    confidence = max(y_proba[0]) if y_proba.any() else 0.5
    
    # Get class label
    class_label = 'FAKE' if y_pred[0] == 1 else 'REAL'
    
    print(f"\nPrediction: {class_label} (Confidence: {confidence:.2f})")
    
    # Save results
    result = {
        'video_path': video_path,
        'prediction': class_label,
        'confidence': float(confidence),
        'features': features_df.iloc[0].to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON
    result_file = os.path.join(output_dir, 'prediction_results.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {result_file}")
    
    return result

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = ensure_dir(args.output)
    
    # Initialize detector
    detector = AdvancedDeepfakeDetector()
    
    # Train mode
    if args.train:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file not found: {args.dataset}")
            sys.exit(1)
            
        # Load dataset
        df = load_dataset(args.dataset)
        
        # Separate features and target
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Train model
        detector = train_model(detector, X, y, output_dir)
        
        # Save model
        detector.save_model(args.model)
        print(f"\nModel saved to {args.model}")
    
    # Load existing model
    elif os.path.exists(args.model):
        print(f"Loading model from {args.model}...")
        detector = AdvancedDeepfakeDetector(args.model)
    else:
        print("Error: Model file not found. Use --train to train a new model.")
        sys.exit(1)
    
    # Analyze video if provided
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
            
        analyze_video(detector, args.video, output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
