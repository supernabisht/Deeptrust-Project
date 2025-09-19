"""
Evaluate a regression model for binary classification.

This script handles the evaluation of a regression model that outputs
continuous values between 0 and 1, converting them to binary predictions
for classification metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import json
from pathlib import Path
import time

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn')

class RegressionModelEvaluator:
    """Evaluator for regression models used in binary classification tasks."""
    
    def __init__(self, model_path: str = None, output_dir: str = 'evaluation_results'):
        """Initialize the evaluator."""
        self.model = None
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.class_names = ['REAL', 'FAKE']
        
        # Create output directories
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        
        for directory in [self.plots_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load model if path is provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk."""
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model type: {type(self.model).__name__}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, x_path: str, y_path: str):
        """Load and prepare the dataset."""
        try:
            logger.info(f"Loading data from {x_path} and {y_path}")
            
            # Load features and labels
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path, header=None, names=['label']).squeeze()
            
            # Convert labels to binary (0/1)
            if y.dtype == 'object':
                y = y.map({'REAL': 0, 'FAKE': 1, 'real': 0, 'fake': 1, 0: 0, 1: 1})
            
            # Ensure X and y have the same number of samples
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
                logger.warning(f"Adjusted dataset size to {min_len} samples")
            
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            logger.info(f"Class distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate the regression model on the test set."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            logger.info("Starting model evaluation...")
            start_time = time.time()
            
            # Make predictions
            logger.info("Making predictions...")
            y_pred_proba = self.model.predict(X_test)
            
            # Ensure predictions are between 0 and 1
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
            
            # Convert to binary predictions using threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Generate visualizations
            self._generate_visualizations(y_test, y_pred, y_pred_proba)
            
            # Save metrics to file
            self._save_metrics(metrics)
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _generate_visualizations(self, y_true, y_pred, y_pred_proba):
        """Generate evaluation visualizations."""
        try:
            # Plot confusion matrix
            self._plot_confusion_matrix(y_true, y_pred)
            
            # Plot ROC curve
            self._plot_roc_curve(y_true, y_pred_proba)
            
            # Plot Precision-Recall curve
            self._plot_pr_curve(y_true, y_pred_proba)
            
            # Plot probability distribution
            self._plot_probability_distribution(y_true, y_pred_proba)
            
            logger.info("Generated all visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png')
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png')
        plt.close()
    
    def _plot_pr_curve(self, y_true, y_pred_proba):
        """Plot and save Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', color='b', alpha=0.2, lw=2)
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (AP = {pr_auc:.2f})')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pr_curve.png')
        plt.close()
    
    def _plot_probability_distribution(self, y_true, y_pred_proba):
        """Plot and save probability distribution."""
        plt.figure(figsize=(10, 6))
        
        # Plot histograms for each class
        for label in [0, 1]:
            sns.histplot(
                y_pred_proba[y_true == label],
                bins=20,
                label=f'Class {self.class_names[label]}',
                alpha=0.6,
                kde=True
            )
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Predicted Probability Distribution by True Class')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'probability_distribution.png')
        plt.close()
    
    def _save_metrics(self, metrics):
        """Save evaluation metrics to files."""
        try:
            # Save full metrics as JSON
            with open(self.reports_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save classification report as text
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                with open(self.reports_dir / 'classification_report.txt', 'w') as f:
                    f.write(classification_report(
                        [0, 1], [0, 1],  # Dummy data for formatting
                        target_names=self.class_names,
                        output_dict=False
                    ))
                    f.write("\n\nDetailed Report:\n")
                    f.write(json.dumps(report, indent=2))
            
            logger.info(f"Saved evaluation metrics to {self.reports_dir}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

def main():
    """Main function to run the evaluation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Regression Model for Binary Classification')
    parser.add_argument('--model', type=str, default='models/latest_model.pkl',
                       help='Path to the trained model file')
    parser.add_argument('--x_test', type=str, default='data/processed/X_test_cleaned.csv',
                       help='Path to test features CSV file')
    parser.add_argument('--y_test', type=str, default='data/processed/y_test_cleaned.csv',
                       help='Path to test labels CSV file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for converting regression outputs to binary predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = RegressionModelEvaluator(model_path=args.model, output_dir=args.output_dir)
        
        if evaluator.model is None:
            logger.error("Failed to load model. Exiting.")
            return 1
        
        # Load test data
        X_test, y_test = evaluator.load_data(args.x_test, args.y_test)
        
        if X_test is None or y_test is None:
            logger.error("Failed to load test data. Exiting.")
            return 1
        
        # Evaluate model
        metrics = evaluator.evaluate(X_test, y_test, threshold=args.threshold)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Test samples: {len(X_test)}")
        print(f"Threshold: {args.threshold}")
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"\nDetailed results and visualizations saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
