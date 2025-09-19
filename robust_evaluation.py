"""
Robust Deepfake Detection Model Evaluation

This script provides a comprehensive evaluation of the deepfake detection model,
including detailed metrics, visualizations, and performance analysis.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Machine Learning Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss
)

# Model Analysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.inspection import permutation_importance

# Visualization
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.ticker as mtick

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
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# Constants
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores

class DeepfakeEvaluator:
    """Comprehensive evaluator for deepfake detection models."""
    
    def __init__(self, model_path: str = None, output_dir: str = 'evaluation_results'):
        """Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model file.
            output_dir: Directory to save evaluation results.
        """
        self.model = None
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.feature_names = None
        self.model_metadata = {}
        self.evaluation_metrics = {}
        self.class_names = ['REAL', 'FAKE']
        
        # Create output directories
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        self.models_dir = self.output_dir / 'models'
        
        for directory in [self.plots_dir, self.reports_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load model if path is provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            bool: True if model was loaded successfully, False otherwise.
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            model_data = joblib.load(model_path)
            
            # Handle different model storage formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.model_metadata = {k: v for k, v in model_data.items() 
                                     if k not in ['model', 'feature_names']}
            else:
                self.model = model_data
                self.feature_names = None
                self.model_metadata = {}
            
            if self.model is None:
                logger.error("No model found in the loaded file.")
                return False
            
            # Log model information
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Get feature information if available
            if hasattr(self.model, 'n_features_in_'):
                logger.info(f"Model expects {self.model.n_features_in_} features")
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Model has {len(self.feature_names)} feature names from training")
            elif self.feature_names is not None:
                logger.info(f"Model has {len(self.feature_names)} saved feature names")
            
            # Save model metadata
            self.model_metadata.update({
                'model_type': type(self.model).__name__,
                'n_features_expected': getattr(self.model, 'n_features_in_', None),
                'feature_names': self.feature_names,
                'load_timestamp': datetime.now().isoformat()
            })
            
            # Save model metadata to file
            with open(self.models_dir / 'model_metadata.json', 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, x_path: str, y_path: str, n_features: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the dataset.
        
        Args:
            x_path: Path to the features CSV file.
            y_path: Path to the labels CSV file.
            n_features: Expected number of features. If None, use model's expected features.
            
        Returns:
            Tuple containing features DataFrame and labels Series.
        """
        try:
            logger.info(f"Loading data from {x_path} and {y_path}")
            
            # Load features
            X = pd.read_csv(x_path)
            
            # Load labels
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
            
            # Get expected number of features
            if n_features is None and hasattr(self.model, 'n_features_in_'):
                n_features = self.model.n_features_in_
            
            # Align features if needed
            if n_features is not None:
                X = self._align_features(X, n_features)
            
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            logger.info(f"Class distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _align_features(self, X: pd.DataFrame, n_features: int) -> pd.DataFrame:
        """Align features to match the expected number of features.
        
        Args:
            X: Input features DataFrame.
            n_features: Expected number of features.
            
        Returns:
            Aligned features DataFrame.
        """
        current_features = X.shape[1]
        
        if current_features == n_features:
            return X
        
        logger.warning(f"Feature count mismatch: Expected {n_features}, got {current_features}")
        
        if current_features > n_features:
            # If too many features, keep only the first n_features
            logger.warning(f"Keeping first {n_features} features and dropping the rest")
            return X.iloc[:, :n_features]
        else:
            # If too few features, pad with zeros
            padding = np.zeros((len(X), n_features - current_features))
            padding_columns = [f'padding_{i}' for i in range(n_features - current_features)]
            padding_df = pd.DataFrame(padding, columns=padding_columns, index=X.index)
            return pd.concat([X, padding_df], axis=1)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Evaluate the model on the test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            cv_folds: Number of cross-validation folds.
            
        Returns:
            Dictionary containing evaluation metrics and results.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            logger.info("Starting model evaluation...")
            start_time = time.time()
            
            # Make predictions
            y_pred, y_prob = self._make_predictions(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            
            # Generate visualizations
            self._generate_visualizations(X_test, y_test, y_pred, y_prob)
            
            # Perform cross-validation if requested
            if cv_folds > 1:
                cv_scores = self._cross_validate(X_test, y_test, cv_folds)
                metrics['cross_validation'] = cv_scores
            
            # Calculate feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self._plot_feature_importance(X_test, y_test)
            
            # Save metrics to file
            self._save_metrics(metrics)
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _make_predictions(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the loaded model.
        
        Args:
            X: Input features.
            
        Returns:
            Tuple of (y_pred, y_prob) where y_prob may be None if not available.
        """
        try:
            # Get class predictions
            y_pred = self.model.predict(X)
            
            # Get probability predictions if available
            y_prob = None
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X)[:, 1]
            
            return y_pred, y_prob
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (optional).
            
        Returns:
            Dictionary of calculated metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
            metrics['log_loss'] = log_loss(y_true, y_prob)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _generate_visualizations(self, X: pd.DataFrame, y_true: np.ndarray, 
                               y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None):
        """Generate evaluation visualizations.
        
        Args:
            X: Input features.
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (optional).
        """
        try:
            # Plot confusion matrix
            self._plot_confusion_matrix(y_true, y_pred)
            
            # Plot ROC and PR curves if probabilities are available
            if y_prob is not None:
                self._plot_roc_curve(y_true, y_prob)
                self._plot_pr_curve(y_true, y_prob)
                self._plot_calibration_curve(y_true, y_prob)
            
            # Plot feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self._plot_feature_importance(X, y_true)
            
            logger.info("Generated all visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
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
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
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
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot and save Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
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
    
    def _plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot and save calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'calibration_curve.png')
        plt.close()
    
    def _plot_feature_importance(self, X: pd.DataFrame, y: np.ndarray, top_n: int = 20):
        """Plot and save feature importance."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Get feature importance
                importance = self.model.feature_importances_
                
                # Get feature names
                if hasattr(X, 'columns'):
                    feature_names = X.columns.tolist()
                elif self.feature_names is not None:
                    feature_names = self.feature_names
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                # Create a DataFrame for visualization
                feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(importance)],
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Save to CSV
                feature_importance.to_csv(self.reports_dir / 'feature_importance.csv', index=False)
                
                # Plot top N features
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', 
                           data=feature_importance.head(top_n))
                plt.title(f'Top {top_n} Most Important Features')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(self.plots_dir / 'feature_importance.png')
                plt.close()
                
                logger.info("Generated feature importance plot")
                
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {str(e)}")
    
    def _cross_validate(self, X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation.
        
        Args:
            X: Input features.
            y: Target labels.
            cv_folds: Number of cross-validation folds.
            
        Returns:
            Dictionary of cross-validation results.
        """
        try:
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            
            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted',
                'roc_auc': 'roc_auc_ovr'
            }
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            
            cv_results = {}
            for metric_name, metric in scoring.items():
                try:
                    scores = cross_val_score(
                        self.model, X, y, 
                        cv=cv, scoring=metric, n_jobs=N_JOBS
                    )
                    cv_results[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'values': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Could not compute {metric_name} in CV: {str(e)}")
            
            logger.info("Cross-validation completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            return {}
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save evaluation metrics to files.
        
        Args:
            metrics: Dictionary of evaluation metrics.
        """
        try:
            # Save full metrics as JSON
            with open(self.reports_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save classification report as text
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                with open(self.reports_dir / 'classification_report.txt', 'w') as f:
                    # Convert dictionary report to string format
                    f.write(classification_report(
                        [0, 1], [0, 1],  # Dummy data, we'll replace with actual values
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
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model')
    parser.add_argument('--model', type=str, default='models/latest_model.pkl',
                       help='Path to the trained model file')
    parser.add_argument('--x_test', type=str, default='data/processed/X_test.csv',
                       help='Path to test features CSV file')
    parser.add_argument('--y_test', type=str, default='data/processed/y_test.csv',
                       help='Path to test labels CSV file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = DeepfakeEvaluator(model_path=args.model, output_dir=args.output_dir)
        
        if evaluator.model is None:
            logger.error("Failed to load model. Exiting.")
            sys.exit(1)
        
        # Load test data
        X_test, y_test = evaluator.load_data(args.x_test, args.y_test)
        
        if X_test is None or y_test is None:
            logger.error("Failed to load test data. Exiting.")
            sys.exit(1)
        
        # Evaluate model
        metrics = evaluator.evaluate(X_test, y_test, cv_folds=args.cv_folds)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Test samples: {len(X_test)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nDetailed results and visualizations saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
