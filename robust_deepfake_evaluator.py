"""
Robust Deepfake Detection Model Evaluator

This script provides a comprehensive evaluation of deepfake detection models,
handling both regression and classification models with robust error handling
and detailed reporting.
"""

import os
import sys
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, asdict

# Machine Learning Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('seaborn')
sns.set_palette('colorblind')

@dataclass
class ModelInfo:
    """Container for model information and metadata."""
    model: Any
    model_type: str  # 'classifier' or 'regressor'
    feature_names: Optional[List[str]] = None
    model_path: Optional[str] = None
    load_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding the model object."""
        result = asdict(self)
        result.pop('model', None)  # Don't include the actual model in the dict
        return result

class DeepfakeModelEvaluator:
    """Comprehensive evaluator for deepfake detection models."""
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        """Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.model_info = None
        self.output_dir = Path(output_dir)
        self.class_names = ['REAL', 'FAKE']
        self.class_colors = {'REAL': 'green', 'FAKE': 'red'}
        
        # Create output directories
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        self.data_dir = self.output_dir / 'data'
        
        for directory in [self.plots_dir, self.reports_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk with enhanced error handling and feature extraction.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Load the model
            model_data = joblib.load(model_path)
            
            # Handle different model storage formats
            if isinstance(model_data, dict):
                # If it's a dictionary, extract the model and other info
                model = model_data.get('model')
                feature_names = model_data.get('feature_names')
                model_type = model_data.get('model_type', 'classifier')
                
                # If feature names aren't in the dict, try to get them from the model
                if feature_names is None and hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                
                # If still no feature names, try to get from model parameters
                if feature_names is None and hasattr(model, 'get_params'):
                    params = model.get_params()
                    if 'feature_names' in params:
                        feature_names = params['feature_names']
                
                # If we have a pipeline, try to get feature names from the last step
                if feature_names is None and hasattr(model, 'steps') and isinstance(model.steps, list):
                    last_step = model.steps[-1][1]
                    if hasattr(last_step, 'feature_names_in_'):
                        feature_names = last_step.feature_names_in_
            else:
                # Assume it's the model object directly
                model = model_data
                feature_names = None
                # Try to determine model type
                model_type = self._infer_model_type(model)
                
                # Try to get feature names from the model
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif hasattr(model, 'get_params'):
                    params = model.get_params()
                    if 'feature_names' in params:
                        feature_names = params['feature_names']
            
            if model is None:
                logger.error("No model found in the loaded data")
                return False
            
            # Store model info
            self.model_info = ModelInfo(
                model=model,
                model_type=model_type,
                feature_names=feature_names,
                model_path=model_path,
                load_time=time.time() - start_time
            )
            
            logger.info(f"Successfully loaded {model_type} model")
            logger.info(f"Model class: {model.__class__.__name__}")
            
            if feature_names is not None:
                logger.info(f"Model expects {len(feature_names)} features")
                logger.info(f"First 5 feature names: {feature_names[:5]}...")
            else:
                logger.warning("No feature names available in the model")
                
                # Try to infer number of features from the model
                if hasattr(model, 'n_features_in_'):
                    logger.info(f"Model expects {model.n_features_in_} features")
                elif hasattr(model, 'coef_'):
                    if isinstance(model.coef_, np.ndarray):
                        logger.info(f"Model coefficients suggest {model.coef_.shape[1]} features")
                elif hasattr(model, 'feature_importances_'):
                    logger.info(f"Model feature importances suggest {len(model.feature_importances_)} features")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def _infer_model_type(self, model) -> str:
        """Infer whether the model is a classifier or regressor."""
        model_class = model.__class__.__name__.lower()
        
        # Common regressor indicators
        regressor_indicators = ['regressor', 'regression']
        if any(indicator in model_class for indicator in regressor_indicators):
            return 'regressor'
            
        # Common classifier indicators
        classifier_indicators = ['classifier', 'classify', 'predict_proba', 'predict_classes']
        if any(hasattr(model, attr) for attr in classifier_indicators):
            return 'classifier'
            
        # Default to classifier if uncertain
        return 'classifier'
    
    def load_data(self, x_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the dataset with enhanced error handling and logging."""
        try:
            logger.info(f"Loading data from {x_path} and {y_path}")
            
            # Load features and labels
            logger.info("Loading feature data...")
            X = pd.read_csv(x_path)
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            
            logger.info("Loading labels...")
            y = pd.read_csv(y_path, header=None, names=['label']).squeeze()
            logger.info(f"Loaded {len(y)} labels")
            
            # Convert labels to binary (0/1)
            y = self._normalize_labels(y)
            logger.info("Normalized labels")
            
            # Log initial data info
            logger.info(f"Initial data shape: {X.shape}")
            logger.info(f"Sample features: {X.iloc[0].to_dict()}")
            logger.info(f"Label distribution:\n{y.value_counts()}")
            
            # Align features with model's expected features if available
            if self.model_info and hasattr(self.model_info, 'feature_names') and self.model_info.feature_names is not None:
                logger.info("Aligning features with model's expected features...")
                X_before = X.shape
                X = self._align_features(X, self.model_info.feature_names)
                logger.info(f"Feature alignment complete. Shape before: {X_before}, after: {X.shape}")
            else:
                logger.warning("No feature names available in the model. Using features as-is.")
                # If model expects a specific number of features, check and pad if needed
                if hasattr(self.model_info.model, 'n_features_in_'):
                    expected_n = self.model_info.model.n_features_in_
                    if X.shape[1] < expected_n:
                        logger.warning(f"Adding {expected_n - X.shape[1]} zero features to match model's expected input")
                        for i in range(X.shape[1], expected_n):
                            X[f'feature_{i}'] = 0.0
                    elif X.shape[1] > expected_n:
                        logger.warning(f"Selecting first {expected_n} features to match model's expected input")
                        X = X.iloc[:, :expected_n]
            
            # Handle missing values
            X, y = self._handle_missing_values(X, y)
            
            # Ensure X and y have the same number of samples
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
                logger.warning(f"Adjusted dataset size to {min_len} samples")
            
            # Save processed data for reference
            X.to_csv(self.data_dir / 'X_processed.csv', index=False)
            y.to_csv(self.data_dir / 'y_processed.csv', index=False, header=['label'])
            
            logger.info(f"Final dataset shape: {X.shape}")
            logger.info(f"Final class distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def _normalize_labels(self, y):
        """Convert various label formats to binary (0/1)."""
        if y.dtype == 'object':
            # Handle string labels
            y = y.str.upper().map({'REAL': 0, 'FAKE': 1, '0': 0, '1': 1, 0: 0, 1: 1})
        
        # Ensure numeric type
        y = pd.to_numeric(y, errors='coerce')
        
        # Check if we have exactly two classes
        unique_values = y.unique()
        if len(unique_values) != 2:
            logger.warning(f"Expected 2 unique classes but found {len(unique_values)}: {unique_values}")
        
        return y
    
    def _align_features(self, X, expected_features):
        """Align dataset features with model's expected features with enhanced handling."""
        if expected_features is None:
            logger.warning("No expected features provided, cannot align features")
            return X
            
        # Convert to sets for easier comparison
        actual_features = set(X.columns)
        expected_set = set(expected_features)
        
        missing_features = expected_set - actual_features
        extra_features = actual_features - expected_set
        
        if missing_features or extra_features:
            logger.info(f"Aligning features - Missing: {len(missing_features)}, Extra: {len(extra_features)}")
            
            # Add missing features with default values (0)
            if missing_features:
                logger.info(f"Adding {len(missing_features)} missing features with default values")
                for feature in missing_features:
                    # Try to determine a good default value based on the feature name
                    default_value = 0.0
                    if any(x in str(feature).lower() for x in ['mean', 'avg', 'average']):
                        # For mean-like features, use 0
                        default_value = 0.0
                    elif any(x in str(feature).lower() for x in ['std', 'deviation', 'var']):
                        # For standard deviation features, use 1 (assuming standardized)
                        default_value = 1.0
                    
                    X[feature] = default_value
                    logger.debug(f"Added missing feature '{feature}' with default value {default_value}")
            
            # Reorder features to match expected order and remove extra features
            try:
                # First, ensure all expected features exist
                for feat in expected_features:
                    if feat not in X.columns:
                        X[feat] = 0.0  # Shouldn't happen due to above, but just in case
                
                # Now reorder and select only the expected features
                X = X[list(expected_features)]
                logger.info(f"Aligned features. New shape: {X.shape}")
                
            except Exception as e:
                logger.error(f"Error aligning features: {str(e)}")
                logger.error("This might indicate a critical mismatch between model and data")
                raise
        
        # Verify the alignment
        if list(X.columns) != list(expected_features):
            logger.warning("Feature alignment may not be correct. Column order might be different than expected.")
        
        return X
    
    def _handle_missing_values(self, X, y):
        """Handle missing values in features and labels."""
        # Check for missing values in features
        missing_values = X.isna().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found {missing_values.sum()} missing values in features. Imputing...")
            
            # Simple imputation (can be enhanced with more sophisticated methods)
            for col in X.columns[X.isna().any()]:
                if X[col].dtype in ['float64', 'int64']:
                    # For numeric columns, use median
                    fill_value = X[col].median()
                else:
                    # For categorical columns, use mode
                    fill_value = X[col].mode()[0]
                
                X[col].fillna(fill_value, inplace=True)
                logger.debug(f"Imputed {missing_values[col]} missing values in {col} with {fill_value}")
        
        # Check for missing values in labels
        if y.isna().sum() > 0:
            logger.warning(f"Found {y.isna().sum()} missing values in labels. Dropping...")
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
        
        return X, y
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate the model on the test set.
        
        Args:
            X_test: Test features
            y_test: Test labels (0/1)
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model_info is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            logger.info("Starting model evaluation...")
            start_time = time.time()
            
            # Make predictions
            logger.info("Making predictions...")
            if self.model_info.model_type == 'regressor':
                # For regression models, get probabilities and apply threshold
                y_pred_proba = self.model_info.model.predict(X_test)
                y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Ensure in [0, 1] range
                y_pred = (y_pred_proba >= threshold).astype(int)
            else:
                # For classifiers, get class probabilities
                if hasattr(self.model_info.model, 'predict_proba'):
                    y_pred_proba = self.model_info.model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba >= threshold).astype(int)
                else:
                    # Fallback to decision function or predict
                    y_pred = self.model_info.model.predict(X_test)
                    y_pred_proba = y_pred  # Use hard predictions as probabilities
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, threshold)
            
            # Generate visualizations
            self._generate_visualizations(y_test, y_pred, y_pred_proba, threshold)
            
            # Save metrics and predictions
            self._save_results(metrics, y_test, y_pred, y_pred_proba)
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, threshold):
        """Calculate evaluation metrics."""
        metrics = {
            'threshold': threshold,
            'model_type': self.model_info.model_type,
            'model_class': self.model_info.model.__class__.__name__
        }
        
        try:
            # Basic classification metrics
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'cohen_kappa': cohen_kappa_score(y_true, y_pred),
                'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
            })
            
            # Probability-based metrics
            if len(np.unique(y_pred_proba)) > 2:  # Only if we have continuous probabilities
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_pred_proba),
                    'average_precision': average_precision_score(y_true, y_pred_proba),
                    'log_loss': log_loss(y_true, y_pred_proba, labels=[0, 1])
                })
            
            # Classification report
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
                'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                'true_negative_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                'precision_recall_curve': self._get_precision_recall_curve(y_true, y_pred_proba)
            })
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
        
        return metrics
    
    def _get_precision_recall_curve(self, y_true, y_pred_proba):
        """Calculate precision-recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': np.append(thresholds, 1.0).tolist()  # Add endpoint for plotting
        }
    
    def _generate_visualizations(self, y_true, y_pred, y_pred_proba, threshold):
        """Generate evaluation visualizations."""
        try:
            # Plot confusion matrix
            self._plot_confusion_matrix(y_true, y_pred)
            
            # Plot ROC curve if we have probability estimates
            if len(np.unique(y_pred_proba)) > 2:
                self._plot_roc_curve(y_true, y_pred_proba)
                self._plot_pr_curve(y_true, y_pred_proba)
                self._plot_probability_distribution(y_true, y_pred_proba, threshold)
                self._plot_threshold_analysis(y_true, y_pred_proba)
            
            logger.info("Generated all visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
    
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
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.plots_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distribution(self, y_true, y_pred_proba, threshold):
        """Plot and save probability distribution."""
        plt.figure(figsize=(10, 6))
        
        # Plot histograms for each class
        for label in [0, 1]:
            sns.histplot(
                y_pred_proba[y_true == label],
                bins=20,
                label=f'Class {self.class_names[label]}',
                alpha=0.6,
                kde=True,
                color=self.class_colors[self.class_names[label]]
            )
        
        # Add threshold line
        plt.axvline(x=threshold, color='red', linestyle='--', 
                   label=f'Decision Threshold ({threshold:.2f})')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Predicted Probability Distribution by True Class')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, y_true, y_pred_proba, n_thresholds=100):
        """Plot metrics across different decision thresholds."""
        thresholds = np.linspace(0, 1, n_thresholds)
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        
        plt.figure(figsize=(10, 6))
        for metric_name, values in metrics.items():
            plt.plot(thresholds, values, label=metric_name.capitalize())
        
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Model Metrics vs. Decision Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, metrics, y_true, y_pred, y_pred_proba):
        """Save evaluation results to files."""
        try:
            # Save full metrics as JSON
            with open(self.reports_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save classification report as text
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                with open(self.reports_dir / 'classification_report.txt', 'w') as f:
                    f.write(classification_report(
                        y_true, y_pred,
                        target_names=self.class_names,
                        output_dict=False,
                        zero_division=0
                    ))
            
            # Save predictions
            results_df = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'predicted_probability': y_pred_proba
            })
            results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
            results_df.to_csv(self.reports_dir / 'predictions.csv', index=False)
            
            # Save model info
            if self.model_info:
                with open(self.reports_dir / 'model_info.json', 'w') as f:
                    json.dump(self.model_info.to_dict(), f, indent=2)
            
            logger.info(f"Saved evaluation results to {self.reports_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)

def main():
    """Main function to run the evaluation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robust Deepfake Detection Model Evaluation')
    parser.add_argument('--model', type=str, default='models/latest_model.pkl',
                      help='Path to the trained model file')
    parser.add_argument('--x_test', type=str, default='data/processed/X_test_cleaned.csv',
                      help='Path to test features CSV file')
    parser.add_argument('--y_test', type=str, default='data/processed/y_test_cleaned.csv',
                      help='Path to test labels CSV file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for converting probabilities to binary predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = DeepfakeModelEvaluator(output_dir=args.output_dir)
        
        # Load model
        if not evaluator.load_model(args.model):
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
        print("\n" + "="*60)
        print("DEEPFAKE DETECTION MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Model Type: {metrics.get('model_type', 'unknown')}")
        print(f"Test samples: {len(X_test)}")
        print(f"Threshold: {args.threshold:.2f}")
        print("\nPerformance Metrics:")
        print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  - Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  - Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"  - F1 Score: {metrics.get('f1', 'N/A'):.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
        if 'average_precision' in metrics:
            print(f"  - Average Precision: {metrics['average_precision']:.4f}")
        
        print(f"\nDetailed results and visualizations saved to: {args.output_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
