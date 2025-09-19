import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class DeepfakeEvaluator:
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        self.model_path = model_path or self._find_latest_model()
        
        if self.model_path:
            self.load_model(self.model_path)
    
    def _find_latest_model(self):
        """Find the most recent model in the models directory."""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            print(f"Models directory '{models_dir}' not found.")
            return None
            
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))
                      and not f.startswith('tmp')]
        
        if not model_files:
            print("No model files found in the models directory.")
            return None
            
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), 
                        reverse=True)
        return os.path.join(models_dir, model_files[0])
    
    def load_model(self, model_path):
        """Load the model from the specified path."""
        try:
            print(f"\nLoading model from: {model_path}")
            model_data = joblib.load(model_path)
            
            # Handle different model formats
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
                print("Error: Model not found in the loaded file.")
                return False
                
            # Print model information
            print(f"Model type: {type(self.model).__name__}")
            if hasattr(self.model, 'n_features_in_'):
                print(f"Expected number of features: {self.model.n_features_in_}")
            if self.feature_names is not None:
                print(f"Feature names available: {len(self.feature_names)}")
                print("First 5 features:", self.feature_names[:5])
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, x_path, y_path):
        """Load and prepare the dataset."""
        try:
            print(f"\nLoading data from {x_path} and {y_path}...")
            
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
                print(f"Adjusted dataset size to {min_len} samples")
            
            print(f"Loaded {len(X)} samples with {X.shape[1]} features")
            print(f"Class distribution:\n{y.value_counts()}")
            
            # If model expects a specific number of features, adjust the dataset
            if hasattr(self.model, 'n_features_in_') and X.shape[1] != self.model.n_features_in_:
                print(f"\nAdjusting features to match model's expected count ({self.model.n_features_in_})...")
                if X.shape[1] > self.model.n_features_in_:
                    # If dataset has more features, keep only the first n_features_in_ features
                    X = X.iloc[:, :self.model.n_features_in_]
                    print(f"Kept first {self.model.n_features_in_} features")
                else:
                    # If dataset has fewer features, pad with zeros
                    padding = pd.DataFrame(
                        np.zeros((len(X), self.model.n_features_in_ - X.shape[1])),
                        columns=[f'padding_{i}' for i in range(self.model.n_features_in_ - X.shape[1])]
                    )
                    X = pd.concat([X, padding], axis=1)
                    print(f"Padded with {self.model.n_features_in_ - X.shape[1]} zero columns")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test set."""
        if self.model is None:
            print("Error: No model loaded.")
            return None
        
        try:
            print("\n" + "="*50)
            print("MODEL EVALUATION")
            print("="*50)
            
            # Make predictions
            print("\nMaking predictions...")
            y_pred = self.model.predict(X_test)
            
            # Get probabilities if available
            y_prob = None
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            self.plot_confusion_matrix(cm, ['REAL', 'FAKE'])
            
            # ROC and PR curves if probabilities are available
            if y_prob is not None:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                self.plot_roc_curve(fpr, tpr, roc_auc)
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = average_precision_score(y_test, y_prob)
                self.plot_pr_curve(recall, precision, pr_auc)
            
            # Feature importance if available
            self.plot_feature_importance(X_test)
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, cmap=plt.cm.Blues):
        """Plot the confusion matrix."""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Save the figure
        filename = 'reports/confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Confusion matrix saved to {filename}")
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC curve."""
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
        
        # Save the figure
        filename = 'reports/roc_curve.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"ROC curve saved to {filename}")
    
    def plot_pr_curve(self, recall, precision, pr_auc):
        """Plot the Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', color='b', alpha=0.2, lw=2)
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (AP = {pr_auc:.2f})')
        
        # Save the figure
        filename = 'reports/pr_curve.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Precision-Recall curve saved to {filename}")
    
    def plot_feature_importance(self, X, top_n=20):
        """Plot feature importance if the model supports it."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                print("\nGenerating feature importance plot...")
                
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
                
                # Plot top N features
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', 
                           data=feature_importance.head(top_n))
                plt.title(f'Top {top_n} Most Important Features')
                plt.tight_layout()
                
                # Ensure reports directory exists
                os.makedirs('reports', exist_ok=True)
                
                # Save the figure
                filename = 'reports/feature_importance.png'
                plt.savefig(filename)
                plt.close()
                print(f"Feature importance plot saved to {filename}")
                
                # Save feature importance to CSV
                feature_importance.to_csv('reports/feature_importance.csv', index=False)
                
        except Exception as e:
            print(f"\nCould not generate feature importance plot: {str(e)}")

def main():
    # Initialize evaluator
    evaluator = DeepfakeEvaluator()
    
    if evaluator.model is None:
        print("No model available for evaluation.")
        return
    
    # Define paths
    X_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'
    
    # Check if files exist
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Test data not found at {X_test_path} and {y_test_path}")
        return
    
    # Load and evaluate the model
    X_test, y_test = evaluator.load_data(X_test_path, y_test_path)
    
    if X_test is not None and y_test is not None:
        # Evaluate the model
        results = evaluator.evaluate(X_test, y_test)
        
        if results is not None:
            print("\n" + "="*50)
            print("EVALUATION COMPLETE")
            print("="*50)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print("\nCheck the 'reports' directory for visualizations.")

if __name__ == "__main__":
    main()
