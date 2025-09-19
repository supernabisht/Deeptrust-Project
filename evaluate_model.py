import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model_path=None):
        """Initialize the model evaluator."""
        self.model = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model."""
        try:
            print(f"Loading model from {model_path}...")
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
            else:
                self.model = model_data
            
            if self.model is None:
                print("Error: Could not load model from the file.")
                return False
                
            print(f"Successfully loaded {self.model.__class__.__name__} model")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, x_path: str, y_path: str) -> tuple:
        """Load features and labels from CSV files."""
        try:
            print(f"Loading data from {x_path} and {y_path}...")
            
            # Try to load aligned features if they exist
            aligned_x_path = x_path.replace('.csv', '_aligned.csv')
            if os.path.exists(aligned_x_path):
                print(f"Using aligned features from {aligned_x_path}")
                X = pd.read_csv(aligned_x_path)
            else:
                X = pd.read_csv(x_path)
            
            y = pd.read_csv(y_path, header=None, names=['label']).squeeze()
            
            # Convert labels to binary (0/1)
            if y.dtype == 'object':
                y = y.map({'REAL': 0, 'FAKE': 1})
            
            # Ensure X and y have the same number of samples
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if self.model is None:
            print("Error: No model loaded.")
            return None
            
        try:
            print("\nEvaluating model...")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
            else:
                y_prob = None
                roc_auc = None
                fpr, tpr = None, None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Print results
            print("\n" + "="*50)
            print("Model Evaluation Results")
            print("="*50)
            print(f"Accuracy: {accuracy:.4f}")
            if roc_auc is not None:
                print(f"ROC AUC: {roc_auc:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
            
            # Plot confusion matrix
            self.plot_confusion_matrix(cm, ['REAL', 'FAKE'])
            
            # Plot ROC curve if available
            if roc_auc is not None and fpr is not None and tpr is not None:
                self.plot_roc_curve(fpr, tpr, roc_auc)
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix.png')
        plt.close()
        print("\nConfusion matrix saved to 'reports/confusion_matrix.png'")
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the figure
        plt.savefig('reports/roc_curve.png')
        plt.close()
        print("ROC curve saved to 'reports/roc_curve.png'")

def main():
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Find the latest model in the models directory
    models_dir = 'models'
    model_path = None
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))]
        
        # Filter out any temporary files
        model_files = [f for f in model_files if not f.startswith('tmp')]
        
        if model_files:
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            model_path = os.path.join(models_dir, model_files[0])
            print(f"Found {len(model_files)} model(s). Using most recent: {model_path}")
        else:
            print("No model files found in the models directory.")
            return
    
    if model_path and os.path.exists(model_path):
        # Load the model
        if not evaluator.load_model(model_path):
            print("Failed to load model. Please check the model file.")
            return
        
        # Load test data
        X_test_path = 'data/processed/X_test.csv'
        y_test_path = 'data/processed/y_test.csv'
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print("Test data not found. Please run prepare_dataset.py first.")
            return
        
        X_test, y_test = evaluator.load_data(X_test_path, y_test_path)
        
        if X_test is not None and y_test is not None:
            # Evaluate the model
            results = evaluator.evaluate(X_test, y_test)
            
            # Save results to a file
            if results:
                with open('reports/evaluation_results.txt', 'w') as f:
                    f.write("Model Evaluation Results\n")
                    f.write("======================\n\n")
                    f.write(f"Model: {model_path}\n")
                    f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                    if results['roc_auc'] is not None:
                        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")
                    
                    f.write("\nClassification Report:\n")
                    f.write(classification_report(
                        y_test, 
                        evaluator.model.predict(X_test),
                        target_names=['REAL', 'FAKE']
                    ))
                    
                    f.write("\nConfusion Matrix:\n")
                    f.write(np.array2string(
                        results['confusion_matrix'], 
                        separator=', '
                    ))
                
                print("\nEvaluation complete!")
                print(f"Results saved to 'reports/evaluation_results.txt'")
    else:
        print("No model file found in the 'models' directory.")
        print("Please make sure you have a trained model before running evaluation.")

if __name__ == "__main__":
    main()
