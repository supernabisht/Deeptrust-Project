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

# Set up paths
MODEL_PATH = 'models/latest_model.pkl'
X_TEST_PATH = 'data/processed/X_test.csv'
Y_TEST_PATH = 'data/processed/y_test.csv'
REPORTS_DIR = 'reports'

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_model(model_path):
    """Load the model from the specified path."""
    print(f"Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Type: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_data(x_path, y_path, n_features=84):
    """Load and prepare the dataset."""
    print(f"\nLoading data from {x_path} and {y_path}")
    try:
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
        
        # Ensure correct number of features
        if X.shape[1] > n_features:
            X = X.iloc[:, :n_features]
            print(f"Using first {n_features} features")
        elif X.shape[1] < n_features:
            padding = pd.DataFrame(
                np.zeros((len(X), n_features - X.shape[1])),
                columns=[f'padding_{i}' for i in range(n_features - X.shape[1])]
            )
            X = pd.concat([X, padding], axis=1)
            print(f"Padded with {n_features - X.shape[1]} zero columns")
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_true, y_prob, output_dir):
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
    
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")

def plot_pr_curve(y_true, y_prob, output_dir):
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
    
    output_path = os.path.join(output_dir, 'pr_curve.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Precision-Recall curve saved to {output_path}")

def main():
    # Load the model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Load the test data
    X_test, y_test = load_data(X_TEST_PATH, Y_TEST_PATH, n_features=84)
    if X_test is None or y_test is None:
        return
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        # Check if the model has predict_proba method
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, ['REAL', 'FAKE'], REPORTS_DIR)
        
        # Plot ROC and PR curves if probabilities are available
        if y_prob is not None:
            plot_roc_curve(y_test, y_prob, REPORTS_DIR)
            plot_pr_curve(y_test, y_prob, REPORTS_DIR)
        
        print("\nEvaluation complete!")
        print(f"Results saved to the '{REPORTS_DIR}' directory.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()
