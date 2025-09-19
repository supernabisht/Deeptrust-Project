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

def load_model(model_path):
    """Load the model from the specified path."""
    try:
        print(f"\nLoading model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_names = model_data.get('feature_names')
            model_metadata = {k: v for k, v in model_data.items() 
                           if k not in ['model', 'feature_names']}
        else:
            model = model_data
            feature_names = None
            model_metadata = {}
        
        if model is None:
            print("Error: Model not found in the loaded file.")
            return None, None, None
            
        # Print model information
        print(f"Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            print(f"Expected number of features: {model.n_features_in_}")
        if feature_names:
            print(f"Feature names available: {len(feature_names)}")
            print("First 5 features:", feature_names[:5])
        
        return model, feature_names, model_metadata
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def load_data(x_path, y_path, feature_names=None):
    """Load and prepare the dataset."""
    try:
        print(f"\nLoading data from {x_path} and {y_path}...")
        
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
            y = y.map({'REAL': 0, 'FAKE': 1, 'real': 0, 'fake': 1, 0: 0, 1: 1})
        
        # Ensure X and y have the same number of samples
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            print(f"Adjusted dataset size to {min_len} samples")
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Class distribution:\n{y.value_counts()}")
        
        # Align features if feature names are provided
        if feature_names is not None and len(feature_names) > 0:
            X = align_features(X, feature_names)
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def align_features(X, feature_names):
    """Align features to match model's expected format."""
    print(f"\nAligning features to match model's expected format...")
    print(f"Expected features: {len(feature_names)}")
    print(f"Available features: {X.shape[1]}")
    
    # Create a new DataFrame with the expected features
    aligned_X = pd.DataFrame(columns=feature_names)
    
    # Fill in matching columns
    missing_features = []
    for col in feature_names:
        if col in X.columns:
            aligned_X[col] = X[col].values
        else:
            missing_features.append(col)
            aligned_X[col] = 0.0  # Fill missing with zeros
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in dataset. Filled with zeros.")
        print("First 5 missing features:", missing_features[:5])
    
    # Add any extra features from the dataset (with warning)
    extra_features = set(X.columns) - set(feature_names)
    if extra_features:
        print(f"Warning: {len(extra_features)} extra features in dataset will be dropped.")
        print("First 5 extra features:", list(extra_features)[:5])
    
    return aligned_X

def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
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

def plot_roc_curve(fpr, tpr, roc_auc):
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

def plot_pr_curve(recall, precision, pr_auc):
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

def evaluate_model(model, X_test, y_test, feature_names=None):
    """Evaluate the model on the test set."""
    if model is None:
        print("Error: No model provided.")
        return None
    
    try:
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Align features if needed
        if hasattr(model, 'n_features_in_') and X_test.shape[1] != model.n_features_in_:
            print(f"\nFeature mismatch: Model expects {model.n_features_in_} features, got {X_test.shape[1]}")
            if feature_names is None and hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            X_test = align_features(X_test, feature_names if feature_names is not None else X_test.columns.tolist())
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ['REAL', 'FAKE'])
        
        # ROC and PR curves if probabilities are available
        if y_prob is not None:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            plot_pr_curve(recall, precision, pr_auc)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            print("\nGenerating feature importance plot...")
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', 
                       data=feature_importance.head(20))
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            
            # Save the figure
            filename = 'reports/feature_importance.png'
            plt.savefig(filename)
            plt.close()
            print(f"Feature importance plot saved to {filename}")
            
            # Save feature importance to CSV
            feature_importance.to_csv('reports/feature_importance.csv', index=False)
        
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

def main():
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Find the latest model
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))
                  and not f.startswith('tmp')]
    
    if not model_files:
        print("No model files found in the models directory.")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), 
                    reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    
    # Load the model
    model, feature_names, model_metadata = load_model(model_path)
    if model is None:
        print("Failed to load model.")
        return
    
    # Load test data
    X_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Test data not found at {X_test_path} and {y_test_path}")
        return
    
    X_test, y_test = load_data(X_test_path, y_test_path, feature_names)
    
    if X_test is not None and y_test is not None:
        # Evaluate the model
        results = evaluate_model(model, X_test, y_test, feature_names)
        
        if results is not None:
            print("\n" + "="*50)
            print("EVALUATION COMPLETE")
            print("="*50)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print("\nCheck the 'reports' directory for visualizations.")

if __name__ == "__main__":
    main()
