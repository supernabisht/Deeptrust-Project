# optimized_model_trainer_clean.py
import numpy as np
import pandas as pd
import os
import joblib
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Union, Any
import logging

# Data processing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, 
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, auc,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, log_loss
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced model trainer with advanced features and better performance tracking"""
    
    def __init__(self, model_dir: str = 'models', random_state: int = 42):
        """
        Initialize the enhanced model trainer
        
        Args:
            model_dir: Directory to save models and artifacts
            random_state: Random seed for reproducibility
        """
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.random_state = random_state
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.models = self._initialize_models()
        self.best_model = None
        self.cv_results = {}
        self.feature_importances_ = None
        self.classes_ = None
        self.feature_names = None
        
        # Create necessary subdirectories
        self.artifacts_dir = os.path.join(self.model_dir, 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        logger.info(f"Model trainer initialized. Saving artifacts to: {self.artifacts_dir}")

    def _initialize_models(self) -> dict:
        """Initialize base models with optimized hyperparameters"""
        base_models = {
            'xgb': XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=50,
                tree_method='hist',
                enable_categorical=True,
                use_label_encoder=False
            ),
            'lgbm': LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                importance_type='gain'
            ),
            'catboost': CatBoostClassifier(
                iterations=1000,
                learning_rate=0.01,
                depth=6,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=100,
                task_type='GPU' if self._check_gpu() else 'CPU',
                eval_metric='Logloss',
                early_stopping_rounds=50,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                border_count=254
            ),
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced',
                verbose=1
            )
        }
        
        # Define meta-learner
        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Create stacking ensemble
        stack = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=meta_learner,
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=1
        )
        
        return {
            **{name: model for name, model in base_models.items()},
            'stacking': stack
        }
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for CatBoost"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_data(self, data_path: str, label_col: str = 'label', test_size: float = 0.2) -> tuple:
        """
        Load and preprocess the dataset
        
        Args:
            data_path: Path to the CSV file containing the dataset
            label_col: Name of the target column
            test_size: Fraction of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Separate features and target
            X = df.drop(columns=[label_col], errors='ignore')
            y = df[label_col]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Convert labels to binary (0/1) if needed
            if y.dtype == 'object' or y.nunique() > 2:
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.classes_ = le.classes_
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
            logger.info(f"Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def preprocess_data(self, X_train, X_test):
        """Preprocess the data with scaling and imputation"""
        logger.info("Preprocessing data...")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # Convert back to DataFrame to preserve column names
        X_train_processed = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_processed = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        return X_train_processed, X_test_processed
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, cv_folds=5):
        """
        Train all models with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Dictionary of trained models
        """
        logger.info("Starting model training...")
        
        # Prepare validation data
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        trained_models = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"\nTraining {name}...")
                start_time = time.time()
                
                # Train with early stopping if supported
                if hasattr(model, 'fit') and hasattr(model, 'set_params'):
                    fit_params = {}
                    if 'early_stopping_rounds' in model.get_params() and eval_set:
                        fit_params['eval_set'] = eval_set
                        fit_params['verbose'] = False
                    
                    # Train the model
                    model.fit(X_train, y_train, **fit_params)
                    
                    # Cross-validation
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                    
                    # Store CV results
                    self.cv_results[name] = {
                        'mean_cv_score': np.mean(cv_scores),
                        'std_cv_score': np.std(cv_scores),
                        'cv_scores': cv_scores.tolist(),
                        'training_time': time.time() - start_time
                    }
                    
                    logger.info(f"{name} trained in {self.cv_results[name]['training_time']:.2f}s")
                    logger.info(f"Mean CV ROC-AUC: {self.cv_results[name]['mean_cv_score']:.4f} (±{self.cv_results[name]['std_cv_score']:.4f})")
                    
                    trained_models[name] = model
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}", exc_info=True)
        
        # Select best model based on CV score
        if self.cv_results:
            best_model_name = max(self.cv_results, key=lambda k: self.cv_results[k]['mean_cv_score'])
            self.best_model = trained_models[best_model_name]
            logger.info(f"\nBest model: {best_model_name} (ROC-AUC: {self.cv_results[best_model_name]['mean_cv_score']:.4f})")
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate models on test set and generate reports"""
        logger.info("\nEvaluating models on test set...")
        
        results = {}
        
        for name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                    'log_loss': log_loss(y_test, y_proba) if y_proba is not None else None,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                results[name] = metrics
                
                # Log metrics
                logger.info(f"\n{name} Test Metrics:")
                logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"Precision: {metrics['precision']:.4f}")
                logger.info(f"Recall: {metrics['recall']:.4f}")
                logger.info(f"F1-Score: {metrics['f1']:.4f}")
                if metrics['roc_auc'] is not None:
                    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                
                # Generate visualizations
                self._generate_plots(name, y_test, y_pred, y_proba, metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}", exc_info=True)
        
        return results
    
    def _generate_plots(self, model_name, y_true, y_pred, y_proba, metrics):
        """Generate evaluation plots"""
        try:
            # Create plots directory
            plots_dir = os.path.join(self.artifacts_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.classes_ if self.classes_ is not None else ['0', '1'], 
                        yticklabels=self.classes_ if self.classes_ is not None else ['0', '1'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()
            
            # 2. ROC Curve
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{model_name}_roc_curve.png'))
                plt.close()
            
            # 3. Feature Importance (if available)
            if hasattr(self.models[model_name], 'feature_importances_'):
                importances = self.models[model_name].feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                
                plt.figure(figsize=(10, 8))
                plt.title(f'Feature Importances - {model_name}')
                plt.barh(range(len(indices)), importances[indices], 
                         color='b', align='center')
                plt.yticks(range(len(indices)), 
                           [self.feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{model_name}_feature_importance.png'))
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating plots for {model_name}: {str(e)}", exc_info=True)
    
    def save_models(self, models, save_dir=None):
        """Save trained models to disk"""
        if save_dir is None:
            save_dir = os.path.join(self.model_dir, 'trained_models')
        
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = {}
        for name, model in models.items():
            try:
                # Create a dictionary with model and metadata
                model_data = {
                    'model': model,
                    'feature_names': self.feature_names,
                    'classes': self.classes_.tolist() if self.classes_ is not None else None,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'classifier',
                    'metrics': self.cv_results.get(name, {})
                }
                
                # Save the model
                model_path = os.path.join(save_dir, f'{name}_model.joblib')
                joblib.dump(model_data, model_path)
                saved_paths[name] = model_path
                logger.info(f"Saved {name} model to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {name} model: {str(e)}", exc_info=True)
        
        return saved_paths
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            
            # Update metadata
            if 'feature_names' in model_data:
                self.feature_names = model_data['feature_names']
            if 'classes' in model_data:
                self.classes_ = np.array(model_data['classes'])
            
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            return None
