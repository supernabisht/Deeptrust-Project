# optimized_model_trainer.py
import numpy as np
import pandas as pd
import os
import joblib
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Union
import logging

# Data processing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
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
        
        # Create necessary subdirectories
        self.artifacts_dir = os.path.join(self.model_dir, 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        logger.info(f"Model trainer initialized. Saving artifacts to: {self.artifacts_dir}")

    def _initialize_models(self) -> dict:
        """Initialize base models with optimized hyperparameters and improved defaults"""
        base_models = {
            'xgb': XGBClassifier(
                n_estimators=2000,  # Increased for better convergence
                learning_rate=0.005,  # Reduced learning rate
                max_depth=8,  # Slightly deeper trees
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,  # Increased regularization
                min_child_weight=5,  # Increased for more conservative model
                reg_alpha=0.2,  # Increased L1 regularization
                reg_lambda=1.5,  # Increased L2 regularization
                random_state=self.random_state,
                n_jobs=-1,
                objective='binary:logistic',
                eval_metric='aucpr',  # Better for imbalanced data
                early_stopping_rounds=100,  # More patience
                tree_method='hist',
                enable_categorical=True,
                use_label_encoder=False,  # Avoid deprecation warning
                scale_pos_weight=2.0,  # Handle class imbalance
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
            tuple: (X_train, X_test, y_train, y_test, feature_names)
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
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.classes_ = le.classes_
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=self.random_state,
                stratify=y if y.nunique() > 1 else None
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
                        xticklabels=self.classes_, yticklabels=self.classes_)
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
            )
                verbose=0,
                allow_writing_files=False,  # Don't write any files
                loss_function='RMSE',  # Explicitly set loss function
                eval_metric='RMSE',  # Evaluation metric
                early_stopping_rounds=20,  # Stop training if no improvement
                use_best_model=True  # Use the best model during training
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        }
        return base_models

    def _train_meta_model(self, X_train, y_train, base_results):
        """Train a meta-model on the outputs of base models"""
        try:
            # Get predictions from base models
            meta_features = []
            for name, result in base_results.items():
                model = result['model']
                preds = model.predict_proba(X_train)[:, 1]
                meta_features.append(preds)
            
            # Stack predictions
            X_meta = np.column_stack(meta_features)
            
            # Train logistic regression as meta-model
            meta_model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            meta_model.fit(X_meta, y_train)
            return meta_model
            
        except Exception as e:
            logger.error(f"Error training meta-model: {str(e)}")
            return None

    def train_ensemble(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, feature_names=None, n_splits: int = 5):
        """Train and evaluate the ensemble model with improved validation
        
        Args:
            X: Feature matrix
            y: Target values
            test_size: Proportion of data to use for testing
            feature_names: List of feature names (optional)
            n_splits: Number of cross-validation folds
        """
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Handle class imbalance
        class_weights = {0: 1.0, 1: len(y[y==0]) / len(y[y==1])}  # Inverse ratio weighting
        
        # Initialize models with class weights where supported
        self.models = self._initialize_models()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Dictionary to store results
        results = {}
        best_score = 0
        
        # Train and evaluate each model
        for name, model in self.models.items():
            if len(X) != len(y):
                raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
                
            # Convert target to float if needed
            y = y.astype(float)
            
            # Check for NaN or infinite values
            if np.isnan(X).any() or np.isinf(X).any():
                raise ValueError("X contains NaN or infinite values")
                
            if np.isnan(y).any() or np.isinf(y).any():
                raise ValueError("y contains NaN or infinite values")
            
            print(f"\n=== Data Summary ===")
            print(f"Samples: {len(X)}")
            print(f"Features: {X.shape[1] if len(X.shape) > 1 else 1}")
            print(f"Target range: {y.min():.4f} to {y.max():.4f}")
            print(f"Mean target: {y.mean():.4f}, Std: {y.std():.4f}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            print("\n=== Training Data Summary ===")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            
            # Scale features
            print("\nScaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train base models
            base_results = {}
            for name, model in self.models.items():
                try:
                    print(f"\n=== Training {name.upper()} ===")
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    print(f"\n{name.upper()} Results:")
                    print(f"MSE: {mse:.4f}")
                    print(f"MAE: {mae:.4f}")
                    print(f"R²: {r2:.4f}")
                    
                    # Save results
                    base_results[name] = {
                        'model': model,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'predictions': y_pred
                    }
                    
                    # Plot predictions vs actual
                    self._plot_predictions(y_test, y_pred, name)
                    
                except Exception as e:
                    print(f"\nError training {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not base_results:
                raise RuntimeError("All base models failed to train")
            
            # Train stacking regressor with successful models
            print("\n=== Training Stacking Regressor ===")
            estimators = [(name, result['model']) for name, result in base_results.items()]
            
            try:
                self.best_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                    n_jobs=-1
                )
                
                self.best_model.fit(X_train_scaled, y_train)
                
                # Evaluate stacking
                y_pred = self.best_model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print("\nStacking Regressor Results:")
                print(f"MSE: {mse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
                
                # Plot predictions vs actual for stacking
                self._plot_predictions(y_test, y_pred, 'stacking')
                
                # Save the model with feature names if available
                self._save_model(feature_names if feature_names is not None else [f'f{i}' for i in range(X.shape[1])])
                
                return {
                    'model': self.best_model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'base_results': base_results
                }
                
            except Exception as e:
                print(f"\nError training stacking regressor: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # If stacking fails, return the best performing base model
                best_model_name = min(base_results.items(), key=lambda x: x[1]['mse'])[0]
                print(f"\nUsing best performing base model: {best_model_name}")
                self.best_model = base_results[best_model_name]['model']
                
                return {
                    'model': self.best_model,
                    **base_results[best_model_name]
                }
                
        except Exception as e:
            print(f"\nError in train_ensemble: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _plot_predictions(self, y_true, y_pred, model_name):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(f'Predictions vs Actual - {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Add metrics to plot
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, 
                f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig(f'reports/predictions_{model_name}.png')
        plt.close()
    
    def _save_model(self, feature_columns: list):
        """Save the trained model and metadata with versioning
        
        Args:
            feature_columns: List of feature names
        """
        if not hasattr(self, 'best_model') or self.best_model is None:
            logger.warning("No trained model to save")
            return
            
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.model_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(self.best_model['model'], model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model['name'],
            'training_date': timestamp,
            'feature_columns': feature_columns,
            'metrics': self.best_model['metrics'],
            'model_type': type(self.best_model['model']).__name__,
            'feature_importances': self.feature_importances_.tolist() if hasattr(self, 'feature_importances_') else None
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importances plot
        if hasattr(self, 'feature_importances_') and feature_columns is not None:
            self._plot_feature_importances(
                self.feature_importances_,
                feature_columns,
                os.path.join(model_dir, 'feature_importances.png')
            )
        
        logger.info(f"Model saved to {model_dir}")
        
        # Update latest model symlink (Windows compatible)
        latest_path = os.path.join(self.model_dir, 'latest')
        if os.path.exists(latest_path):
            if os.path.islink(latest_path) or os.path.isfile(latest_path):
                os.remove(latest_path)
            elif os.path.isdir(latest_path):
                import shutil
                shutil.rmtree(latest_path)
        
        # On Windows, create a junction instead of a symlink
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(model_dir, latest_path)
        else:
            os.symlink(model_dir, latest_path, target_is_directory=True)
            
        return model_dir
            latest_path = os.path.join(self.model_dir, 'latest_model.pkl')
            
            # Save both versions
            joblib.dump(model_metadata, model_path)
            joblib.dump(model_metadata, latest_path)
            
            print(f"\nModel saved to {model_path}")
            print(f"Latest model also saved to {latest_path}")
            
            # Save feature importances if available
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Save feature importances to CSV
                feature_importance_df = pd.DataFrame({
                    'feature': [feature_columns[i] for i in indices],
                    'importance': importances[indices]
                })
                
                importance_path = os.path.join(self.model_dir, 'feature_importances.csv')
                feature_importance_df.to_csv(importance_path, index=False)
                print(f"Feature importances saved to {importance_path}")
                
        except Exception as e:
            print(f"\nError saving model: {str(e)}")
            import traceback
            traceback.print_exc()