import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Union, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepfakePredictor:
    """
    A class to load and use the trained Deepfake detection model.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str, optional): Path to the trained model file. 
                                      If None, looks in the 'models' directory.
        """
        self.model = None
        self.feature_names = None
        self.model_metadata = {}
        self.model_path = model_path or self._find_latest_model()
        
        if self.model_path:
            self.load_model()
    
    def _find_latest_model(self) -> str:
        """Find the latest model in the models directory."""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            logger.error("Models directory not found!")
            return None
            
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith(('.pkl', '.joblib', '.pkl.gz', '.joblib.gz'))]
        
        if not model_files:
            logger.error("No model files found in the models directory!")
            return None
            
        # Get the most recently modified model file
        latest_model = max(
            model_files, 
            key=lambda x: os.path.getmtime(os.path.join(models_dir, x))
        )
        return os.path.join(models_dir, latest_model)
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load the trained model and its metadata.
        
        Args:
            model_path (str, optional): Path to the model file.
                                      If None, uses the path provided during initialization.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if model_path:
            self.model_path = model_path
            
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
            
        try:
            logger.info(f"Loading model from: {self.model_path}")
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.model_metadata = {
                    'training_date': model_data.get('training_date', 'Unknown'),
                    'model_type': model_data.get('model_type', 'Unknown'),
                    'metrics': model_data.get('metrics', {})
                }
            else:
                self.model = model_data
                logger.warning("Loaded model doesn't contain metadata. Feature names unknown.")
            
            if self.model is None:
                logger.error("Failed to load model from the file.")
                return False
                
            logger.info(f"Successfully loaded {self.model_metadata.get('model_type', 'model')} "
                      f"trained on {self.model_metadata.get('training_date', 'unknown date')}")
            
            if 'metrics' in self.model_metadata:
                logger.info("Model performance metrics:")
                for metric, value in self.model_metadata['metrics'].items():
                    logger.info(f"  - {metric.upper()}: {value:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, Dict[str, Any]], 
               return_proba: bool = False) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input data for prediction. Can be:
               - numpy array (shape: n_samples, n_features)
               - pandas DataFrame with feature columns
               - Dictionary with feature names as keys and values as lists/arrays
            return_proba: If True, returns probability scores (only for classification).
                        For regression models, this will be the same as predict.
                        
        Returns:
            numpy.ndarray: Predictions (and probabilities if return_proba=True)
        """
        if self.model is None:
            logger.error("Model not loaded! Call load_model() first.")
            return None
            
        try:
            # Convert input to the correct format
            if isinstance(X, dict):
                X_df = pd.DataFrame(X)
                # Reorder columns to match training data if feature names are known
                if self.feature_names:
                    missing = set(self.feature_names) - set(X_df.columns)
                    if missing:
                        logger.warning(f"Missing features: {missing}")
                    # Add missing columns with NaN
                    for col in missing:
                        X_df[col] = np.nan
                    # Reorder columns to match training data
                    X_df = X_df[self.feature_names]
                X_array = X_df.values
            elif isinstance(X, pd.DataFrame):
                if self.feature_names:
                    # Reorder columns to match training data
                    missing = set(self.feature_names) - set(X.columns)
                    if missing:
                        logger.warning(f"Missing features: {missing}")
                    # Add missing columns with NaN
                    for col in missing:
                        X[col] = np.nan
                    X = X[self.feature_names]
                X_array = X.values
            else:  # numpy array
                X_array = X
                if self.feature_names and X_array.shape[1] != len(self.feature_names):
                    logger.warning(f"Number of features ({X_array.shape[1]}) doesn't match "
                                 f"the number of features in the model ({len(self.feature_names)})")
            
            # Make predictions
            predictions = self.model.predict(X_array)
            
            # If the model supports predict_proba and it was requested
            if return_proba and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_array)
                return predictions, proba
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            return None
    
    def get_feature_importances(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importances from the model if available.
        
        Args:
            top_n: Number of top features to return.
            
        Returns:
            pd.DataFrame: DataFrame with feature names and their importance scores.
        """
        if self.model is None:
            logger.error("Model not loaded! Call load_model() first.")
            return None
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                logger.warning("Model doesn't support feature importances.")
                return None
                
            if self.feature_names and len(self.feature_names) == len(importances):
                feature_importances = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                })
            else:
                logger.warning("Feature names not available or don't match importance scores.")
                feature_importances = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importances))],
                    'importance': importances
                })
            
            return feature_importances.sort_values('importance', ascending=False).head(top_n)
            
        except Exception as e:
            logger.error(f"Error getting feature importances: {str(e)}", exc_info=True)
            return None

def example_usage():
    """Example of how to use the DeepfakePredictor class."""
    # Initialize the predictor (automatically loads the latest model)
    predictor = DeepfakePredictor()
    
    if predictor.model is None:
        print("Failed to load model. Please check the logs.")
        return
    
    # Example 1: Predict using a dictionary of features
    sample_data = {
        'mfcc_1_mean': [0.1, 0.2],
        'mfcc_1_std': [0.5, 0.4],
        # Add more features as needed
    }
    
    print("\nExample 1: Predict using dictionary input")
    predictions = predictor.predict(sample_data)
    print(f"Predictions: {predictions}")
    
    # Example 2: Get feature importances
    print("\nTop 10 most important features:")
    feature_importances = predictor.get_feature_importances(10)
    if feature_importances is not None:
        print(feature_importances)
    
    # Example 3: Load data from CSV and make predictions
    try:
        if os.path.exists('data/processed/X_test.csv'):
            print("\nExample 3: Predict on test data")
            X_test = pd.read_csv('data/processed/X_test.csv')
            predictions = predictor.predict(X_test)
            print(f"First 5 predictions: {predictions[:5]}")
    except Exception as e:
        print(f"\nCould not load test data: {str(e)}")

if __name__ == "__main__":
    print("Deepfake Detection Model Predictor")
    print("==================================")
    example_usage()
