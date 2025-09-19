import os
import sys
import time
import logging
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from optimized_model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set random seed for reproducibility
np.random.seed(42)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/processed',
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")

def load_data():
    """Load preprocessed data"""
    logger.info("Loading preprocessed data...")
    
    # Setup directories
    setup_directories()
    
    # Check if processed data exists
    required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    missing_files = [f for f in required_files if not os.path.exists(f'data/processed/{f}')]
    
    if missing_files:
        error_msg = f"Missing required data files: {', '.join(missing_files)}. Please run preprocess_data.py first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info("Loading training and test features...")
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        
        # Log data shapes
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        
        # Get feature names
        feature_names = X_train.columns.tolist()
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Load targets
        logger.info("Loading target variables...")
        try:
            y_train = pd.read_csv('data/processed/y_train.csv', header=None, skiprows=1, names=['target']).squeeze()
            y_test = pd.read_csv('data/processed/y_test.csv', header=None, skiprows=1, names=['target']).squeeze()
            
            # Log target statistics
            logger.info(f"y_train shape: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
            logger.info(f"y_test shape: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
            logger.info(f"y_train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Mean: {y_train.mean():.4f}")
            logger.info(f"y_test - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}")
            
        except pd.errors.EmptyDataError as e:
            error_msg = "One or more target files are empty or malformed."
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"Error loading target files: {str(e)}", exc_info=True)
            raise
        
        # Convert to numeric and clean data
        def clean_data(X, y, set_name):
            if len(y) == 0:
                return None, None, []
                
            y = pd.to_numeric(y, errors='coerce')
            valid_idx = y.notna()
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
                valid_idx = valid_idx.iloc[:min_len]
            
            X_clean = X[valid_idx].values
            y_clean = y[valid_idx].values
            
            print(f"{set_name} - X shape: {X_clean.shape}, y shape: {y_clean.shape}")
            return X_clean, y_clean, valid_idx
        
        # Clean training and test data
        X_train_clean, y_train_clean, train_idx = clean_data(X_train, y_train, 'Training')
        X_test_clean, y_test_clean, test_idx = clean_data(X_test, y_test, 'Test')
        
        # Ensure we have valid data
        if X_train_clean is None or X_test_clean is None:
            raise ValueError("Could not load valid training or test data")
        
        return X_train_clean, X_test_clean, y_train_clean, y_test_clean, feature_names
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_and_evaluate():
    """Train and evaluate the model"""
    start_time = time.time()
    logger.info("\n" + "="*80)
    logger.info("Starting Model Training")
    logger.info("="*80)
    
    try:
        # Load data with feature names
        logger.info("Loading training and test data...")
        X_train, X_test, y_train, y_test, feature_names = load_data()
        
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        model_trainer = ModelTrainer()
        
        # Combine train and test for cross-validation
        logger.info("Preparing data for cross-validation...")
        try:
            X = np.vstack((X_train, X_test))
            y = np.concatenate((y_train, y_test))
            logger.info(f"Combined data - X shape: {X.shape}, y shape: {y.shape}")
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}", exc_info=True)
            raise
        
        # Train the model
        logger.info("\n=== Starting Model Training ===")
        logger.info(f"Feature names: {feature_names[:5]}... (total: {len(feature_names)} features)")
        
        # Train the model with feature names
        logger.info("Training ensemble model...")
        results = model_trainer.train_ensemble(X, y, test_size=0.2, feature_names=feature_names)
        model = results['model']
        
        # Log training time
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        return True
        
        # Evaluate on test set
        print("\n=== Model Evaluation ===")
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        print("\nRegression Metrics:")
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Plot predictions vs actual values
        print("\nGenerating prediction plots...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Predictions vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # Plot feature importances if available
        if hasattr(model, 'feature_importances_'):
            print("Generating feature importance plot...")
            plt.figure(figsize=(12, 8))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Show top 20 features
            
            # Create a horizontal bar plot
            plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title('Top 20 Feature Importances')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('reports/feature_importances.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save the model with feature names and metadata
        print("\nSaving model...")
        model_path = 'models/optimized_deepfake_model.joblib'
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2
            },
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_shape': X_train.shape[1:],
            'model_type': model.__class__.__name__
        }
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_data, model_path)
        
        # Save metrics to a text file
        with open('reports/model_metrics.txt', 'w') as f:
            f.write(f"Model Training Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Training Date: {model_data['training_date']}\n")
            f.write(f"Model Type: {model_data['model_type']}\n")
            f.write(f"Input Shape: {model_data['input_shape']}\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"- Mean Squared Error: {mse:.4f}\n")
            f.write(f"- Mean Absolute Error: {mae:.4f}\n")
            f.write(f"- R² Score: {r2:.4f}\n")
        
        print(f"\nModel and training report saved to {os.path.abspath('models')}")
        print("\n=== Training Complete ===")
        return True
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("DeepTrust Model Training Script")
        print("="*80)
        print("Logs are being saved to 'training.log'\n")
        
        start_time = time.time()
        success = train_and_evaluate()
        end_time = time.time()
        
        print("\n" + "="*80)
        if success:
            print(f"\n✓ Training completed successfully in {(end_time - start_time)/60:.2f} minutes")
            print("✓ Model has been saved to the 'models' directory")
            print("✓ Training reports and plots are available in the 'reports' directory")
            print("\nNext steps:")
            print("1. Check the 'reports' directory for model performance metrics and visualizations")
            print("2. Use the trained model for predictions with 'predict.py'")
        else:
            print("\n✗ Training failed. Please check the error messages above and in 'training.log'")
            print("\nTroubleshooting tips:")
            print("1. Verify that your input data is correctly formatted")
            print("2. Check for missing or infinite values in your dataset")
            print("3. Ensure you have enough system resources (CPU/RAM)")
            print("4. Review the detailed error logs in 'training.log'")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.critical("Critical error during training", exc_info=True)
        print(f"\n✗ A critical error occurred: {str(e)}")
        print("Please check 'training.log' for detailed error information.\n")
        sys.exit(1)
