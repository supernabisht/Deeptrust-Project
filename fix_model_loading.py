import joblib
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_model_integrity(model_path):
    """Check if the model file exists and can be loaded"""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        model_data = joblib.load(model_path)
        # Check if it's a dictionary with required keys
        if not isinstance(model_data, dict):
            logger.error("Model file is not in the expected format (not a dictionary)")
            return False
            
        required_keys = ['model', 'feature_columns', 'scaler']
        for key in required_keys:
            if key not in model_data:
                logger.warning(f"Model file is missing key: {key}")
                
        logger.info("Model file appears to be valid")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def find_latest_model(model_dir='models'):
    """Find the most recent model file"""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None
        
    model_files = list(model_dir.glob('*.pkl'))
    if not model_files:
        return None
        
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    return str(model_files[0])

def main():
    # First try the default model path
    default_model = 'models/ultimate_deepfake_model.pkl'
    
    if check_model_integrity(default_model):
        logger.info(f"Default model is valid: {default_model}")
        return
        
    # If default model is invalid, try to find another one
    logger.warning("Default model is invalid, searching for alternatives...")
    
    # Look for other model files
    latest_model = find_latest_model()
    
    if latest_model and check_model_integrity(latest_model):
        logger.info(f"Found valid model: {latest_model}")
        # Update the model path in run_enhanced_detection.py
        update_model_path(latest_model)
    else:
        logger.error("No valid model found. You may need to train a new model.")
        logger.info("Run 'python optimized_model_trainer.py' to train a new model.")

def update_model_path(new_path):
    """Update the model path in run_enhanced_detection.py"""
    try:
        with open('run_enhanced_detection.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Update the model path - use raw string and forward slashes
        normalized_path = str(new_path).replace('\\', '/')
        new_content = content.replace(
            "self.model_path = model_path or 'models/ultimate_deepfake_model.pkl'",
            f"self.model_path = model_path or r'{normalized_path}'"
        )
        
        with open('run_enhanced_detection.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        logger.info(f"Updated run_enhanced_detection.py to use model: {new_path}")
        
    except Exception as e:
        logger.error(f"Error updating model path: {str(e)}")

if __name__ == "__main__":
    main()
