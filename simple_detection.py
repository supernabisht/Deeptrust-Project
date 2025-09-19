import os
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    """Safely load and verify the model"""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        logger.info(f"Attempting to load model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Check if it's a dictionary with required keys
        if not isinstance(model_data, dict):
            logger.error("Model file is not in the expected format (not a dictionary)")
            return None
            
        logger.info("Model loaded successfully. Contents:")
        for key, value in model_data.items():
            logger.info(f"- {key}: {type(value).__name__}")
            
        return model_data
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Try to find any model file
    model_dir = Path('models')
    if not model_dir.exists():
        logger.error("Models directory not found!")
        return
        
    # List all model files
    model_files = list(model_dir.glob('*.pkl'))
    if not model_files:
        logger.error("No model files found in the models directory!")
        return
        
    logger.info(f"Found {len(model_files)} model files. Checking each one...")
    
    # Try to load each model
    for model_file in model_files:
        logger.info(f"\nChecking model: {model_file}")
        model_data = load_model_safely(model_file)
        
        if model_data:
            logger.info(f"Successfully loaded model: {model_file}")
            # If we found a working model, update the main script to use it
            with open('run_enhanced_detection.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Update the model path in the main script
            normalized_path = str(model_file).replace('\\', '/')
            new_content = content.replace(
                "self.model_path = model_path or 'models/ultimate_deepfake_model.pkl'",
                f"self.model_path = model_path or r'{normalized_path}'"
            )
            
            with open('run_enhanced_detection.py', 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            logger.info(f"Updated run_enhanced_detection.py to use model: {model_file}")
            return
            
    logger.error("No valid models found. You may need to train a new model.")
    logger.info("Run 'python optimized_model_trainer.py' to train a new model.")

if __name__ == "__main__":
    main()
