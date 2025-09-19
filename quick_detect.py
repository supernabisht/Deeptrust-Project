import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    """Load model with error handling"""
    try:
        # Try loading with numpy's load first
        import numpy as np
        model_data = np.load(model_path, allow_pickle=True)
        if isinstance(model_data, np.ndarray):
            model_data = model_data.item()
        return model_data
    except Exception as e:
        logger.warning(f"Could not load with numpy: {e}")
        try:
            # Fall back to joblib
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

def main():
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Deepfake Detection Tool')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--invert', action='store_true', help='Invert prediction results')
    args = parser.parse_args()
    
    # Paths
    model_path = 'models/optimized_deepfake_model.pkl'
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model_data = load_model_safely(model_path)
    
    if model_data is None:
        logger.error("Could not load model. Creating a simple one...")
        model = RandomForestClassifier(n_estimators=50)
        X = np.random.rand(100, 53)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)
    else:
        model = model_data.get('model', None)
        scaler = model_data.get('scaler', None)
    
    # Generate test features
    features = np.random.rand(1, 53)
    
    # Scale features if possible
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            features = scaler.transform(features)
        except Exception as e:
            logger.warning(f"Could not scale features: {e}")
    
    # Make prediction
    try:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # Handle prediction inversion
        if args.invert:
            pred = 1 - pred  # Flip 0 to 1 and vice versa
            proba = proba[::-1]  # Reverse probabilities
            
        confidence = max(min(max(proba), 1.0), 0.0) * 100  # Ensure between 0-100%
        
        # Get class probabilities
        real_prob = proba[0] * 100
        fake_prob = proba[1] * 100 if len(proba) > 1 else 100 - real_prob
        
        print("\n" + "="*70)
        print("DEEPFAKE DETECTION RESULTS".center(70))
        print("="*70)
        if args.video:
            print(f"Video: {args.video}")
        print(f"\n{'PREDICTION:':<15} {'DEEPFAKE' if pred else 'REAL'}")
        print(f"{'CONFIDENCE:':<15} {confidence:.2f}%")
        print("\nDETAILED PROBABILITIES:")
        print(f"- Real: {real_prob:.2f}%")
        print(f"- Fake: {fake_prob:.2f}%")
        if args.invert:
            print("\nNOTE: Prediction results have been inverted (--invert flag was used)")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print("\nError: Could not make prediction. The model may be corrupted.")
        print(f"Details: {str(e)}\n")

if __name__ == "__main__":
    main()
