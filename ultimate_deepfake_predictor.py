# ultimate_deepfake_predictor.py
import os
import numpy as np
import joblib
from pathlib import Path

class UltimateDeepfakePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or "models/ultimate_deepfake_model.pkl"
        
    def load_model(self):
        """Load the trained model."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction on input features."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not loaded and could not be loaded from disk")
        
        try:
            if isinstance(features, (list, np.ndarray)):
                features = np.array(features).reshape(1, -1)
            
            prediction = self.model.predict_proba(features)
            return {
                'prediction': int(prediction[0][1] > 0.5),
                'confidence': float(prediction[0][1]),
                'class_probs': {
                    'real': float(prediction[0][0]),
                    'fake': float(prediction[0][1])
                }
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

def create_default_model():
    """Create a default model if none exists."""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ultimate_deepfake_model.pkl")
    print("Created default model at models/ultimate_deepfake_model.pkl")
    return model

if __name__ == "__main__":
    create_default_model()