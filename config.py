# config.py
import json
import os

class DeepTrustConfig:
    def __init__(self, config_file="deeptrust_config.json"):
        self.config_file = config_file
        self.default_config = {
            "paths": {
                "models": ".",
                "data": ".",
                "results": "evaluation/results"
            },
            "feature_extraction": {
                "mfcc": True,
                "facial_landmarks": True,
                "lip_sync": True,
                "optical_flow": False
            },
            "model_params": {
                "input_size": 300,
                "hidden_layers": [512, 256, 128],
                "dropout_rate": 0.5
            },
            "evaluation": {
                "test_size": 0.2,
                "cross_validation_folds": 5,
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
            }
        }
        self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.default_config
            self.save_config()
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

# Initialize configuration
config = DeepTrustConfig()