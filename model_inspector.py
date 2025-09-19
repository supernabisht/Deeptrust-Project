import joblib
import os
import sys

def inspect_model(model_path):
    """Inspect the model to understand its expected features"""
    try:
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)
        
        print("\n=== Model Information ===")
        print(f"Model type: {type(model_data.get('model')).__name__}")
        
        model = model_data.get('model')
        
        # Check for feature names
        if hasattr(model, 'feature_names_in_'):
            print("\nExpected features:")
            for i, name in enumerate(model.feature_names_in_):
                print(f"{i}: {name}")
        elif hasattr(model, 'feature_importances_'):
            print(f"\nModel has {len(model.feature_importances_)} features")
            print("Feature importances:", model.feature_importances_[:10], "...")
        
        # Check for classes
        if 'classes' in model_data:
            print("\nClasses:", model_data['classes'])
        
        # Check for other metadata
        print("\nModel metadata:")
        for key in model_data:
            if key != 'model':
                print(f"{key}: {model_data[key]}" if len(str(model_data[key])) < 100 else f"{key}: {str(model_data[key])[:100]}...")
    
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Look for any model in the trained_models directory
        models_dir = os.path.join('enhanced_models', 'trained_models')
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        if not model_files:
            print("No model files found in enhanced_models/trained_models/")
            sys.exit(1)
            
        model_path = os.path.join(models_dir, model_files[0])
        print(f"Using model: {model_path}")
    
    inspect_model(model_path)
