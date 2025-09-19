"""
Target Advanced Threshold Calibration System for Deepfake Detection
Implements machine learning-based confidence scoring and adaptive thresholds
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime

class AdvancedThresholdCalibrator:
    """
    🧠 Advanced threshold calibration using machine learning
    """
    
    def __init__(self):
        self.confidence_model = None
        self.authenticity_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.calibration_history = []
        
    def extract_calibration_features(self, lip_sync_results):
        """
        Checking Extract comprehensive features for threshold calibration
        """
        features = {}
        
        # Basic metrics
        features['mismatch_score'] = lip_sync_results.get('mismatch_score', 0)
        features['final_score'] = lip_sync_results.get('final_score', 0)
        features['confidence'] = lip_sync_results.get('confidence', 0)
        
        # Correlation features
        corr_scores = lip_sync_results.get('correlation_scores', {})
        features['pearson_correlation'] = abs(corr_scores.get('pearson', 0))
        features['cross_correlation'] = corr_scores.get('cross_correlation', 0)
        features['coherence'] = corr_scores.get('coherence', 0)
        features['dtw_similarity'] = corr_scores.get('dtw_similarity', 0)
        
        # Motion statistics
        motion_stats = lip_sync_results.get('motion_stats', {})
        features['motion_mean'] = motion_stats.get('mean', 0)
        features['motion_std'] = motion_stats.get('std', 0)
        features['motion_max'] = motion_stats.get('max', 0)
        features['motion_energy'] = motion_stats.get('energy', 0)
        
        # Audio statistics
        audio_stats = lip_sync_results.get('audio_stats', {})
        features['audio_mean'] = audio_stats.get('mean', 0)
        features['audio_std'] = audio_stats.get('std', 0)
        features['audio_max'] = audio_stats.get('max', 0)
        features['audio_energy'] = audio_stats.get('energy', 0)
        
        # Confidence components
        conf_components = lip_sync_results.get('confidence_components', {})
        for comp_name, comp_value in conf_components.items():
            features[f'conf_{comp_name}'] = comp_value
        
        # Derived features
        features['sync_quality'] = max(0, 1 - features['mismatch_score'])
        features['correlation_consistency'] = 1 - abs(features['pearson_correlation'] - features['cross_correlation'])
        features['signal_ratio'] = features['motion_energy'] / max(features['audio_energy'], 0.001)
        features['noise_ratio'] = features['motion_std'] / max(features['motion_mean'], 0.001)
        
        return features
    
    def create_synthetic_training_data(self, n_samples=1000):
        """
        🎲 Create synthetic training data for initial calibration
        """
        np.random.seed(42)
        
        # Generate realistic feature distributions
        data = []
        labels_confidence = []
        labels_authenticity = []
        
        for i in range(n_samples):
            # Simulate different video types
            if i < n_samples * 0.3:  # High quality real videos
                base_quality = np.random.uniform(0.7, 0.95)
                authenticity = 1  # Real
                confidence_target = np.random.uniform(0.8, 0.95)
            elif i < n_samples * 0.6:  # Medium quality real videos
                base_quality = np.random.uniform(0.5, 0.8)
                authenticity = 1  # Real
                confidence_target = np.random.uniform(0.6, 0.85)
            elif i < n_samples * 0.8:  # Low quality real videos
                base_quality = np.random.uniform(0.3, 0.6)
                authenticity = 1  # Real
                confidence_target = np.random.uniform(0.4, 0.7)
            else:  # Fake videos
                base_quality = np.random.uniform(0.1, 0.5)
                authenticity = 0  # Fake
                confidence_target = np.random.uniform(0.2, 0.6)
            
            # Generate correlated features
            mismatch = 1 - base_quality + np.random.normal(0, 0.1)
            mismatch = max(0, min(1, mismatch))
            
            pearson = base_quality + np.random.normal(0, 0.15)
            pearson = max(0, min(1, pearson))
            
            cross_corr = base_quality * 0.8 + np.random.normal(0, 0.1)
            cross_corr = max(0, min(1, cross_corr))
            
            coherence = base_quality * 0.7 + np.random.normal(0, 0.12)
            coherence = max(0, min(1, coherence))
            
            features = {
                'mismatch_score': mismatch,
                'final_score': mismatch * np.random.uniform(0.8, 1.2),
                'confidence': confidence_target + np.random.normal(0, 0.05),
                'pearson_correlation': pearson,
                'cross_correlation': cross_corr,
                'coherence': coherence,
                'dtw_similarity': base_quality * 0.6 + np.random.normal(0, 0.1),
                'motion_mean': np.random.uniform(0.1, 0.8),
                'motion_std': np.random.uniform(0.05, 0.3),
                'motion_max': np.random.uniform(0.5, 1.0),
                'motion_energy': np.random.uniform(0.1, 0.9),
                'audio_mean': np.random.uniform(0.1, 0.7),
                'audio_std': np.random.uniform(0.05, 0.25),
                'audio_max': np.random.uniform(0.4, 1.0),
                'audio_energy': np.random.uniform(0.1, 0.8),
                'conf_sync_quality': base_quality * 0.35,
                'conf_pearson_correlation': pearson * 0.25,
                'conf_cross_correlation': cross_corr * 0.20,
                'conf_coherence': coherence * 0.15,
                'conf_dtw_similarity': base_quality * 0.05,
                'sync_quality': base_quality,
                'correlation_consistency': 1 - abs(pearson - cross_corr),
                'signal_ratio': np.random.uniform(0.5, 2.0),
                'noise_ratio': np.random.uniform(0.1, 1.5)
            }
            
            data.append(features)
            labels_confidence.append(confidence_target)
            labels_authenticity.append(authenticity)
        
        return pd.DataFrame(data), np.array(labels_confidence), np.array(labels_authenticity)
    
    def train_calibration_models(self, training_data=None, confidence_labels=None, authenticity_labels=None):
        """
        🎓 Train machine learning models for confidence and authenticity prediction
        """
        print("? Training advanced calibration models...")
        
        if training_data is None:
            print("Report Generating synthetic training data...")
            training_data, confidence_labels, authenticity_labels = self.create_synthetic_training_data()
        
        # Prepare features
        X = self.scaler.fit_transform(training_data)
        
        # Train confidence prediction model
        print("Target Training confidence prediction model...")
        self.confidence_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Convert confidence to categories for classification
        confidence_categories = np.digitize(confidence_labels, bins=[0, 0.3, 0.6, 0.8, 1.0]) - 1
        self.confidence_model.fit(X, confidence_categories)
        
        # Train authenticity prediction model
        print("Checking Training authenticity prediction model...")
        self.authenticity_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.authenticity_model.fit(X, authenticity_labels)
        
        # Calculate feature importance
        feature_names = list(training_data.columns)
        self.feature_importance = dict(zip(
            feature_names,
            self.authenticity_model.feature_importances_
        ))
        
        # Evaluate models
        conf_score = cross_val_score(self.confidence_model, X, confidence_categories, cv=5).mean()
        auth_score = cross_val_score(self.authenticity_model, X, authenticity_labels, cv=5).mean()
        
        print(f"SUCCESS: Confidence model accuracy: {conf_score:.3f}")
        print(f"SUCCESS: Authenticity model accuracy: {auth_score:.3f}")
        
        # Save models
        self.save_models()
        
        return conf_score, auth_score
    
    def predict_advanced_metrics(self, lip_sync_results):
        """
        🔮 Predict advanced confidence and authenticity metrics
        """
        if self.confidence_model is None or self.authenticity_model is None:
            print("WARNING: Models not trained. Training with synthetic data...")
            self.train_calibration_models()
        
        # Extract features
        features = self.extract_calibration_features(lip_sync_results)
        feature_df = pd.DataFrame([features])
        
        # Handle missing columns
        expected_features = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else list(features.keys())
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Scale features
        X = self.scaler.transform(feature_df[expected_features])
        
        # Predict confidence category and convert back to probability
        conf_category = self.confidence_model.predict(X)[0]
        conf_proba = self.confidence_model.predict_proba(X)[0]
        
        # Map category back to confidence value
        confidence_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        if conf_category < len(confidence_ranges):
            conf_min, conf_max = confidence_ranges[conf_category]
            predicted_confidence = conf_min + (conf_max - conf_min) * conf_proba[conf_category]
        else:
            predicted_confidence = 0.5
        
        # Predict authenticity
        authenticity_proba = self.authenticity_model.predict_proba(X)[0]
        authenticity_score = authenticity_proba[1]  # Probability of being real
        
        # Get feature importance for this prediction
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'ml_confidence': predicted_confidence,
            'authenticity_probability': authenticity_score,
            'confidence_category': conf_category,
            'confidence_distribution': conf_proba.tolist(),
            'authenticity_distribution': authenticity_proba.tolist(),
            'top_features': top_features,
            'feature_values': features
        }
    
    def calibrate_thresholds(self, lip_sync_results):
        """
        🎛️ Dynamically calibrate thresholds based on ML predictions
        """
        ml_predictions = self.predict_advanced_metrics(lip_sync_results)
        
        # Base thresholds
        base_thresholds = {
            'excellent': 0.75,
            'good': 0.65,
            'moderate': 0.50,
            'poor': 0.35
        }
        
        # Adjust thresholds based on ML confidence
        ml_confidence = ml_predictions['ml_confidence']
        authenticity_prob = ml_predictions['authenticity_probability']
        
        # Confidence-based adjustment
        if ml_confidence > 0.8:
            threshold_multiplier = 1.1  # Stricter thresholds for high confidence
        elif ml_confidence > 0.6:
            threshold_multiplier = 1.0  # Standard thresholds
        else:
            threshold_multiplier = 0.9  # More lenient thresholds for low confidence
        
        # Authenticity-based adjustment
        if authenticity_prob > 0.7:
            authenticity_bonus = 0.05  # Slight bonus for likely real videos
        elif authenticity_prob < 0.3:
            authenticity_bonus = -0.05  # Penalty for likely fake videos
        else:
            authenticity_bonus = 0
        
        # Apply adjustments
        calibrated_thresholds = {}
        for level, threshold in base_thresholds.items():
            adjusted = threshold * threshold_multiplier + authenticity_bonus
            calibrated_thresholds[level] = max(0.1, min(0.95, adjusted))
        
        return calibrated_thresholds, ml_predictions
    
    def generate_advanced_report(self, lip_sync_results):
        """
        Report Generate comprehensive analysis report with ML insights
        """
        calibrated_thresholds, ml_predictions = self.calibrate_thresholds(lip_sync_results)
        
        # Calculate final assessment
        sync_quality = max(0, 1 - lip_sync_results.get('mismatch_score', 1))
        ml_confidence = ml_predictions['ml_confidence']
        authenticity_prob = ml_predictions['authenticity_probability']
        
        # Determine final classification
        if sync_quality > calibrated_thresholds['excellent'] and authenticity_prob > 0.7:
            final_classification = "AUTHENTIC"
            confidence_level = "VERY HIGH"
        elif sync_quality > calibrated_thresholds['good'] and authenticity_prob > 0.5:
            final_classification = "LIKELY AUTHENTIC"
            confidence_level = "HIGH"
        elif sync_quality > calibrated_thresholds['moderate']:
            final_classification = "UNCERTAIN"
            confidence_level = "MODERATE"
        elif sync_quality > calibrated_thresholds['poor']:
            final_classification = "LIKELY FAKE"
            confidence_level = "LOW"
        else:
            final_classification = "FAKE"
            confidence_level = "VERY LOW"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'final_classification': final_classification,
            'confidence_level': confidence_level,
            'scores': {
                'sync_quality': sync_quality,
                'ml_confidence': ml_confidence,
                'authenticity_probability': authenticity_prob,
                'mismatch_score': lip_sync_results.get('mismatch_score', 0),
                'final_score': lip_sync_results.get('final_score', 0)
            },
            'calibrated_thresholds': calibrated_thresholds,
            'ml_predictions': ml_predictions,
            'original_results': lip_sync_results
        }
        
        return report
    
    def save_models(self, model_dir="models"):
        """
        💾 Save trained models and scaler
        """
        os.makedirs(model_dir, exist_ok=True)
        
        if self.confidence_model is not None:
            joblib.dump(self.confidence_model, os.path.join(model_dir, 'confidence_model.pkl'))
        
        if self.authenticity_model is not None:
            joblib.dump(self.authenticity_model, os.path.join(model_dir, 'authenticity_model.pkl'))
        
        joblib.dump(self.scaler, os.path.join(model_dir, 'threshold_scaler.pkl'))
        
        # Save feature importance
        with open(os.path.join(model_dir, 'feature_importance.json'), 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"SUCCESS: Models saved to {model_dir}/")
    
    def load_models(self, model_dir="models"):
        """
        📂 Load trained models and scaler
        """
        try:
            self.confidence_model = joblib.load(os.path.join(model_dir, 'confidence_model.pkl'))
            self.authenticity_model = joblib.load(os.path.join(model_dir, 'authenticity_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'threshold_scaler.pkl'))
            
            # Load feature importance
            with open(os.path.join(model_dir, 'feature_importance.json'), 'r') as f:
                self.feature_importance = json.load(f)
            
            print(f"SUCCESS: Models loaded from {model_dir}/")
            return True
        except Exception as e:
            print(f"WARNING: Could not load models: {e}")
            return False

def main():
    """
    Starting Main function for testing the calibration system
    """
    print("Target Advanced Threshold Calibration System")
    print("=" * 50)
    
    # Initialize calibrator
    calibrator = AdvancedThresholdCalibrator()
    
    # Try to load existing models, otherwise train new ones
    if not calibrator.load_models():
        print("? Training new calibration models...")
        calibrator.train_calibration_models()
    
    # Example usage with mock data
    mock_results = {
        'mismatch_score': 0.172,
        'final_score': 0.145,
        'confidence': 0.589,
        'correlation_scores': {
            'pearson': 0.456,
            'cross_correlation': 0.234,
            'coherence': 0.123,
            'dtw_similarity': 0.345
        },
        'motion_stats': {
            'mean': 0.234,
            'std': 0.156,
            'max': 0.789,
            'energy': 0.456
        },
        'audio_stats': {
            'mean': 0.345,
            'std': 0.123,
            'max': 0.678,
            'energy': 0.567
        },
        'confidence_components': {
            'sync_quality': 0.290,
            'pearson_correlation': 0.114,
            'cross_correlation': 0.047,
            'coherence': 0.018,
            'dtw_similarity': 0.017
        }
    }
    
    # Generate advanced report
    report = calibrator.generate_advanced_report(mock_results)
    
    print(f"\nTarget ADVANCED CALIBRATION RESULTS:")
    print(f"   Checking Final Classification: {report['final_classification']}")
    print(f"   ? Confidence Level: {report['confidence_level']}")
    print(f"   Target ML Confidence: {report['scores']['ml_confidence']:.3f}")
    print(f"   ? Authenticity Probability: {report['scores']['authenticity_probability']:.3f}")
    
    print(f"\n?? CALIBRATED THRESHOLDS:")
    for level, threshold in report['calibrated_thresholds'].items():
        print(f"   ? {level.title()}: {threshold:.3f}")
    
    print(f"\n? TOP INFLUENTIAL FEATURES:")
    for feature, importance in report['ml_predictions']['top_features']:
        print(f"   ? {feature}: {importance:.3f}")

if __name__ == "__main__":
    main()
