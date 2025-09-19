import os
import cv2
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_recall_curve, roc_curve, auc,
                           f1_score, precision_score, recall_score, average_precision_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            VotingClassifier, StackingClassifier, IsolationForest)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

class AdvancedDeepfakeDetector:
    def __init__(self, model_path=None):
        """Initialize the advanced deepfake detector with enhanced features"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.results = {}
        self.training_history = {}
        self.feature_importances_ = None
        self.classes_ = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_enhanced_features(self, video_path, frame_sample_rate=10):
        """
        Extract comprehensive features from video including:
        - Lip sync features
        - MFCC and other audio features
        - Visual artifacts
        - Color inconsistencies
        - Facial expression analysis
        """
        print(f"Extracting enhanced features from {video_path}...")
        
        # 1. Extract audio features using enhanced MFCC extraction
        audio_features = self._extract_audio_features(video_path)
        
        # 2. Extract visual features
        visual_features = self._extract_visual_features(video_path, frame_sample_rate)
        
        # 3. Combine all features
        all_features = {**audio_features, **visual_features}
        
        # Convert to DataFrame for consistency
        features_df = pd.DataFrame([all_features])
        
        return features_df
    
    def _extract_audio_features(self, video_path):
        """Extract enhanced audio features including MFCCs and other audio metrics"""
        try:
            # Extract audio from video
            temp_audio = "temp_audio.wav"
            cmd = f"ffmpeg -i {video_path} -q:a 0 -map a {temp_audio} -y -loglevel error"
            os.system(cmd)
            
            # Load audio file
            y, sr = librosa.load(temp_audio, sr=22050)  # Standard sample rate
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract other audio features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            
            # Calculate statistics
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist(),
                'chroma_mean': np.mean(chroma),
                'chroma_std': np.std(chroma),
                'mel_mean': np.mean(mel),
                'mel_std': np.std(mel),
                'contrast_mean': np.mean(contrast),
                'contrast_std': np.std(contrast),
                'tonnetz_mean': np.mean(tonnetz),
                'tonnetz_std': np.std(tonnetz),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            }
            
            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return {}
    
    def _extract_visual_features(self, video_path, frame_sample_rate):
        """Extract visual features including facial landmarks, color, and artifacts"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_indices = range(0, frame_count, max(1, int(fps/frame_sample_rate)))
            
            # Initialize feature accumulators
            color_inconsistencies = []
            blur_metrics = []
            face_landmark_changes = []
            
            # Initialize face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            prev_landmarks = None
            
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to grayscale for some analyses
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Color inconsistency
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                color_inconsistencies.append(np.std([l, a, b]))
                
                # 2. Blur metric
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_metrics.append(blur)
                
                # 3. Face landmark changes (simplified)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    # In a real implementation, use dlib for more accurate landmarks
                    x, y, w, h = faces[0]
                    landmarks = np.array([x, y, x+w, y+h])
                    
                    if prev_landmarks is not None:
                        change = np.mean(np.abs(landmarks - prev_landmarks))
                        face_landmark_changes.append(change)
                    
                    prev_landmarks = landmarks
            
            cap.release()
            
            # Calculate statistics
            features = {
                'color_inconsistency_mean': np.mean(color_inconsistencies) if color_inconsistencies else 0,
                'color_inconsistency_std': np.std(color_inconsistencies) if color_inconsistencies else 0,
                'blur_mean': np.mean(blur_metrics) if blur_metrics else 0,
                'blur_std': np.std(blur_metrics) if blur_metrics else 0,
                'landmark_change_mean': np.mean(face_landmark_changes) if face_landmark_changes else 0,
                'landmark_change_std': np.std(face_landmark_changes) if face_landmark_changes else 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting visual features: {str(e)}")
            return {}
    
    def train_ensemble_model(self, X, y, cv=5):
        """Train an ensemble model with enhanced features"""
        print("Training ensemble model with enhanced features...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define base models
        models = [
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('cat', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=5, verbose=0, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
        ]
        
        # Create ensemble model
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train model with cross-validation
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Train final model on full dataset
        self.model = ensemble.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        
        # Get feature importances
        self._calculate_feature_importances(X)
        
        return self.model
    
    def _calculate_feature_importances(self, X):
        """Calculate and store feature importances"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # For voting classifier, average importances from all base models
            importances = []
            for name, model in self.model.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    importances.append(np.mean(np.abs(model.coef_), axis=0))
            
            if importances:
                self.feature_importances_ = np.mean(importances, axis=0)
            else:
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]  # Equal importance if can't determine
    
    def predict(self, X, threshold=0.5):
        """Make predictions with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_ensemble_model first.")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_scaled)
            predictions = (probas[:, 1] >= threshold).astype(int)
            return predictions, probas
        else:
            predictions = self.model.predict(X_scaled)
            return predictions, np.zeros((len(predictions), 2))  # Dummy probabilities
    
    def evaluate(self, X, y_true, threshold=0.5):
        """Evaluate model performance with comprehensive metrics"""
        y_pred, y_proba = self.predict(X, threshold)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return self.results
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, title='Confusion Matrix'):
        """Plot confusion matrix with enhanced visualization"""
        if classes is None:
            classes = ['Real', 'Fake']
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importances(self, feature_names=None, top_n=20):
        """Plot feature importances"""
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Train model first.")
            
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
        
        # Sort features by importance
        indices = np.argsort(self.feature_importances_)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.barh(range(top_n), self.feature_importances_[indices][::-1],
                color='b', align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
        plt.ylim([-1, top_n])
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save model and related objects"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'results': self.results
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load model and related objects"""
        import joblib
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importances_ = model_data.get('feature_importances_')
        self.classes_ = model_data.get('classes_')
        self.results = model_data.get('results', {})
