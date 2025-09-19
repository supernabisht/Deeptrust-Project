import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import librosa
import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import logging
from typing import Tuple, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepfakeEvaluator:
    """
    Enhanced Deepfake Detection Evaluator that includes:
    - MFCC features
    - Lip-sync analysis
    - Audio-visual feature extraction
    - Unusual expression detection
    - Comprehensive model evaluation
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained model and its metadata."""
        try:
            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.scaler = model_data.get('scaler', StandardScaler())
                self.model_metadata = model_data.get('metadata', {})
            else:
                self.model = model_data
                
            logger.info(f"Successfully loaded {self.model.__class__.__name__} model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def extract_audio_features(self, audio_path: str, sr: int = 16000) -> Dict[str, float]:
        """Extract audio features including MFCCs."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sr)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)
            
            # Extract other audio features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Create feature dictionary
            features = {}
            for i in range(20):
                features[f'mfcc_{i+1}_mean'] = mfcc_means[i]
                features[f'mfcc_{i+1}_std'] = mfcc_stds[i]
                
            features.update({
                'chroma_stft_mean': np.mean(chroma),
                'chroma_stft_std': np.std(chroma),
                'spectral_centroid_mean': np.mean(spec_cent),
                'spectral_bandwidth_mean': np.mean(spec_bw),
                'rolloff_mean': np.mean(rolloff),
                'zero_crossing_rate_mean': np.mean(zcr),
                'rms_energy': np.sqrt(np.mean(y**2))
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return {}
    
    def extract_facial_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract facial features and landmarks."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            features = {}
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Calculate eye aspect ratio (EAR) for blink detection
                def eye_aspect_ratio(eye_landmarks):
                    # Vertical eye landmarks
                    A = np.linalg.norm(np.array([eye_landmarks[1].x - eye_landmarks[5].x, 
                                               eye_landmarks[1].y - eye_landmarks[5].y]))
                    B = np.linalg.norm(np.array([eye_landmarks[2].x - eye_landmarks[4].x,
                                               eye_landmarks[2].y - eye_landmarks[4].y]))
                    # Horizontal eye landmarks
                    C = np.linalg.norm(np.array([eye_landmarks[0].x - eye_landmarks[3].x,
                                               eye_landmarks[0].y - eye_landmarks[3].y]))
                    return (A + B) / (2.0 * C)
                
                # Left and right eye landmarks
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                
                left_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in left_eye])
                right_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in right_eye])
                
                # Mouth aspect ratio (MAR)
                mouth_outer = [61, 291, 39, 181, 0, 17, 269, 405]
                mouth_inner = [78, 95, 88, 178, 0, 14, 317, 402]
                
                def mouth_aspect_ratio(mouth_landmarks):
                    # Vertical mouth landmarks
                    A = np.linalg.norm(np.array([mouth_landmarks[2].x - mouth_landmarks[6].x,
                                               mouth_landmarks[2].y - mouth_landmarks[6].y]))
                    # Horizontal mouth landmarks
                    B = np.linalg.norm(np.array([mouth_landmarks[0].x - mouth_landmarks[4].x,
                                               mouth_landmarks[0].y - mouth_landmarks[4].y]))
                    return A / B
                
                mar = mouth_aspect_ratio([face_landmarks.landmark[i] for i in mouth_outer])
                
                # Update features
                features.update({
                    'left_eye_aspect_ratio': left_ear,
                    'right_eye_aspect_ratio': right_ear,
                    'mouth_aspect_ratio': mar,
                    'blink_detected': int((left_ear + right_ear) / 2 < 0.2),  # Threshold for blink
                    'mouth_open': int(mar > 0.7)  # Threshold for mouth open
                })
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return {}
    
    def extract_video_features(self, video_path: str, max_frames: int = 30) -> Dict[str, float]:
        """Extract features from video frames."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            all_features = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Extract facial features
                face_features = self.extract_facial_features(frame)
                if face_features:
                    all_features.append(face_features)
            
            cap.release()
            
            # Aggregate features across frames
            if all_features:
                df = pd.DataFrame(all_features)
                return {
                    'avg_eye_aspect_ratio': df[['left_eye_aspect_ratio', 'right_eye_aspect_ratio']].mean().mean(),
                    'blink_rate': df['blink_detected'].mean(),
                    'mouth_open_ratio': df['mouth_open'].mean(),
                    'face_detected_frames': len(df) / len(frame_indices)
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting video features: {str(e)}")
            return {}
    
    def extract_audio_video_sync(self, video_path: str) -> Dict[str, float]:
        """Extract audio-visual synchronization features."""
        try:
            # Extract audio from video
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                logger.warning("No audio track found in video")
                return {}
                
            # Save audio to temporary file
            temp_audio_path = "temp_audio.wav"
            audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Extract audio features
            audio_features = self.extract_audio_features(temp_audio_path)
            
            # Clean up
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            # Extract video features
            video_features = self.extract_video_features(video_path)
            
            # Combine features
            features = {**audio_features, **video_features}
            return features
            
        except Exception as e:
            logger.error(f"Error in audio-video sync extraction: {str(e)}")
            return {}
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance on test data."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
                
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC and PR curves if probabilities are available
            roc_auc = None
            pr_auc = None
            
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall, precision)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the figure
        plt.savefig('reports/roc_curve.png')
        plt.close()
    
    def generate_report(self, evaluation_results: Dict, output_dir: str = 'reports') -> None:
        """Generate evaluation report with visualizations."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save classification report
            with open(f'{output_dir}/classification_report.txt', 'w') as f:
                f.write("Classification Report\n")
                f.write("===================\n\n")
                f.write(classification_report(
                    evaluation_results['y_true'], 
                    evaluation_results['y_pred']
                ))
                
                if evaluation_results['roc_auc'] is not None:
                    f.write(f"\nAUC-ROC: {evaluation_results['roc_auc']:.4f}")
                if evaluation_results['pr_auc'] is not None:
                    f.write(f"\nAUC-PR: {evaluation_results['pr_auc']:.4f}")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                evaluation_results['confusion_matrix'],
                classes=['Real', 'Fake'],
                title='Confusion Matrix'
            )
            
            # Plot ROC curve if available
            if evaluation_results['roc_auc'] is not None and evaluation_results['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(evaluation_results['y_true'], evaluation_results['y_prob'])
                self.plot_roc_curve(fpr, tpr, evaluation_results['roc_auc'])
            
            logger.info(f"Evaluation report saved to {output_dir}/")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

def load_dataset(csv_path: str, target_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def main():
    """Main function to run the evaluation."""
    try:
        # Initialize evaluator
        evaluator = DeepfakeEvaluator('models/optimized_deepfake_model.joblib')
        
        # Load dataset
        logger.info("Loading dataset...")
        X, y = load_dataset('consolidated_dataset_20250917_091948.csv')
        
        # Split data (if not already split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = evaluator.evaluate_model(X_test, y_test)
        
        # Generate report
        logger.info("Generating evaluation report...")
        evaluator.generate_report(results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
