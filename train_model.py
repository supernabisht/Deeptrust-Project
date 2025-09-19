import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engineering import FeatureExtractor
import cv2
import librosa
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepfakeTrainer:
    def __init__(self, real_videos_dir, fake_videos_dir, model_save_path='models/deepfake_model.pkl'):
        self.real_videos_dir = real_videos_dir
        self.fake_videos_dir = fake_videos_dir
        self.model_save_path = model_save_path
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'aspect_ratio', 'blur_score', 'brightness', 'contrast', 'hue_mean',
            'saturation_mean', 'value_mean', 'edge_density', 'face_count',
            'face_brightness', 'face_contrast', 'face_symmetry', 'texture_contrast',
            'texture_dissimilarity', 'texture_homogeneity', 'texture_energy',
            'texture_correlation', 'mfcc_mean_0', 'mfcc_std_0', 'spectral_centroid',
            'spectral_bandwidth', 'spectral_rolloff', 'zcr', 'rms'
        ]

    def extract_features_from_video(self, video_path):
        """Extract features from a single video file"""
        try:
            # Extract audio
            audio_path = f"temp_audio.wav"
            os.system(f"ffmpeg -y -i \"{video_path}\" -vn -acodec pcm_s16le -ar 22050 -ac 1 \"{audio_path}\" 2>nul")
            
            # Extract frames (just a few for efficiency)
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, frame_count-1, min(30, frame_count), dtype=int)
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frame_indices:
                    frames.append(frame)
            cap.release()
            
            # Extract visual features
            visual_features = []
            for frame in frames:
                features = self.feature_extractor.extract_visual_features(frame)
                if features:
                    visual_features.append(features)
            
            # Extract audio features
            audio_features = {}
            if os.path.exists(audio_path):
                audio_features = self.feature_extractor.extract_audio_features(audio_path)
                os.remove(audio_path)
            
            # Combine features
            combined = self._combine_features(audio_features, visual_features)
            return combined
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return None

    def _combine_features(self, audio_features, visual_features_list):
        """Combine features from multiple frames and audio"""
        combined = {}
        
        # Process visual features (average across frames)
        if visual_features_list and len(visual_features_list) > 0:
            # Get all unique feature names
            all_keys = set()
            for frame_features in visual_features_list:
                if frame_features:
                    all_keys.update(frame_features.keys())
            
            # Calculate mean for each feature
            for key in all_keys:
                values = [f[key] for f in visual_features_list if f and key in f]
                if values:
                    combined[key] = float(np.mean(values))
        
        # Add audio features
        if audio_features:
            for key, value in audio_features.items():
                if isinstance(value, (int, float)):
                    combined[key] = float(value)
        
        return combined

    def prepare_training_data(self):
        """Prepare training data from real and fake videos"""
        X, y = [], []
        
        # Process real videos
        logger.info("Processing real videos...")
        real_videos = [os.path.join(self.real_videos_dir, f) for f in os.listdir(self.real_videos_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_path in tqdm(real_videos, desc="Real videos"):
            features = self.extract_features_from_video(video_path)
            if features:
                X.append(features)
                y.append(0)  # 0 for real
        
        # Process fake videos
        logger.info("Processing fake videos...")
        fake_videos = [os.path.join(self.fake_videos_dir, f) for f in os.listdir(self.fake_videos_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            features = self.extract_features_from_video(video_path)
            if features:
                X.append(features)
                y.append(1)  # 1 for fake
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        
        # Ensure all expected columns exist (fill missing with 0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns
        df = df[self.feature_columns]
        
        return df.values, np.array(y)

    def train(self):
        """Train the model"""
        try:
            # Prepare data
            X, y = self.prepare_training_data()
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid training data found")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training model...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def save_model(self):
        """Save the trained model with metadata"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'model_type': 'RandomForestClassifier',
            'version': '1.2.0',
            'input_shape': (len(self.feature_columns),)
        }
        
        joblib.dump(model_data, self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")

if __name__ == "__main__":
    # Update these paths to point to your real and fake video directories
    REAL_VIDEOS_DIR = "data/real"
    FAKE_VIDEOS_DIR = "data/fake"
    MODEL_SAVE_PATH = "models/trained_deepfake_model.pkl"
    
    # Create directories if they don't exist
    os.makedirs(REAL_VIDEOS_DIR, exist_ok=True)
    os.makedirs(FAKE_VIDEOS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Train the model
    trainer = DeepfakeTrainer(REAL_VIDEOS_DIR, FAKE_VIDEOS_DIR, MODEL_SAVE_PATH)
    success = trainer.train()
    
    if success:
        print("\nTraining completed successfully!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print("\nTo use this model for prediction, run:")
        print(f"python run_enhanced_detection.py --model {MODEL_SAVE_PATH} --video your_video.mp4")
    else:
        print("\nTraining failed. Please check the error messages above.")
