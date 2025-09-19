# feature_engineering.py
import os
import numpy as np
import cv2
import librosa
import soundfile as sf
import mediapipe as mp
from scipy import signal, stats
from scipy.stats import kurtosis, skew, entropy
import pandas as pd
import torch
import torchvision.models as models
from torchvision import transforms

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize face detection models
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

# Initialize face mesh for more detailed facial analysis
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load pre-trained model for deep features
class DeepFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, frame):
        """Extract deep features using ResNet50"""
        try:
            if frame is None or frame.size == 0:
                return np.zeros(2048)  # Return zeros if no frame
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
            
            return features.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Error in deep feature extraction: {str(e)}")
            return np.zeros(2048)  # Return zeros on error

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with deep learning models"""
        self.deep_feature_extractor = DeepFeatureExtractor()
        
    def extract_visual_features(self, frame):
        """
        Extract enhanced visual features from a single frame
        Includes face detection, texture analysis, color features, and deep learning features
        """
        try:
            if frame is None or frame.size == 0:
                return {}
            
            features = {}
            
            # Convert to grayscale for some features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Basic frame properties
            features['aspect_ratio'] = float(width / height) if height > 0 else 0
            
            # Image quality metrics
            features['blur_score'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            features['brightness'] = float(np.mean(gray))
            features['contrast'] = float(np.std(gray))
            features['sharpness'] = float(np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3))))
            
            # Color features in different color spaces
            for color_space, conv in [
                ('hsv', cv2.COLOR_BGR2HSV),
                ('lab', cv2.COLOR_BGR2LAB),
                ('ycr_cb', cv2.COLOR_BGR2YCrCb)
            ]:
                try:
                    cvt = cv2.cvtColor(frame, conv)
                    for i, channel in enumerate(cv2.split(cvt)):
                        features[f'{color_space}_ch{i}_mean'] = float(np.mean(channel))
                        features[f'{color_space}_ch{i}_std'] = float(np.std(channel))
                        features[f'{color_space}_ch{i}_skew'] = float(skew(channel.ravel()))
                except Exception as e:
                    print(f"Error in {color_space} conversion: {str(e)}")
            
            # Edge and gradient features
            edges = cv2.Canny(gray, 100, 200)
            features['edge_density'] = float(np.mean(edges) / 255.0)
            
            # Face detection and analysis using MediaPipe
            face_features = self._extract_face_features(frame)
            features.update(face_features)
            
            # Deep learning features
            deep_features = self.deep_feature_extractor.extract_features(frame)
            for i, val in enumerate(deep_features[:50]):  # Use first 50 deep features
                features[f'deep_feature_{i}'] = float(val)
            
            # Additional statistical features
            features['kurtosis'] = float(kurtosis(gray.ravel()))
            features['entropy'] = float(entropy(gray.ravel()))
            
            return features
            
        except Exception as e:
            print(f"Error in extract_visual_features: {str(e)}")
            return {}
    
    def _extract_face_features(self, frame):
        """Extract detailed facial features using MediaPipe"""
        features = {}
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            results = face_detector.process(frame_rgb)
            features['face_count'] = len(results.detections) if results.detections else 0
            
            if features['face_count'] > 0:
                # Get the largest face
                largest_face = max(
                    results.detections,
                    key=lambda det: (det.location_data.relative_bounding_box.width * 
                                   det.location_data.relative_bounding_box.height)
                )
                
                # Face bounding box
                bbox = largest_face.location_data.relative_bounding_box
                features['face_bbox_area'] = float(bbox.width * bbox.height)
                features['face_bbox_aspect_ratio'] = float(bbox.width / bbox.height) if bbox.height > 0 else 0
                
                # Face landmarks using face mesh
                mesh_results = face_mesh.process(frame_rgb)
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]
                    
                    # Calculate eye aspect ratio (simplified)
                    left_eye = np.array([(landmarks.landmark[33].x, landmarks.landmark[33].y)])
                    right_eye = np.array([(landmarks.landmark[263].x, landmarks.landmark[263].y)])
                    eye_dist = np.linalg.norm(left_eye - right_eye)
                    features['eye_aspect_ratio'] = float(eye_dist)
                    
                    # Mouth aspect ratio
                    mouth_outer = np.array([
                        (landmarks.landmark[13].x, landmarks.landmark[13].y),
                        (landmarks.landmark[14].x, landmarks.landmark[14].y)
                    ])
                    mouth_inner = np.array([
                        (landmarks.landmark[0].x, landmarks.landmark[0].y),
                        (landmarks.landmark[17].x, landmarks.landmark[17].y)
                    ])
                    mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[1])
                    mouth_height = np.linalg.norm(mouth_inner[0] - mouth_inner[1])
                    features['mouth_aspect_ratio'] = float(mouth_width / mouth_height) if mouth_height > 0 else 0
            
        except Exception as e:
            print(f"Error in face feature extraction: {str(e)}")
        
        return features
    
    def extract_audio_features(self, audio_path: str, sr: int = 22050) -> dict:
        """
        Extract comprehensive audio features including MFCCs, spectral features,
        and audio quality metrics
        """
        features = {}
        
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return features
                
            # Load audio file with resampling
            y, sr = librosa.load(audio_path, sr=sr, mono=True)
            
            if y.size == 0:
                print("Empty audio file")
                return features
            
            # Basic audio properties
            features['audio_length'] = len(y) / sr
            features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            
            # Silence ratio
            rms = librosa.feature.rms(y=y)[0]
            silence_threshold = np.percentile(rms, 10)  # 10th percentile as silence threshold
            silence_ratio = np.mean(rms < silence_threshold)
            features['silence_ratio'] = float(silence_ratio)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            })
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i, mfcc in enumerate(mfccs):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfcc))
                features[f'mfcc_{i}_std'] = float(np.std(mfcc))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            for i, chroma_val in enumerate(chroma_mean):
                features[f'chroma_{i}'] = float(chroma_val)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_std'] = float(np.std(zcr))
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = float(np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-10))
            
            # Spectral contrast
            S = np.abs(librosa.stft(y))
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            for i, band in enumerate(contrast):
                features[f'spectral_contrast_band_{i}'] = float(np.mean(band))
            
            # Tempogram features
            hop_length = 512
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            features['tempogram_mean'] = float(np.mean(tempogram))
            features['tempogram_std'] = float(np.std(tempogram))
            
            # Audio quality metrics
            snr = 10 * np.log10(np.mean(y**2) / (np.var(y) + 1e-10))
            features['snr_db'] = float(snr)
            
            # Voice activity detection
            vad = self._detect_voice_activity(y, sr)
            features['voice_activity_ratio'] = float(np.mean(vad))
            
            return features
            
        except Exception as e:
            print(f"Error in audio feature extraction: {str(e)}")
            return features
    
    def _detect_voice_activity(self, y, sr, frame_length=2048, hop_length=512, threshold_db=40):
        """Simple voice activity detection based on energy thresholding"""
        try:
            # Compute short-time energy
            energy = np.array([
                np.sum(np.abs(y[i:i+frame_length]**2))
                for i in range(0, len(y)-frame_length, hop_length)
            ])
            
            # Convert to dB
            energy_db = 10 * np.log10(energy + 1e-10)
            
            # Simple threshold-based VAD
            vad = (energy_db > (np.max(energy_db) - threshold_db)).astype(float)
            
            return vad
            
        except Exception as e:
            print(f"Error in VAD: {str(e)}")
            return np.zeros(10)  # Return array of zeros if VAD fails