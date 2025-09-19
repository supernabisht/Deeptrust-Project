import cv2
import numpy as np
import librosa
import os
from scipy import stats
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

def extract_comprehensive_features(audio_path, video_path):
    """Extract comprehensive features from audio and video"""
    features = {}
    
    try:
        # Extract audio features
        if os.path.exists(audio_path):
            audio_features = extract_audio_features(audio_path)
            features.update(audio_features)
        
        # Extract video features
        if os.path.exists(video_path):
            video_features = extract_video_features(video_path)
            features.update(video_features)
            
            # Extract lip sync features if audio is available
            if os.path.exists(audio_path):
                lip_sync_features = extract_lip_sync_features(audio_path, video_path)
                features.update(lip_sync_features)
    
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # Return empty features if extraction fails
        return {}
    
    return features

def extract_audio_features(audio_path):
    """Extract features from audio file"""
    features = {}
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract various audio features
        features['audio_rms'] = np.mean(librosa.feature.rms(y=y))
        features['audio_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['audio_spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['audio_spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['audio_zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'audio_mfcc_{i}'] = np.mean(mfccs[i])
        
        # Additional features
        features['audio_skew'] = stats.skew(y)
        features['audio_kurtosis'] = stats.kurtosis(y)
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
    
    return features

def extract_video_features(video_path):
    """Extract features from video file"""
    features = {}
    
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count == 0:
            return features
        
        features['video_frame_count'] = frame_count
        features['video_fps'] = fps
        features['video_duration'] = frame_count / fps if fps > 0 else 0
        
        # Sample frames for analysis
        sample_indices = np.linspace(0, frame_count-1, min(20, frame_count), dtype=int)
        frames = []
        grads = []
        
        for i in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                
                # Calculate gradient
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                grads.append(np.mean(grad_magnitude))
        
        cap.release()
        
        if not frames:
            return features
        
        # Calculate frame consistency metrics
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            frame_diffs.append(np.mean(diff))
        
        features['frame_consistency_mean'] = np.mean(frame_diffs) if frame_diffs else 0
        features['frame_consistency_std'] = np.std(frame_diffs) if frame_diffs else 0
        features['gradient_mean'] = np.mean(grads) if grads else 0
        features['gradient_std'] = np.std(grads) if grads else 0
        
    except Exception as e:
        print(f"Error extracting video features: {e}")
    
    return features

def extract_lip_sync_features(audio_path, video_path):
    """Extract lip sync related features"""
    features = {}
    
    try:
        # This is a simplified version - you would need a more sophisticated approach
        # for actual lip sync analysis
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate audio energy
        audio_energy = np.mean(librosa.feature.rms(y=y))
        
        # Sample video frames
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, frame_count-1, min(30, frame_count), dtype=int)
        
        mouth_areas = []
        for i in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Simple mouth region detection (in practice, use face landmarks)
                height, width = frame.shape[:2]
                mouth_roi = frame[int(height*0.6):int(height*0.9), 
                                 int(width*0.25):int(width*0.75)]
                
                if mouth_roi.size > 0:
                    gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                    mouth_areas.append(np.mean(gray_mouth))
        
        cap.release()
        
        if mouth_areas:
            features['lip_mean'] = np.mean(mouth_areas)
            features['lip_std'] = np.std(mouth_areas)
            features['lip_skew'] = stats.skew(mouth_areas)
            features['lip_kurtosis'] = stats.kurtosis(mouth_areas)
            
            # Simple correlation with audio energy (placeholder)
            features['audio_lip_correlation'] = np.corrcoef([audio_energy] * len(mouth_areas), 
                                                           mouth_areas)[0, 1] if len(mouth_areas) > 1 else 0
        
    except Exception as e:
        print(f"Error extracting lip sync features: {e}")
    
    return features