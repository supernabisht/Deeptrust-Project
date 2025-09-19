# extract_mfcc.py
import librosa
import numpy as np
import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

def extract_mfcc(audio_path, n_mfcc=20, sr=22050, n_fft=2048, hop_length=512):
    """Extract MFCC features from audio file.
    
    Returns:
        list: List of 53 features (40 MFCC statistics + 13 additional audio features)
    """
    try:
        # Check if file exists and is not empty
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return None
            
        if os.path.getsize(audio_path) == 0:
            logging.error(f"Audio file is empty: {audio_path}")
            return None
            
        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_path, sr=sr, mono=True)
        except Exception as load_error:
            logging.error(f"Error loading audio file {audio_path}: {str(load_error)}")
            return None
            
        # Check if audio is valid
        if y.size == 0:
            logging.error(f"No audio data found in {audio_path}")
            return None
        
        # 1. Extract MFCC features (40 features: 20 means + 20 stds)
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Calculate statistics over the frames
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # 2. Extract additional audio features (13 features)
        # Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Spectral centroid
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Spectral bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Spectral rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Root Mean Square (RMS) energy
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Chroma features (12 semitones)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_means,          # 20 features
            mfcc_stds,           # 20 features
            [zcr],               # 1 feature
            [spectral_centroids],# 1 feature
            [spectral_bandwidth],# 1 feature
            [spectral_rolloff],  # 1 feature
            [rms],               # 1 feature
            chroma[:8]           # 8 features (first 8 chroma)
        ])
        
        # Ensure we have exactly 53 features
        if len(features) < 53:
            logging.warning(f"Only got {len(features)} features, padding with zeros")
            features = np.pad(features, (0, 53 - len(features)), 'constant')
        elif len(features) > 53:
            logging.warning(f"Got {len(features)} features, truncating to 53")
            features = features[:53]
            
        logging.info(f"Extracted {len(features)} audio features")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logging.error("Invalid feature values (NaN or Inf) detected")
            # Replace NaNs and Infs with zeros
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.tolist()
            
    except Exception as e:
        logging.error(f"Unexpected error in extract_mfcc: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

def extract_mfcc_features(audio_path, features=None):
    """Wrapper function to maintain compatibility with the enhanced detection script.
    
    Args:
        audio_path (str): Path to the audio file
        features: Unused parameter, kept for compatibility
        
    Returns:
        list: MFCC features
    """
    return extract_mfcc(audio_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_mfcc.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    features = extract_mfcc_features(audio_file)
    if features is not None:
        print("MFCC Features:")
        print(features)
    else:
        print("Failed to extract MFCC features")
        sys.exit(1)