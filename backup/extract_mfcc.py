import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from scipy import stats

def extract_audio_features(y, sr, n_mfcc=13):
    """
    Extract comprehensive audio features including MFCCs and other relevant features
    """
    features = {}
    
    # Basic audio features
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i}'] = np.mean(mfcc)
        features[f'mfcc_{i}_std'] = np.std(mfcc)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma)
    features['chroma_stft_std'] = np.std(chroma)
    
    # Mel-scaled spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_spectrogram_mean'] = np.mean(mel)
    features['mel_spectrogram_std'] = np.std(mel)
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)
    
    # Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz_mean'] = np.mean(tonnetz)
    
    return features, mfccs

def plot_audio_analysis(y, sr, mfccs, output_path):
    """Create comprehensive audio analysis plots"""
    plt.figure(figsize=(20, 16))
    
    # 1. Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # 2. MFCCs
    plt.subplot(3, 1, 2)
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCCs')
    
    # 3. Mel-scaled spectrogram
    plt.subplot(3, 1, 3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_mfcc(audio_path, output_path=None, n_mfcc=13, visualize=True):
    """
    Enhanced MFCC extraction with comprehensive audio analysis
    """
    try:
        # Load audio file with resampling to standardize sample rate
        y, sr = librosa.load(audio_path, sr=22050)  # Standardize to 22.05kHz
        
        # Extract comprehensive audio features
        features, mfccs = extract_audio_features(y, sr, n_mfcc)
        
        # Save features if output path is provided
        if output_path:
            # Save both raw MFCCs and extracted features
            feature_output_path = output_path.replace('.npy', '_features.npy')
            np.save(output_path, mfccs)
            np.save(feature_output_path, features)
            
            # Save features to CSV for analysis
            df = pd.DataFrame([features])
            df.to_csv(output_path.replace('.npy', '_features.csv'), index=False)
        
        # Visualize if requested
        if visualize:
            output_img = os.path.splitext(audio_path)[0] + '_analysis.png'
            plot_audio_analysis(y, sr, mfccs, output_img)
            print(f"Audio analysis visualization saved to {output_img}")
        
        return mfccs, features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    audio_file = "my_voice.wav"
    print(f"Processing {audio_file}...")
    mfccs, features = extract_mfcc(audio_file, "real_mfcc.npy", visualize=True)
    
    if mfccs is not None and features is not None:
        print("\nExtracted audio features:")
        for key, value in features.items():
            print(f"{key}: {value:.4f}")