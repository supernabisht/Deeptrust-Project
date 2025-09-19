import os
import sys
import librosa
import numpy as np
import soundfile as sf
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_audio_file(audio_path):
    """Test if the audio file can be loaded and processed."""
    logger.info(f"Testing audio file: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        logger.error("File does not exist!")
        return False
    
    # Check file size
    file_size = os.path.getsize(audio_path)
    logger.info(f"File size: {file_size} bytes")
    if file_size == 0:
        logger.error("File is empty!")
        return False
    
    # Try to load with soundfile first (more reliable for format detection)
    try:
        logger.info("Attempting to load with soundfile...")
        data, sr = sf.read(audio_path)
        logger.info(f"Successfully loaded with soundfile. Sample rate: {sr}, Shape: {data.shape}, Dtype: {data.dtype}")
        
        # Convert to mono if needed
        if len(data.shape) > 1:
            logger.info(f"Converting to mono from {data.shape[1]} channels")
            data = np.mean(data, axis=1)
        
        # Normalize audio
        data = data / np.max(np.abs(data))
        
        # Save as WAV for librosa
        temp_wav = "temp_audio.wav"
        sf.write(temp_wav, data, sr, 'PCM_16')
        
        # Now try with librosa
        logger.info("Attempting to load with librosa...")
        y, sr = librosa.load(temp_wav, sr=None, mono=True)
        logger.info(f"Successfully loaded with librosa. Sample rate: {sr}, Length: {len(y)}")
        
        # Try MFCC extraction
        logger.info("Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        logger.info(f"MFCC shape: {mfccs.shape}")
        
        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_mfcc.py <audio_file>")
        return
        
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return
        
    success = test_audio_file(audio_file)
    if success:
        print("\n✅ Audio file processed successfully!")
    else:
        print("\n❌ Failed to process audio file.")

if __name__ == "__main__":
    main()
