import os
import torch
import cv2
import numpy as np
import librosa
from torchvision import transforms
from deep_models import get_model
from advanced_features import AdvancedFeatureExtractor

class DeepFakeDetector:
    def __init__(self, model_path=None, model_type='resnet', device='cuda'):
        """Initialize the deepfake detector"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Initialize model
        self.model = get_model(model_type=model_type)
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, video_path, num_frames=30):
        """
        Predict if a video is real or fake
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to process
            
        Returns:
            dict: Prediction results with confidence scores
        """
        try:
            # Extract features
            frames = self._extract_frames(video_path, num_frames)
            audio_features = self._extract_audio_features(video_path)
            advanced_features = self._extract_advanced_features(frames)
            
            # Convert to tensors
            frames_tensor = torch.stack([self.transform(frame) for frame in frames])
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # Add batch dim
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
            features_tensor = torch.FloatTensor(advanced_features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(frames_tensor, audio_tensor, features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # Prepare result
            result = {
                'prediction': 'FAKE' if prediction.item() == 1 else 'REAL',
                'confidence': confidence.item(),
                'probabilities': {
                    'REAL': probabilities[0][0].item(),
                    'FAKE': probabilities[0][1].item()
                },
                'features_used': {
                    'visual': frames_tensor.shape,
                    'audio': audio_tensor.shape,
                    'advanced': features_tensor.shape
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_frames(self, video_path, num_frames):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _extract_audio_features(self, video_path, target_length=1000):
        """Extract audio features from video"""
        temp_audio = 'temp_audio.wav'
        os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio}" -y')
        
        try:
            # Load audio file
            y, sr = librosa.load(temp_audio, sr=16000)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = mfcc.T  # [time, n_mfcc]
            
            # Pad or truncate to target length
            if mfcc.shape[0] > target_length:
                mfcc = mfcc[:target_length]
            else:
                pad_width = ((0, target_length - mfcc.shape[0]), (0, 0))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            
            # Flatten and normalize
            mfcc = mfcc.flatten()
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            
            return mfcc
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return np.zeros(target_length * 13)  # 13 MFCC coefficients
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def _extract_advanced_features(self, frames):
        """Extract advanced features from frames"""
        features = []
        
        for frame in frames:
            # Extract blink features
            blink_features = self.feature_extractor.extract_eye_blink_features(frame)
            
            # Extract facial movement features
            face_features = self.feature_extractor.extract_facial_movement_features(frame)
            
            # Combine features
            frame_features = list(blink_features.values()) + list(face_features.values())
            features.append(frame_features)
        
        # Calculate temporal features
        features = np.array(features)
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        
        # Combine all features
        combined_features = np.concatenate([mean_features, std_features])
        
        return combined_features

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DeepFake Video Detection')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--model', type=str, default='deepfake_detector.pth', 
                       help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=['resnet', 'vit'], help='Type of model to use')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DeepFakeDetector(
        model_path=args.model,
        model_type=args.model_type
    )
    
    # Make prediction
    result = detector.predict(args.video_path)
    
    # Print results
    print("\nDEEPFAKE DETECTION RESULTS")
    print("=" * 50)
    print(f"Video: {args.video_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nProbabilities:")
    print(f"  REAL: {result['probabilities']['REAL']:.4f}")
    print(f"  FAKE: {result['probabilities']['FAKE']:.4f}")
    print("\nFeatures used:")
    for feat_type, shape in result['features_used'].items():
        print(f"  {feat_type.capitalize()}: {shape}")

if __name__ == '__main__':
    main()
