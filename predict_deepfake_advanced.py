import os
import cv2
import torch
import numpy as np
import librosa
from torchvision import transforms
from advanced_training_pipeline import DeepFakeDetector, AdvancedFeatureExtractor
from tqdm import tqdm

class AdvancedDeepFakeDetector:
    def __init__(self, model_path=None, device='cuda'):
        """Initialize the advanced deepfake detector"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Initialize model
        self.model = DeepFakeDetector(num_classes=2, feature_dim=12, use_pretrained=False)
        
        # Load model weights
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: Model weights not found. Using random initialization.")
        
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
        
        # If couldn't read enough frames, pad with last frame
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            
        return frames[:num_frames]
    
    def _extract_audio_features(self, video_path, target_length=1000):
        """Extract MFCC features from video"""
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
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Extract blink features
            blink_features = self.feature_extractor.extract_eye_blink_features(frame_bgr)
            
            # Extract lip movement features
            lip_features = self.feature_extractor.extract_lip_movement_features(frame_bgr)
            
            # Combine features
            frame_features = list(blink_features.values()) + list(lip_features.values())
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
    parser = argparse.ArgumentParser(description='Advanced DeepFake Video Detection')
    parser.add_argument('video_path', type=str, help='Path to the video file or directory')
    parser.add_argument('--model', type=str, default='models/deepfake_detector_final.pth', 
                       help='Path to the trained model')
    parser.add_argument('--batch', action='store_true', 
                       help='Process a directory of videos in batch mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AdvancedDeepFakeDetector(model_path=args.model)
    
    if args.batch:
        # Process all videos in the directory
        if not os.path.isdir(args.video_path):
            print(f"Error: {args.video_path} is not a directory")
            return
        
        video_files = [f for f in os.listdir(args.video_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"No video files found in {args.video_path}")
            return
        
        results = []
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(args.video_path, video_file)
            result = detector.predict(video_path)
            result['video'] = video_file
            results.append(result)
        
        # Print results
        print("\nBATCH PREDICTION RESULTS")
        print("=" * 50)
        for result in results:
            print(f"\nVideo: {result['video']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            print(f"  REAL: {result['probabilities']['REAL']:.4f}")
            print(f"  FAKE: {result['probabilities']['FAKE']:.4f}")
            
    else:
        # Process single video
        if not os.path.isfile(args.video_path):
            print(f"Error: {args.video_path} does not exist")
            return
        
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
