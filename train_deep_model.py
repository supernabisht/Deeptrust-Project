import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import librosa
from tqdm import tqdm
import pickle

from deep_models import get_model
from advanced_features import AdvancedFeatureExtractor

class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, seq_length=30, audio_length=1000):
        """
        Args:
            data_dir (str): Directory with all the data
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied
            seq_length (int): Number of frames per video
            audio_length (int): Length of audio features
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seq_length = seq_length
        self.audio_length = audio_length
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.video_files = self.metadata['video_path'].tolist()
        self.labels = self.metadata['label'].tolist()
        
        # Split data
        indices = np.arange(len(self.video_files))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        else:  # test
            self.indices = test_idx
    
    def _load_metadata(self):
        # This should be replaced with your actual metadata loading logic
        # For now, we'll create a dummy metadata DataFrame
        video_files = []
        labels = []
        
        # Add real videos
        real_dir = os.path.join(self.data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.mp4'):
                    video_files.append(os.path.join(real_dir, f))
                    labels.append(0)  # 0 for real
        
        # Add fake videos
        fake_dir = os.path.join(self.data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.mp4'):
                    video_files.append(os.path.join(fake_dir, f))
                    labels.append(1)  # 1 for fake
        
        return pd.DataFrame({'video_path': video_files, 'label': labels})
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        video_path = self.video_files[self.indices[idx]]
        label = self.labels[self.indices[idx]]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        # Extract audio features
        audio_features = self._extract_audio_features(video_path)
        
        # Extract advanced features
        advanced_features = self._extract_advanced_features(frames)
        
        # Convert to tensors
        frames_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        audio_tensor = torch.FloatTensor(audio_features)
        features_tensor = torch.FloatTensor(advanced_features)
        label_tensor = torch.LongTensor([label])
        
        return {
            'frames': frames_tensor,
            'audio': audio_tensor,
            'features': features_tensor,
            'label': label_tensor
        }
    
    def _load_video_frames(self, video_path, target_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.seq_length:
            ret, frame = cap.read()
            if not ret:
                # If video ends, loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize and normalize
            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        return np.stack(frames[:self.seq_length])
    
    def _extract_audio_features(self, video_path, target_length=1000):
        # Extract audio from video
        temp_audio = 'temp_audio.wav'
        os.system(f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {temp_audio} -y')
        
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
        """Extract advanced features from video frames"""
        features = []
        
        # Extract features from each frame
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Extract blink features
            blink_features = self.feature_extractor.extract_eye_blink_features(frame_uint8)
            
            # Extract facial movement features
            face_features = self.feature_extractor.extract_facial_movement_features(frame_uint8)
            
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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    """Train the model"""
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for batch in tqdm(dataloaders[phase], desc=phase):
                frames = batch['frames'].to(device)
                audio = batch['audio'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Process each frame in the sequence
                    batch_size, seq_len, C, H, W = frames.shape
                    frames = frames.view(batch_size * seq_len, C, H, W)
                    
                    # Get model outputs
                    outputs = model(frames, audio, features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    data_dir = 'data'  # Update this to your data directory
    
    # Create data loaders
    batch_size = 4
    num_workers = 4
    
    # Create datasets
    train_dataset = DeepFakeDataset(data_dir, split='train')
    val_dataset = DeepFakeDataset(data_dir, split='val')
    test_dataset = DeepFakeDataset(data_dir, split='test')
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    # Initialize model
    model = get_model('resnet')
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=25,
        device=device
    )
    
    # Save the model
    torch.save(model.state_dict(), 'deepfake_detector.pth')
    print("Model saved to deepfake_detector.pth")

if __name__ == '__main__':
    main()
