import os
import cv2
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import time
from datetime import datetime
from collections import defaultdict
import mediapipe as mp
import dlib
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class AdvancedFeatureExtractor:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        try:
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except:
            print("Warning: Could not load shape predictor. Lip-sync features will be limited.")
            self.landmark_predictor = None
        
        # Eye aspect ratio constants
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        
    def extract_eye_blink_features(self, frame):
        """Extract eye blink rate and duration"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return {"blink_rate": 0, "avg_blink_duration": 0, "eye_aspect_ratio": 0}
        
        if self.landmark_predictor:
            landmarks = self.landmark_predictor(gray, faces[0])
            
            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            # Calculate eye aspect ratio
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Simple blink detection
            blink = 1 if ear < self.EYE_AR_THRESH else 0
            
            return {
                "blink_rate": 1 if blink else 0,
                "avg_blink_duration": 1 if blink else 0,
                "eye_aspect_ratio": ear
            }
        else:
            return {"blink_rate": 0, "avg_blink_duration": 0, "eye_aspect_ratio": 0}
    
    @staticmethod
    def eye_aspect_ratio(eye):
        """Calculate eye aspect ratio"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def extract_lip_movement_features(self, frame):
        """Extract lip movement features using MediaPipe"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return {"lip_movement": 0, "mouth_openness": 0, "lip_symmetry": 1.0}
            
            # Get lip landmarks (simplified lip contour)
            lip_landmarks = []
            for idx in [61, 291, 39, 181, 0, 17, 269, 405]:
                landmark = results.multi_face_landmarks[0].landmark[idx]
                lip_landmarks.append((landmark.x, landmark.y))
            
            # Calculate mouth openness (vertical distance)
            upper_lip = lip_landmarks[0][1]  # Top lip
            lower_lip = lip_landmarks[4][1]   # Bottom lip
            mouth_openness = abs(upper_lip - lower_lip)
            
            # Calculate lip symmetry
            left = lip_landmarks[0][0]  # Left corner
            right = lip_landmarks[4][0]  # Right corner
            center = (left + right) / 2
            
            # Calculate horizontal symmetry
            left_dist = abs(lip_landmarks[1][0] - center)
            right_dist = abs(lip_landmarks[3][0] - center)
            lip_symmetry = 1 - (abs(left_dist - right_dist) / max(left_dist, right_dist + 1e-6))
            
            return {
                "lip_movement": 0,  # Would need temporal analysis
                "mouth_openness": mouth_openness,
                "lip_symmetry": lip_symmetry
            }
            
        except Exception as e:
            print(f"Error in lip movement extraction: {e}")
            return {"lip_movement": 0, "mouth_openness": 0, "lip_symmetry": 1.0}

class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, seq_length=30, audio_length=1000):
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
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        audio_tensor = torch.FloatTensor(audio_features)
        features_tensor = torch.FloatTensor(advanced_features)
        label_tensor = torch.LongTensor([label])
        
        return {
            'frames': frames_tensor,
            'audio': audio_tensor,
            'features': features_tensor,
            'label': label_tensor,
            'video_path': video_path
        }
    
    def _load_video_frames(self, video_path, target_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.seq_length, dtype=int)
        
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
        
        cap.release()
        
        # If couldn't read enough frames, pad with last frame
        while len(frames) < self.seq_length:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
            
        return frames[:self.seq_length]
    
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

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, feature_dim=12, use_pretrained=True):
        super(DeepFakeDetector, self).__init__()
        
        # Visual feature extractor (ResNet50)
        self.visual_extractor = models.resnet50(pretrained=use_pretrained)
        self.visual_extractor.fc = nn.Identity()  # Remove final classification layer
        
        # Audio feature extractor (1D CNN)
        self.audio_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature dimensions
        visual_feat_dim = 2048  # ResNet50 feature dimension
        audio_feat_dim = 128
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(visual_feat_dim + audio_feat_dim + feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, visual_input, audio_input, handcrafted_features):
        batch_size, seq_len, C, H, W = visual_input.shape
        
        # Process each frame in the sequence
        visual_input = visual_input.view(batch_size * seq_len, C, H, W)
        visual_features = self.visual_extractor(visual_input)
        visual_features = visual_features.view(batch_size, seq_len, -1).mean(dim=1)  # Average over sequence
        
        # Process audio features
        audio_features = self.audio_extractor(audio_input.unsqueeze(1)).squeeze(-1)
        
        # Concatenate all features
        combined = torch.cat([visual_features, audio_features, handcrafted_features], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    """Train the model"""
    model = model.to(device)
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
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
                    outputs = model(frames, audio, features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * frames.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'best_model.pth')
                print(f'Model saved with validation accuracy: {best_acc:.4f}')
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate the model on test set"""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    video_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].squeeze()
            
            outputs = model(frames, audio, features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            video_paths.extend(batch['video_path'])
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE'], output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Save results
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'video_paths': video_paths,
        'report': report,
        'confusion_matrix': conf_matrix
    }
    
    return results

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], 
                yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Creating datasets...")
    data_dir = 'data'  # Update this to your data directory
    
    train_dataset = DeepFakeDataset(data_dir, split='train', transform=transform)
    val_dataset = DeepFakeDataset(data_dir, split='val', transform=transform)
    test_dataset = DeepFakeDataset(data_dir, split='test', transform=transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 4
    num_workers = 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Initialize model
    print("Initializing model...")
    model = DeepFakeDetector(num_classes=2, feature_dim=12)  # 12 advanced features
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=25,
        device=device
    )
    
    # Save the final model
    torch.save(model.state_dict(), 'models/deepfake_detector_final.pth')
    print("Final model saved to models/deepfake_detector_final.pth")
    
    # Plot training history
    plot_training_history(history, 'reports/training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, dataloaders['test'], device)
    
    # Save evaluation results
    with open('reports/classification_report.txt', 'w') as f:
        f.write(classification_report(results['labels'], results['predictions'], 
                                    target_names=['REAL', 'FAKE']))
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], 'reports/confusion_matrix.png')
    
    print("\nTest set results:")
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=['REAL', 'FAKE']))
    
    # Save full results
    results_df = pd.DataFrame({
        'video_path': results['video_paths'],
        'true_label': ['REAL' if x == 0 else 'FAKE' for x in results['labels']],
        'predicted_label': ['REAL' if x == 0 else 'FAKE' for x in results['predictions']],
        'correct': [1 if x == y else 0 for x, y in zip(results['labels'], results['predictions'])]
    })
    
    results_df.to_csv('reports/detailed_results.csv', index=False)
    print("\nDetailed results saved to reports/detailed_results.csv")

if __name__ == '__main__':
    main()
