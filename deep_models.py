import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import ViTModel, ViTConfig

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, feature_dim=53, use_pretrained=True):
        super(DeepFakeDetector, self).__init__()
        
        # Visual feature extractor (CNN)
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
        # Visual feature extraction
        visual_features = self.visual_extractor(visual_input)
        
        # Audio feature extraction
        audio_features = self.audio_extractor(audio_input.unsqueeze(1)).squeeze(-1)
        
        # Concatenate all features
        combined = torch.cat([visual_features, audio_features, handcrafted_features], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

class VisionTransformerDetector(nn.Module):
    def __init__(self, num_classes=2, feature_dim=53, img_size=224):
        super(VisionTransformerDetector, self).__init__()
        
        # Initialize Vision Transformer
        config = ViTConfig(
            image_size=img_size,
            patch_size=16,
            num_classes=0,  # No classification head
            num_channels=3
        )
        self.vit = ViTModel(config)
        
        # Audio feature extractor (same as before)
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
        visual_feat_dim = 768  # ViT base feature dimension
        audio_feat_dim = 128
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(visual_feat_dim + audio_feat_dim + feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, visual_input, audio_input, handcrafted_features):
        # Visual feature extraction with ViT
        visual_features = self.vit(visual_input).last_hidden_state[:, 0, :]  # [CLS] token
        
        # Audio feature extraction
        audio_features = self.audio_extractor(audio_input.unsqueeze(1)).squeeze(-1)
        
        # Concatenate all features
        combined = torch.cat([visual_features, audio_features, handcrafted_features], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

def get_model(model_name='resnet', **kwargs):
    """Factory function to get model by name"""
    models = {
        'resnet': DeepFakeDetector,
        'vit': VisionTransformerDetector
    }
    return models[model_name.lower()](**kwargs)

# Example usage
if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a test input
    batch_size = 2
    visual_input = torch.randn(batch_size, 3, 224, 224).to(device)
    audio_input = torch.randn(batch_size, 1000).to(device)  # 1000 audio features
    handcrafted_features = torch.randn(batch_size, 53).to(device)  # 53 handcrafted features
    
    # Test ResNet model
    model = DeepFakeDetector().to(device)
    output = model(visual_input, audio_input, handcrafted_features)
    print(f"ResNet model output shape: {output.shape}")
    
    # Test ViT model
    model = VisionTransformerDetector().to(device)
    output = model(visual_input, audio_input, handcrafted_features)
    print(f"ViT model output shape: {output.shape}")
