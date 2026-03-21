"""
classifier.py - Brain Tumor Classification Model

EfficientNet-B0 based classifier with pretrained ImageNet weights.
Fine-tuned for 4-class brain tumor classification:
  Glioma | Meningioma | No Tumor | Pituitary
"""

import torch
import torch.nn as nn
from torchvision import models


class BrainTumorClassifier(nn.Module):
    """
    EfficientNet-B0 transfer learning model for brain tumor classification.
    
    Architecture:
      - EfficientNet-B0 backbone (pretrained on ImageNet)
      - Custom classifier head with dropout for regularization
      - 4-class output (Glioma, Meningioma, No Tumor, Pituitary)
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.3, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability in classifier head
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(BrainTumorClassifier, self).__init__()
        
        # Load EfficientNet-B0 with pretrained weights
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get the number of features from the backbone's classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Store config
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x):
        """
        Extract feature maps from the last convolutional layer.
        Used by Grad-CAM for generating heatmaps.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Feature maps from the last conv block
        """
        # EfficientNet features extraction (everything before classifier)
        return self.backbone.features(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers (for initial fine-tuning)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen — only classifier head will be trained")
    
    def unfreeze_backbone(self, from_layer=5):
        """
        Unfreeze backbone layers from a specific point.
        
        Args:
            from_layer (int): Unfreeze from this layer index onwards.
                            EfficientNet-B0 has 9 feature blocks (0-8).
                            Default=5 unfreezes the last 4 blocks.
        """
        # First freeze everything
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Then unfreeze from the specified layer
        for i, block in enumerate(self.backbone.features):
            if i >= from_layer:
                for param in block.parameters():
                    param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Backbone unfrozen from layer {from_layer} — "
              f"{trainable:,}/{total:,} parameters trainable ({trainable/total*100:.1f}%)")


def create_model(num_classes=4, dropout_rate=0.3, pretrained=True, device=None):
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes (int): Number of classes
        dropout_rate (float): Dropout rate
        pretrained (bool): Use pretrained weights
        device: Target device (auto-detected if None)
    
    Returns:
        BrainTumorClassifier on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BrainTumorClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: EfficientNet-B0")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model
