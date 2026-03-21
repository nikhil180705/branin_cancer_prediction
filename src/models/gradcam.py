"""
gradcam.py - Gradient-weighted Class Activation Mapping

Implements Grad-CAM for visualizing which regions of the MRI image
the model focuses on when making predictions. This provides
explainable AI capabilities for the brain tumor detector.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization" (2017)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from src.data.preprocessing import get_inference_transforms, denormalize, IMAGE_SIZE


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    
    Hooks into the last convolutional layer to capture
    activations and gradients during forward/backward passes.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: Trained BrainTumorClassifier model
            target_layer: The convolutional layer to compute Grad-CAM for.
                         Defaults to the last feature block of EfficientNet.
        """
        self.model = model
        self.model.eval()
        
        # Default to last features block of EfficientNet
        if target_layer is None:
            self.target_layer = model.backbone.features[-1]
        else:
            self.target_layer = target_layer
        
        # Storage for hooks
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class index to generate heatmap for.
                         If None, uses the predicted class.
        
        Returns:
            dict: {
                'heatmap': numpy array (224, 224) normalized [0, 1],
                'predicted_class': int,
                'confidence': float,
                'probabilities': numpy array of class probabilities
            }
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Get prediction
        probabilities = torch.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probabilities[0, target_class].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                           mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return {
            'heatmap': cam,
            'predicted_class': target_class,
            'confidence': confidence,
            'probabilities': probabilities[0].detach().cpu().numpy()
        }
    
    def generate_overlay(self, image_path, target_class=None, alpha=0.5):
        """
        Generate a Grad-CAM heatmap overlay on the original MRI image.
        
        Args:
            image_path: Path to the MRI image
            target_class: Class to visualize (None = predicted class)
            alpha: Blending factor for the overlay
        
        Returns:
            dict: {
                'overlay': numpy array (H, W, 3) - blended image,
                'heatmap': numpy array (H, W) - raw heatmap,
                'heatmap_colored': numpy array (H, W, 3) - colored heatmap,
                'original': numpy array (H, W, 3) - original image,
                'predicted_class': int,
                'confidence': float,
                'probabilities': numpy array
            }
        """
        device = next(self.model.parameters()).device
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        transform = get_inference_transforms()
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate Grad-CAM
        result = self.generate(input_tensor, target_class)
        heatmap = result['heatmap']
        
        # Create colored heatmap (jet colormap)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = np.float32(heatmap_colored) * alpha + np.float32(original) * (1 - alpha)
        overlay = np.uint8(np.clip(overlay, 0, 255))
        
        result.update({
            'overlay': overlay,
            'heatmap_colored': heatmap_colored,
            'original': original,
        })
        
        return result


def compute_tumor_activation_area(heatmap, threshold=0.5):
    """
    Compute the percentage of the image showing tumor activation.
    Used for tumor size estimation.
    
    Args:
        heatmap: Grad-CAM heatmap array (H, W) with values [0, 1]
        threshold: Activation threshold for considering a region as "tumor"
    
    Returns:
        float: Fraction of image area with activation above threshold
    """
    binary_mask = (heatmap > threshold).astype(np.float32)
    activation_area = binary_mask.sum() / (heatmap.shape[0] * heatmap.shape[1])
    return float(activation_area)
