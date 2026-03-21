"""
preprocessing.py - Data Augmentation & Transform Pipelines

Defines the image transforms for training, validation, and inference.
Uses ImageNet normalization stats for transfer learning compatibility.
"""

from torchvision import transforms

# ImageNet normalization statistics (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for EfficientNet-B0
IMAGE_SIZE = 224


def get_train_transforms():
    """
    Training transforms with data augmentation.
    Includes random flips, rotations, color jitter, and affine transforms
    to improve model generalization and reduce overfitting.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Validation/Testing transforms - no augmentation.
    Only resize and normalize for consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms():
    """
    Inference transforms for single image prediction.
    Same as validation but can be used for individual images.
    """
    return get_val_transforms()


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization purposes.
    Converts a normalized tensor back to displayable [0, 1] range.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
    
    Returns:
        Denormalized tensor with values clipped to [0, 1]
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    # Move mean/std to same device as tensor
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
