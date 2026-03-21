"""
dataset.py - PyTorch Dataset and DataLoader Factory

Handles loading the brain tumor MRI dataset with proper train/val/test splits.
Supports the folder-based dataset structure from the Kaggle brain tumor dataset.
"""

import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image

from src.data.preprocessing import get_train_transforms, get_val_transforms

# Class names matching dataset folder structure
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Human-readable display names
DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary'
}

# Number of classes
NUM_CLASSES = 4


class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for brain tumor MRI images.
    
    Wraps torchvision's ImageFolder with additional metadata capabilities.
    Provides image path tracking for Grad-CAM visualization.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to dataset directory containing class subfolders
            transform: torchvision transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Use ImageFolder for automatic label assignment from folder names
        self.dataset = ImageFolder(root_dir, transform=None)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples  # List of (path, class_index) tuples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image_tensor, label, image_path)
        """
        img_path, label = self.samples[idx]
        
        # Load image and convert to RGB (some MRI images may be grayscale)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path
    
    def get_class_distribution(self):
        """
        Returns a dictionary with class names and their sample counts.
        """
        distribution = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def create_data_loaders(data_dir, batch_size=32, val_split=0.2, num_workers=0, seed=42):
    """
    Create train, validation, and test DataLoaders.
    
    The training data is split into train/val sets. The test set uses
    the separate Testing directory from the dataset.
    
    Args:
        data_dir (str): Root directory containing 'Training' and 'Testing' folders
        batch_size (int): Batch size for DataLoaders
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducible splits
    
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader,
               'train_dataset': Dataset, 'val_dataset': Dataset, 'test_dataset': Dataset}
    """
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')
    
    # Create full training dataset (no transforms yet — we'll apply per-subset)
    full_train_dataset = BrainTumorDataset(train_dir, transform=None)
    test_dataset = BrainTumorDataset(test_dir, transform=get_val_transforms())
    
    # Stratified train/val split
    total_samples = len(full_train_dataset)
    indices = list(range(total_samples))
    
    # Group indices by class for stratified splitting
    class_indices = {}
    for idx, (_, label) in enumerate(full_train_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Split each class proportionally
    random.seed(seed)
    train_indices = []
    val_indices = []
    
    for label, idx_list in class_indices.items():
        random.shuffle(idx_list)
        split_point = int(len(idx_list) * (1 - val_split))
        train_indices.extend(idx_list[:split_point])
        val_indices.extend(idx_list[split_point:])
    
    # Shuffle final index lists
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    # Create subset datasets with appropriate transforms
    train_dataset = _TransformedSubset(full_train_dataset, train_indices, get_train_transforms())
    val_dataset = _TransformedSubset(full_train_dataset, val_indices, get_val_transforms())
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_names': full_train_dataset.classes,
        'num_classes': len(full_train_dataset.classes)
    }


class _TransformedSubset(Dataset):
    """
    A dataset subset with its own transform pipeline.
    Allows different augmentations for train vs validation splits.
    """
    
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path
