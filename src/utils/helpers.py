"""
helpers.py - Common Utility Functions

Provides shared helper functions used across the project:
device detection, checkpoint management, and configuration constants.
"""

import os
import torch

# ── Project-wide configuration ──────────────────────────────────────────────

# Base directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset directory
DATASET_DIR = os.path.join(PROJECT_ROOT, 'archive (1)')

# Model save directory
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# Output directory for visualizations
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Uploads directory for backend
UPLOADS_DIR = os.path.join(PROJECT_ROOT, 'uploads')


def get_device():
    """
    Detect and return the best available compute device.
    Prefers CUDA GPU > CPU.
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch (int): Current epoch
        val_acc (float): Validation accuracy at this checkpoint
        filepath (str): Path to save the checkpoint
    """
    ensure_dir(os.path.dirname(filepath))
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(model, filepath, optimizer=None, device=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        filepath (str): Path to the checkpoint file
        optimizer: Optional optimizer to restore state
        device: Device to map checkpoint to
    
    Returns:
        dict: Checkpoint dictionary with epoch and val_acc
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filepath} (epoch={checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', '?'):.4f})")
    
    return checkpoint
