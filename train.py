"""
train.py - Model Training Script

Trains the EfficientNet-B0 brain tumor classifier with:
  - Two-phase training (frozen backbone → unfrozen fine-tuning)
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping to prevent overfitting
  - Best model checkpointing
  - Training history logging with plots
"""

import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.classifier import create_model
from src.data.dataset import create_data_loaders
from src.utils.helpers import (
    get_device, DATASET_DIR, SAVED_MODELS_DIR, OUTPUT_DIR, ensure_dir
)

# ── Training Configuration ──────────────────────────────────────────────

CONFIG = {
    'batch_size': 32,
    'num_epochs_phase1': 5,     # Phase 1: frozen backbone
    'num_epochs_phase2': 15,    # Phase 2: fine-tuning
    'lr_phase1': 1e-3,          # Higher LR for classifier head only
    'lr_phase2': 1e-4,          # Lower LR for fine-tuning
    'weight_decay': 1e-4,
    'patience': 5,              # Early stopping patience
    'unfreeze_from': 5,         # Unfreeze backbone from this layer
    'dropout_rate': 0.3,
    'val_split': 0.2,
}


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    """
    Train the model for one epoch.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device, desc="Validation"):
    """
    Validate the model.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_history(history, output_path):
    """
    Plot training and validation loss/accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'o-', color='#e94560', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 's-', color='#48c9b0', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves', fontsize=14, color='white')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    
    # Add phase boundary line
    phase1_epochs = CONFIG['num_epochs_phase1']
    if phase1_epochs < len(history['train_loss']):
        ax1.axvline(x=phase1_epochs + 0.5, color='#f0e68c', linestyle='--', alpha=0.5, label='Fine-tune start')
        ax2.axvline(x=phase1_epochs + 0.5, color='#f0e68c', linestyle='--', alpha=0.5)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'o-', color='#e94560', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 's-', color='#48c9b0', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curves', fontsize=14, color='white')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy', color='white')
    ax2.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Training curves saved: {output_path}")


def train():
    """
    Main training function with two-phase transfer learning strategy.
    
    Phase 1: Train classifier head only (backbone frozen)
    Phase 2: Fine-tune entire network with lower learning rate
    """
    device = get_device()
    
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFIER — TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    data = create_data_loaders(
        DATASET_DIR,
        batch_size=CONFIG['batch_size'],
        val_split=CONFIG['val_split']
    )
    
    print(f"  Train: {len(data['train'].dataset)} samples ({len(data['train'])} batches)")
    print(f"  Val:   {len(data['val'].dataset)} samples ({len(data['val'])} batches)")
    print(f"  Classes: {data['class_names']}")
    
    # ── Model ─────────────────────────────────────────────────────────
    print("\nCreating model...")
    model = create_model(
        num_classes=data['num_classes'],
        dropout_rate=CONFIG['dropout_rate'],
        device=device
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    ensure_dir(SAVED_MODELS_DIR)
    
    start_time = time.time()
    
    # ── Phase 1: Train Classifier Head ────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 1: Training classifier head (backbone frozen)")
    print("─" * 60)
    
    model.freeze_backbone()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_phase1'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    for epoch in range(1, CONFIG['num_epochs_phase1'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, data['train'], criterion, optimizer, device,
            desc=f"Phase1 Epoch {epoch}/{CONFIG['num_epochs_phase1']}"
        )
        val_loss, val_acc = validate(model, data['val'], criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG,
                'class_names': data['class_names']
            }, checkpoint_path)
            print(f"  ★ New best model saved (val_acc={val_acc:.4f})")
    
    # ── Phase 2: Fine-tune Entire Network ─────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 2: Fine-tuning entire network")
    print("─" * 60)
    
    model.unfreeze_backbone(from_layer=CONFIG['unfreeze_from'])
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['lr_phase2'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    patience_counter = 0
    
    for epoch in range(1, CONFIG['num_epochs_phase2'] + 1):
        global_epoch = CONFIG['num_epochs_phase1'] + epoch
        
        train_loss, train_acc = train_one_epoch(
            model, data['train'], criterion, optimizer, device,
            desc=f"Phase2 Epoch {epoch}/{CONFIG['num_epochs_phase2']}"
        )
        val_loss, val_acc = validate(model, data['val'], criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch:2d} (global {global_epoch}) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG,
                'class_names': data['class_names']
            }, checkpoint_path)
            print(f"  ★ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n  Early stopping triggered after {patience_counter} epochs without improvement")
                break
    
    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Model saved: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    ensure_dir(OUTPUT_DIR)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plot_training_history(history, plot_path)
    
    return model, history


if __name__ == '__main__':
    train()
