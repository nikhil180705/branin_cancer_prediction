"""
evaluate.py - Model Evaluation Script

Evaluates the trained brain tumor classifier on the test set:
  - Per-class and overall accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  - Detailed classification report
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.classifier import create_model
from src.data.dataset import create_data_loaders, CLASS_NAMES, DISPLAY_NAMES
from src.utils.helpers import (
    get_device, DATASET_DIR, SAVED_MODELS_DIR, OUTPUT_DIR, ensure_dir
)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Generate and save a styled confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    display_names = [DISPLAY_NAMES.get(c, c) for c in class_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=display_names, yticklabels=display_names,
                ax=ax1, linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, color='white', pad=15)
    ax1.set_ylabel('True Label', fontsize=12, color='white')
    ax1.set_xlabel('Predicted Label', fontsize=12, color='white')
    ax1.tick_params(colors='white')
    
    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names,
                ax=ax2, linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, color='white', pad=15)
    ax2.set_ylabel('True Label', fontsize=12, color='white')
    ax2.set_xlabel('Predicted Label', fontsize=12, color='white')
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


@torch.no_grad()
def evaluate():
    """
    Run full evaluation on the test set using the best saved model.
    """
    device = get_device()
    
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFIER — EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading test dataset...")
    data = create_data_loaders(DATASET_DIR, batch_size=32)
    class_names = data['class_names']
    
    # Load model
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    print(f"Loading model from: {checkpoint_path}")
    
    model = create_model(num_classes=len(class_names), pretrained=False, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(val_acc={checkpoint.get('val_acc', 0):.4f})")
    
    # Run predictions on test set
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning predictions on test set...")
    for images, labels, _ in tqdm(data['test'], desc="Evaluating"):
        images = images.to(device)
        
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    display_names_list = [DISPLAY_NAMES.get(c, c) for c in class_names]
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Print results
    print("\n" + "─" * 60)
    print("  CLASSIFICATION REPORT")
    print("─" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=display_names_list,
        digits=4
    ))
    
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Average metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print(f"  Macro    — P: {macro_p:.4f} R: {macro_r:.4f} F1: {macro_f1:.4f}")
    print(f"  Weighted — P: {weighted_p:.4f} R: {weighted_r:.4f} F1: {weighted_f1:.4f}")
    
    # Save results
    eval_dir = ensure_dir(os.path.join(OUTPUT_DIR, 'evaluation'))
    
    results = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'per_class': {}
    }
    
    for i, cls in enumerate(class_names):
        results['per_class'][cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        os.path.join(eval_dir, 'confusion_matrix.png')
    )
    
    print("\n✅ Evaluation complete!")
    return results


if __name__ == '__main__':
    evaluate()
