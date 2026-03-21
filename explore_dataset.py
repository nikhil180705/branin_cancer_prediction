"""
explore_dataset.py - Dataset Exploration & Visualization

Generates visualizations for understanding the brain tumor MRI dataset:
1. Class distribution bar chart
2. Sample images from each class
3. Image dimension statistics
4. Pixel intensity distribution
"""

import os
import sys
import random
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import BrainTumorDataset, CLASS_NAMES, DISPLAY_NAMES
from src.utils.helpers import DATASET_DIR, OUTPUT_DIR, ensure_dir


def plot_class_distribution(dataset, output_path):
    """
    Create a bar chart showing the number of images per class.
    """
    distribution = dataset.get_class_distribution()
    
    # Map to display names
    classes = [DISPLAY_NAMES.get(c, c) for c in distribution.keys()]
    counts = list(distribution.values())
    total = sum(counts)
    
    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    # Color palette
    colors = ['#e94560', '#0f3460', '#533483', '#48c9b0']
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f'{count}\n({count/total*100:.1f}%)',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='white'
        )
    
    ax.set_title('Brain Tumor MRI Dataset — Class Distribution',
                 fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Tumor Type', fontsize=13, color='white')
    ax.set_ylabel('Number of Images', fontsize=13, color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved class distribution plot: {output_path}")


def plot_sample_images(dataset, output_path, samples_per_class=4):
    """
    Display a grid of sample MRI images from each class.
    """
    # Group samples by class
    class_samples = {}
    for img_path, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name not in class_samples:
            class_samples[class_name] = []
        class_samples[class_name].append(img_path)
    
    num_classes = len(class_samples)
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(4 * samples_per_class, 4 * num_classes))
    fig.patch.set_facecolor('#1a1a2e')
    
    for row, class_name in enumerate(CLASS_NAMES):
        if class_name not in class_samples:
            continue
        
        # Randomly select sample images
        selected = random.sample(class_samples[class_name],
                                 min(samples_per_class, len(class_samples[class_name])))
        
        for col, img_path in enumerate(selected):
            ax = axes[row, col] if num_classes > 1 else axes[col]
            
            img = Image.open(img_path).convert('RGB')
            ax.imshow(np.array(img), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add class label to first column
            if col == 0:
                ax.set_ylabel(DISPLAY_NAMES.get(class_name, class_name),
                             fontsize=14, fontweight='bold', color='white',
                             rotation=0, labelpad=80, va='center')
            
            # Add image dimensions as title
            ax.set_title(f'{img.size[0]}×{img.size[1]}',
                        fontsize=9, color='#aaa')
    
    fig.suptitle('Sample MRI Images by Class',
                 fontsize=18, fontweight='bold', color='white', y=1.01)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved sample images grid: {output_path}")


def analyze_image_dimensions(dataset, output_path):
    """
    Analyze and visualize the distribution of image dimensions in the dataset.
    """
    widths = []
    heights = []
    
    print("Analyzing image dimensions...")
    for img_path, _ in tqdm(dataset.samples[:500], desc="Scanning images"):
        try:
            img = Image.open(img_path)
            widths.append(img.size[0])
            heights.append(img.size[1])
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Width distribution
    axes[0].hist(widths, bins=30, color='#e94560', edgecolor='white', alpha=0.8)
    axes[0].set_title('Image Width Distribution', fontsize=13, color='white')
    axes[0].set_xlabel('Width (px)', color='white')
    axes[0].set_ylabel('Count', color='white')
    
    # Height distribution
    axes[1].hist(heights, bins=30, color='#533483', edgecolor='white', alpha=0.8)
    axes[1].set_title('Image Height Distribution', fontsize=13, color='white')
    axes[1].set_xlabel('Height (px)', color='white')
    axes[1].set_ylabel('Count', color='white')
    
    fig.suptitle(f'Image Dimensions (sampled {len(widths)} images)\n'
                 f'Width: {min(widths)}–{max(widths)}px (median={np.median(widths):.0f}) | '
                 f'Height: {min(heights)}–{max(heights)}px (median={np.median(heights):.0f})',
                 fontsize=13, color='white', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved dimension analysis: {output_path}")


def print_dataset_summary(train_dataset, test_dataset):
    """
    Print a formatted summary of the dataset statistics.
    """
    train_dist = train_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()
    
    print("\n" + "=" * 65)
    print("  BRAIN TUMOR MRI DATASET — SUMMARY")
    print("=" * 65)
    
    print(f"\n{'Class':<15} {'Training':>10} {'Testing':>10} {'Total':>10}")
    print("-" * 50)
    
    total_train = 0
    total_test = 0
    
    for cls in CLASS_NAMES:
        tr = train_dist.get(cls, 0)
        te = test_dist.get(cls, 0)
        total_train += tr
        total_test += te
        display = DISPLAY_NAMES.get(cls, cls)
        print(f"{display:<15} {tr:>10} {te:>10} {tr+te:>10}")
    
    print("-" * 50)
    print(f"{'TOTAL':<15} {total_train:>10} {total_test:>10} {total_train+total_test:>10}")
    print("=" * 65)


def main():
    """Run all dataset exploration steps."""
    print("Brain Tumor MRI Dataset — Exploration")
    print("=" * 50)
    
    # Create output directory
    explore_dir = ensure_dir(os.path.join(OUTPUT_DIR, 'exploration'))
    
    # Load datasets
    train_dir = os.path.join(DATASET_DIR, 'Training')
    test_dir = os.path.join(DATASET_DIR, 'Testing')
    
    print(f"\nLoading training data from: {train_dir}")
    train_dataset = BrainTumorDataset(train_dir)
    
    print(f"Loading testing data from: {test_dir}")
    test_dataset = BrainTumorDataset(test_dir)
    
    # 1. Print summary
    print_dataset_summary(train_dataset, test_dataset)
    
    # 2. Class distribution
    plot_class_distribution(
        train_dataset,
        os.path.join(explore_dir, 'class_distribution.png')
    )
    
    # 3. Sample images
    plot_sample_images(
        train_dataset,
        os.path.join(explore_dir, 'sample_images.png')
    )
    
    # 4. Dimension analysis  
    analyze_image_dimensions(
        train_dataset,
        os.path.join(explore_dir, 'dimension_analysis.png')
    )
    
    print("\n✅ Dataset exploration complete!")
    print(f"Visualizations saved to: {explore_dir}")


if __name__ == '__main__':
    main()
