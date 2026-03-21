# NeuroScan AI — Brain Tumor Detection System Walkthrough

## Summary

Built a complete end-to-end AI-powered brain tumor detection system from MRI images. The system classifies tumors (Glioma, Meningioma, Pituitary, No Tumor), generates Grad-CAM heatmaps, estimates tumor size, predicts risk level, and tracks progression across scans.

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.78%** |
| **Validation Accuracy** | **99.04%** |
| Macro Precision | 98.77% |
| Macro Recall | 98.67% |
| Macro F1-Score | 98.71% |
| Training Time | 28.2 minutes (RTX 2050) |

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 100.00% | 97.67% | 98.82% |
| Meningioma | 96.47% | 98.37% | 97.41% |
| No Tumor | 99.26% | 100.00% | 99.63% |
| Pituitary | 99.33% | 98.67% | 99.00% |

### Training Curves
![Training curves showing loss and accuracy over 18 epochs](d:\PROJECTS\branin_cancer_prediction\outputs\training_curves.png)

### Confusion Matrix
![Confusion matrix showing 16 misclassifications out of 1311 test images](d:\PROJECTS\branin_cancer_prediction\outputs\evaluation\confusion_matrix.png)

## Dataset Exploration

### Class Distribution
![Class distribution showing balanced dataset across 4 classes](d:\PROJECTS\branin_cancer_prediction\outputs\exploration\class_distribution.png)

### Sample MRI Images
![Grid of sample MRI images from each tumor class](d:\PROJECTS\branin_cancer_prediction\outputs\exploration\sample_images.png)

## Frontend UI

````carousel
![Analyze tab — upload zone with drag-and-drop support](C:\Users\NIKHIL.R\.gemini\antigravity\brain\1fbcea35-70d7-4666-abcb-6a52e70aafe3\initial_ui_load_1773981959770.png)
<!-- slide -->
![Compare tab — multi-scan progression tracking](C:\Users\NIKHIL.R\.gemini\antigravity\brain\1fbcea35-70d7-4666-abcb-6a52e70aafe3\compare_ui_load_1773981979331.png)
````

## API Testing

```
# Glioma test image
Prediction: Glioma | Confidence: 84.21% | Size: Large | Risk: High ✅

# No Tumor test image  
Prediction: No Tumor | Confidence: 90.85% | Size: None | Risk: Low ✅

# Health check
Status: ok | Model loaded: True | GPU: True | Device: cuda ✅
```

## Project Structure

```
branin_cancer_prediction/
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset + DataLoaders
│   │   └── preprocessing.py    # Augmentation & transforms
│   ├── models/
│   │   ├── classifier.py       # EfficientNet-B0 classifier
│   │   └── gradcam.py          # Grad-CAM heatmap generator
│   ├── analysis/
│   │   ├── tumor_size.py       # Size estimation (Small/Medium/Large)
│   │   ├── risk_level.py       # Risk assessment (Low/Medium/High)
│   │   └── progress.py         # Multi-scan comparison
│   └── utils/
│       └── helpers.py          # Device detection, checkpoints
├── backend/
│   └── app.py                  # Flask API server
├── frontend/
│   ├── index.html              # Main UI
│   ├── style.css               # Premium dark theme
│   └── script.js               # Frontend logic
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── explore_dataset.py          # Dataset exploration
├── requirements.txt            # Dependencies
├── saved_models/
│   └── best_model.pth          # Trained model (99.04% val acc)
└── outputs/
    ├── training_curves.png
    ├── training_history.json
    ├── evaluation/
    │   ├── evaluation_results.json
    │   └── confusion_matrix.png
    └── exploration/
        ├── class_distribution.png
        ├── sample_images.png
        └── dimension_analysis.png
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (optional — pretrained model already saved)
python train.py

# Evaluate on test set
python evaluate.py

# Start the web application
python backend/app.py
# Then open http://127.0.0.1:5000
```
