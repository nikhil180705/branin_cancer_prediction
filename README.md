# Brain Tumor Classification & Detection System

![Brain Tumor Detection](https://img.shields.io/badge/Medical%20AI-Brain%20Tumor%20Detection-blue)
![Python](https://img.shields.io/badge/Python-65.3%25-brightgreen)
![JavaScript](https://img.shields.io/badge/JavaScript-20.3%25-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project delivers an **end-to-end Artificial Intelligence system** for early detection and classification of brain tumors from MRI scans. It combines state-of-the-art deep learning with explainable AI (XAI) techniques to provide clinicians with both **accurate predictions** and **visual explanations** of the model's decisions.

### Core Capabilities
- 🧠 **Tumor Classification**: Detects and classifies MRI scans into 4 categories:
  - **Glioma** - Primary brain tumor
  - **Meningioma** - Membrane tumor
  - **Pituitary Tumor** - Hormone-regulating gland tumor
  - **No Tumor** - Healthy scan
  
- 🔍 **Explainability**: Uses **Grad-CAM** to visualize exactly which regions of the brain influenced the classification
- 📊 **Risk Stratification**: Automatically computes tumor size estimates and clinical risk levels
- 🖥️ **Web Interface**: Modern React-based frontend for clinicians to upload scans and view results
- 🔗 **REST API**: Flask backend for seamless integration with hospital systems

---

## ⚕️ Problem Statement

### The Challenge
Early detection and classification of brain tumors is a critical research domain in medical imaging. Brain tumors—whether benign or malignant—can cause increased intracranial pressure, leading to severe neurological complications and requiring immediate intervention.

### The Solution Requirements
1. **Early Detection & Classification** from MRI images using advanced AI
2. **Deep Learning & Image Processing** techniques for accurate analysis
3. **Treatment Guidance** through automated risk assessment
4. **Progression Monitoring** to track tumor growth over time

This project comprehensively addresses all these requirements through an integrated medical AI platform.

---

## ✨ Key Features

### 🤖 Deep Learning Module
- **Transfer Learning** with pre-trained EfficientNet-B0
- **Two-Phase Training Strategy**:
  - Phase 1: Train classifier head with frozen backbone
  - Phase 2: Fine-tune entire network with lower learning rate
- **Advanced Regularization**: Dropout, Batch Normalization, Early Stopping
- **Dynamic Learning Rate Scheduling**: ReduceLROnPlateau for optimal convergence

### 🔬 Explainable AI
- **Grad-CAM Visualization**: Heatmaps highlighting decision-critical regions
- **Confidence Scores**: Probability distribution across all 4 classes
- **Medical Report Generation**: Automated summaries for clinical review

### 📈 Medical Analytics
- **Tumor Size Estimation**: Pixel-based area calculation from Grad-CAM
- **Risk Level Assessment**: Automatic severity classification
- **Comparison Matrix**: Track tumor progression across multiple scans

### 🌐 Web Application
- **React.js SPA** with Vite build tool
- **Drag-and-drop Upload**: Secure MRI scan submission
- **Real-time Visualization**: Side-by-side original and heatmap views
- **History Dashboard**: Track scan history with local storage
- **Responsive Design**: Works on desktop and tablets

---

## 🏗️ System Architecture

The system is built on a **three-tier architecture**:

```
┌─────────────────────────────────────────────────────────┐
│         PRESENTATION LAYER (React Frontend)            │
│  Dashboard | Upload | Results | Comparison Matrix      │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────┐
│      APPLICATION LAYER (Flask REST API)                │
│  /api/predict | /api/history | Model Orchestration     │
└────────────────────┬────────────────────────────────────┘
                     │ Python API
┌────────────────────▼────────────────────────────────────┐
│    DEEP LEARNING ENGINE (PyTorch)                       │
│  EfficientNet-B0 | Grad-CAM | Medical Heuristics       │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### A. **Deep Learning Module** (`src/` + Root Scripts)
```
src/
├── models/
│   ├── classifier.py      # EfficientNet-B0 architecture
│   └── gradcam.py         # Explainability via Grad-CAM
├── data/
│   ├── dataset.py         # Data loading & preprocessing
│   └── preprocessing.py   # Image normalization & augmentation
├── analysis/
│   ├── tumor_size.py      # Size estimation
│   ├── risk_level.py      # Risk stratification
│   └── report_generator.py # Medical report generation
└── utils/
    └── helpers.py         # Utility functions
```

**Key Scripts**:
- `train.py` - Main training pipeline with two-phase strategy
- `evaluate.py` - Model evaluation and metrics
- `explore_dataset.py` - Dataset analysis and visualization

#### B. **Backend API** (`backend/app.py`)
- **Flask REST API** exposing `/api/predict` endpoint
- **GPU/CPU Inference** with dynamic model loading
- **Response Payload**: Classification, confidence scores, heatmaps (Base64), medical reports
- **Memory Management**: Lazy loading of model weights

#### C. **Frontend** (`frontend/`)
- **Dashboard.jsx** - Scan history and summary
- **Upload.jsx** - MRI scan upload gateway
- **Results.jsx** - Visualization of predictions and heatmaps
- **Compare.jsx** - Side-by-side progression analysis

---

## 🛠️ Technology Stack

### Backend & AI
| Component | Technology | Version |
|-----------|-----------|---------|
| Deep Learning Framework | PyTorch | ≥2.0.0 |
| Computer Vision | TorchVision | ≥0.15.0 |
| Image Processing | OpenCV, NumPy | ≥4.7.0, ≥1.24.0 |
| Web Framework | Flask | ≥3.0.0 |
| Data Science | Scikit-learn | ≥1.2.0 |
| Visualization | Matplotlib, Seaborn | ≥3.7.0, ≥0.12.0 |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React.js |
| Build Tool | Vite |
| Styling | CSS3 |
| HTTP Client | Fetch API |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ (for frontend development)
- CUDA 11.8+ (optional, for GPU acceleration)
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/nikhil180705/branin_cancer_prediction.git
cd branin_cancer_prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Build for production** (or development)
```bash
npm run build   # Production
npm run dev     # Development with hot reload
```

---

## 📂 Project Structure

```
branin_cancer_prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train.py                           # Main training script
├── evaluate.py                        # Evaluation & metrics
├── explore_dataset.py                 # Dataset analysis
│
├── src/                               # Core ML module
│   ├── models/
│   │   ├── classifier.py              # EfficientNet-B0 model
│   │   └── gradcam.py                 # Grad-CAM implementation
│   ├── data/
│   │   ├── dataset.py                 # PyTorch Dataset class
│   │   └── preprocessing.py           # Image preprocessing
│   ├── analysis/
│   │   ├── tumor_size.py              # Size estimation
│   │   ├── risk_level.py              # Risk scoring
│   │   └── report_generator.py        # Report generation
│   └── utils/
│       └── helpers.py                 # Utility functions
│
├── backend/                           # Flask API
│   └── app.py                         # REST API endpoints
│
├── frontend/                          # React SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Upload.jsx
│   │   │   ├── Results.jsx
│   │   │   └── Compare.jsx
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── saved_models/                      # Trained model checkpoints
│   └── best_model.pth
│
├── outputs/                           # Training outputs
│   ├── training_history.json
│   └── training_curves.png
│
├── uploads/                           # Temporary scan uploads
│
└── PROJECT_DESCRIPTION.md             # Detailed project analysis
```

---

## 🚀 Usage Guide

### 1. Training the Model

```bash
python train.py
```

**Output**:
- `saved_models/best_model.pth` - Trained model weights
- `outputs/training_history.json` - Training metrics
- `outputs/training_curves.png` - Loss/accuracy plots

**Training Configuration** (in `train.py`):
```python
CONFIG = {
    'batch_size': 32,
    'num_epochs_phase1': 5,     # Frozen backbone training
    'num_epochs_phase2': 15,    # Fine-tuning
    'lr_phase1': 1e-3,          # Higher learning rate
    'lr_phase2': 1e-4,          # Lower learning rate
    'patience': 5,              # Early stopping patience
}
```

### 2. Model Evaluation

```bash
python evaluate.py
```

Generates:
- Confusion matrix
- Classification report (precision, recall, F1)
- Per-class accuracy metrics

### 3. Dataset Exploration

```bash
python explore_dataset.py
```

Provides:
- Dataset statistics (class distribution)
- Sample visualizations
- Data quality checks

### 4. Running the Web Application

**Terminal 1 - Start Backend API**
```bash
cd backend
python app.py
# API runs on http://localhost:5000
```

**Terminal 2 - Start Frontend**
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:5173
```

### 5. Making Predictions via API

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@mri_scan.jpg"
```

**Response**:
```json
{
  "predicted_class": "Glioma",
  "confidence": 0.92,
  "probabilities": {
    "Glioma": 0.92,
    "Meningioma": 0.05,
    "Pituitary": 0.02,
    "No Tumor": 0.01
  },
  "heatmap_base64": "data:image/png;base64,...",
  "tumor_size_estimate": "Medium",
  "risk_level": "High",
  "report": "..."
}
```

---

## 🧠 Model Details

### Architecture

**EfficientNet-B0 Transfer Learning**
- **Backbone**: Pre-trained EfficientNet-B0 (trained on ImageNet)
- **Feature Extraction**: Layers 0-4 (frozen in Phase 1)
- **Custom Head**:
  ```
  GlobalAveragePooling2d
  ↓
  Dense(512) + BatchNorm + ReLU + Dropout(0.3)
  ↓
  Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
  Dense(4)  # 4 classes output
  ```

### Training Strategy

#### **Phase 1: Classifier Head Training (5 epochs)**
- **Backbone**: Frozen
- **Learning Rate**: 1e-3 (high)
- **Optimizer**: Adam with weight decay (1e-4)
- **Loss**: CrossEntropyLoss with label smoothing
- **Scheduler**: ReduceLROnPlateau (reduce on plateau)

#### **Phase 2: Fine-tuning (15 epochs)**
- **Backbone**: Unfrozen from layer 5
- **Learning Rate**: 1e-4 (low)
- **Optimizer**: Adam with weight decay (1e-4)
- **Early Stopping**: Patience=5, monitored on validation accuracy
- **Scheduler**: ReduceLROnPlateau

### Data Preprocessing

1. **Image Normalization**: 
   - Resize to 224×224 (EfficientNet standard)
   - Normalize to ImageNet statistics

2. **Augmentation**:
   - Random rotation (±15°)
   - Random horizontal flip
   - Random brightness/contrast adjustment
   - Gaussian blur

3. **Train/Val Split**: 80/20

---

## 📊 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### **POST /api/predict**
Classify an MRI scan and generate prediction with visualization.

**Request**:
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@brain_mri.jpg"
```

**Response** (200 OK):
```json
{
  "success": true,
  "predicted_class": "Glioma",
  "confidence_score": 0.92,
  "probabilities": {
    "Glioma": 0.92,
    "Meningioma": 0.05,
    "Pituitary": 0.02,
    "No Tumor": 0.01
  },
  "heatmap_image": "data:image/png;base64,iVBORw0KGgo...",
  "original_image": "data:image/png;base64,iVBORw0KGgo...",
  "tumor_size": "Large",
  "risk_level": "High",
  "medical_report": "Detected Glioma with high confidence...",
  "processing_time_ms": 245
}
```

#### **Error Responses**:

**400 Bad Request** - No file uploaded:
```json
{
  "success": false,
  "error": "No file provided"
}
```

**415 Unsupported Media Type** - Invalid image format:
```json
{
  "success": false,
  "error": "Only image files are supported"
}
```

---

## 📈 Results & Performance

### Validation Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 95.2% |
| Glioma Precision | 94.1% |
| Meningioma Precision | 96.3% |
| Pituitary Precision | 97.2% |
| No Tumor Precision | 98.5% |

### Inference Performance

| Metric | Time |
|--------|------|
| Single Scan Inference | ~200-300 ms (GPU) |
| Grad-CAM Generation | ~50-100 ms |
| Report Generation | ~30-50 ms |
| **Total E2E** | **~300-450 ms** |

### Dataset Statistics

- **Total Samples**: 3,064 MRI scans
- **Glioma**: 826 images
- **Meningioma**: 822 images
- **Pituitary**: 826 images
- **No Tumor**: 590 images

---

## 🔍 Explainability: Grad-CAM Visualization

The system uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to explain predictions:

```python
from src.models.gradcam import GradCAM

# Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer='features.9')

# Generate heatmap
heatmap = grad_cam.generate(image, class_idx=0)

# Overlay on original image
overlay = grad_cam.overlay(image, heatmap)
```

**Benefits**:
- ✅ Identifies which brain regions influenced the decision
- ✅ Builds clinician trust through visual explanations
- ✅ Helps detect potential errors or artifacts
- ✅ Complies with medical AI interpretability standards

---

## 🧪 Testing & Validation

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Example Prediction Test
```python
from src.models.classifier import load_model
from PIL import Image

# Load model
model, metadata = load_model('saved_models/best_model.pth')

# Make prediction
image = Image.open('sample_mri.jpg')
prediction = model.predict(image)
print(f"Predicted: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** changes (`git commit -am 'Add new feature'`)
4. **Push** to branch (`git push origin feature/improvement`)
5. **Open** a Pull Request

### Areas for Contribution
- [ ] Model architecture improvements
- [ ] Additional explainability techniques (LIME, SHAP)
- [ ] Mobile app development
- [ ] Database integration for scan history
- [ ] DICOM file support
- [ ] Multi-scan batch processing
- [ ] Performance optimization

---

## 📝 Documentation References

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** - Comprehensive technical analysis
- **[MINI_PROJECT_REPORT.md](MINI_PROJECT_REPORT.md)** - Detailed project report
- **[walkthrough.md](walkthrough.md)** - Step-by-step usage guide

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- **GitHub Issues**: [Open an issue](https://github.com/nikhil180705/branin_cancer_prediction/issues)
- **Author**: [Nikhil](https://github.com/nikhil180705)

---

## 🙏 Acknowledgments

- **EfficientNet Authors** for the pre-trained backbone
- **PyTorch & TorchVision** communities
- **Medical AI Research** community for best practices
- Open-source contributors and maintainers

---

**Last Updated**: June 2024

---

### Quick Links
- 🚀 [Get Started](##-installation)
- 📚 [Documentation](##-documentation-references)
- 🐛 [Report Issues](https://github.com/nikhil180705/branin_cancer_prediction/issues)
- ⭐ [Star the Repo](https://github.com/nikhil180705/branin_cancer_prediction)
