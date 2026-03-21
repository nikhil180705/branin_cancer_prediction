"""
app.py - Flask Backend Application

Main backend server for the Brain Tumor Detection System.
Serves the frontend and provides API endpoints for:
  - /api/predict         — Tumor classification with Grad-CAM and risk analysis
  - /api/compare         — Compare multiple MRI scans for progress tracking
  - /api/generate-report — AI-powered medical report generation
"""

import os
import sys
import json
import uuid
import base64
import io
import traceback

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import create_model
from src.models.gradcam import GradCAM, compute_tumor_activation_area
from src.data.dataset import CLASS_NAMES, DISPLAY_NAMES
from src.data.preprocessing import get_inference_transforms, IMAGE_SIZE
from src.analysis.tumor_size import estimate_tumor_size
from src.analysis.risk_level import predict_risk_level
from src.analysis.progress import compare_scans
from src.analysis.report_generator import generate_report
from src.utils.helpers import SAVED_MODELS_DIR, UPLOADS_DIR, ensure_dir

# ── App Configuration ────────────────────────────────────────────────────

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Ensure upload directory exists
ensure_dir(UPLOADS_DIR)

# ── Model Loading ────────────────────────────────────────────────────────

_model = None
_gradcam = None
_device = None


def get_model():
    """Lazy-load the trained model and Grad-CAM instance."""
    global _model, _gradcam, _device
    
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {_device}...")
        
        checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained model found at {checkpoint_path}. "
                "Run train.py first."
            )
        
        _model = create_model(num_classes=4, pretrained=False, device=_device)
        checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        
        _gradcam = GradCAM(_model)
        
        print(f"Model loaded successfully (val_acc={checkpoint.get('val_acc', 0):.4f})")
    
    return _model, _gradcam, _device


def analyze_image(image_path):
    """
    Run full analysis pipeline on a single MRI image.
    
    Returns dict with prediction, heatmap, size, risk, and confidence.
    """
    model, gradcam, device = get_model()
    
    # Generate Grad-CAM overlay
    result = gradcam.generate_overlay(image_path)
    
    # Get prediction details
    pred_class_idx = result['predicted_class']
    pred_class = CLASS_NAMES[pred_class_idx]
    display_name = DISPLAY_NAMES.get(pred_class, pred_class)
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Tumor size estimation
    activation_area = compute_tumor_activation_area(result['heatmap'])
    size_info = estimate_tumor_size(activation_area, pred_class)
    
    # Risk level prediction
    risk_info = predict_risk_level(pred_class, size_info['size_category'], confidence)
    
    # Encode images to base64 for frontend
    overlay_b64 = _numpy_to_base64(result['overlay'])
    heatmap_b64 = _numpy_to_base64(result['heatmap_colored'])
    original_b64 = _numpy_to_base64(result['original'])
    
    # Is tumor present?
    has_tumor = pred_class != 'notumor'
    
    return {
        'prediction': {
            'class': pred_class,
            'display_name': display_name,
            'confidence': round(confidence * 100, 2),
            'has_tumor': has_tumor,
            'probabilities': {
                DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]): round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            }
        },
        'tumor_size': size_info,
        'risk_level': risk_info,
        'images': {
            'original': original_b64,
            'heatmap': heatmap_b64,
            'overlay': overlay_b64
        },
        'activation_percentage': round(activation_area * 100, 2)
    }


def _numpy_to_base64(img_array):
    """Convert numpy array to base64-encoded PNG string."""
    if img_array.dtype == np.float64 or img_array.dtype == np.float32:
        img_array = np.uint8(np.clip(img_array * 255, 0, 255))
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ── Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Serve static frontend files."""
    return send_from_directory(app.static_folder, path)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Analyze an uploaded MRI image.
    
    Expects: multipart/form-data with 'file' field
    Returns: JSON with prediction, heatmap, size, risk analysis
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed:
            return jsonify({'error': f'Unsupported file type: {ext}. Allowed: {", ".join(allowed)}'}), 400
        
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOADS_DIR, filename)
        file.save(filepath)
        
        # Run analysis
        result = analyze_image(filepath)
        result['filename'] = file.filename
        
        # Auto-generate template report (fast, always available)
        report_data = generate_report({
            'tumor_detected': result['prediction']['has_tumor'],
            'tumor_type': result['prediction']['class'],
            'confidence': result['prediction']['confidence'] / 100.0,
            'tumor_size': result['tumor_size']['size_category'],
            'risk_level': result['risk_level']['risk_level']
        }, use_llm=False)  # Template only for fast response
        result['report'] = report_data
        
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Compare multiple MRI scans for progress tracking.
    
    Expects: multipart/form-data with multiple 'files' fields
    Returns: JSON with per-scan analysis and comparison results
    """
    try:
        files = request.files.getlist('files')
        if len(files) < 2:
            return jsonify({'error': 'At least 2 files are required for comparison'}), 400
        
        if len(files) > 5:
            return jsonify({'error': 'Maximum 5 files allowed for comparison'}), 400
        
        scan_results = []
        
        for i, file in enumerate(files):
            # Save file
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(UPLOADS_DIR, filename)
            file.save(filepath)
            
            # Analyze
            result = analyze_image(filepath)
            result['scan_label'] = f"Scan {i + 1}"
            result['original_filename'] = file.filename
            
            # Flatten for comparison module
            scan_data = {
                'scan_label': result['scan_label'],
                'original_filename': file.filename,
                'tumor_class': result['prediction']['class'],
                'display_name': result['prediction']['display_name'],
                'confidence': result['prediction']['confidence'],
                'has_tumor': result['prediction']['has_tumor'],
                'size_category': result['tumor_size']['size_category'],
                'activation_percentage': result['activation_percentage'],
                'risk_level': result['risk_level']['risk_level'],
                'images': result['images'],
                'probabilities': result['prediction']['probabilities']
            }
            scan_results.append(scan_data)
        
        # Run comparison
        comparison = compare_scans(scan_results)
        
        return jsonify({
            'scans': scan_results,
            'comparison': comparison
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@app.route('/api/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    Generate an AI-powered medical report from prediction data.
    
    Expects JSON body with prediction data, or multipart with 'file' field.
    If a file is provided, runs full analysis first then generates report.
    Uses LLM when available, falls back to template.
    """
    try:
        # Option 1: File upload → analyze → report
        if 'file' in request.files:
            file = request.files['file']
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(UPLOADS_DIR, filename)
            file.save(filepath)
            
            result = analyze_image(filepath)
            prediction_data = {
                'tumor_detected': result['prediction']['has_tumor'],
                'tumor_type': result['prediction']['class'],
                'confidence': result['prediction']['confidence'] / 100.0,
                'tumor_size': result['tumor_size']['size_category'],
                'risk_level': result['risk_level']['risk_level']
            }
        # Option 2: JSON body with prediction data
        elif request.is_json:
            prediction_data = request.get_json()
        else:
            return jsonify({'error': 'Provide either a file upload or JSON prediction data'}), 400
        
        # Use LLM for this dedicated endpoint
        use_llm = request.args.get('use_llm', 'true').lower() == 'true'
        report_data = generate_report(prediction_data, use_llm=use_llm)
        
        return jsonify(report_data)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    return jsonify({
        'status': 'ok',
        'model_loaded': _model is not None,
        'model_exists': os.path.exists(model_path),
        'gpu_available': torch.cuda.is_available(),
        'device': str(_device) if _device else 'not initialized'
    })


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR DETECTION SYSTEM — SERVER")
    print("=" * 60)
    
    # Pre-load model
    try:
        get_model()
    except FileNotFoundError as e:
        print(f"\nWARNING: {e}")
        print("The server will start but predictions won't work until model is trained.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
