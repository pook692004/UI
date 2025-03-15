import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import torch.nn as nn
from torchvision import models
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'quality_fruit_app_secret_key'

# Configure storage directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
QUALITY_MODEL_PATH = r"D:\2025\fruit-classification-project\trained_models\fruit_quality_model.pth"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels for quality
quality_classes = ['Bad Quality', 'Good Quality', 'Mixed Quality']

# Load quality classification model
def load_quality_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, len(quality_classes))
    
    if os.path.exists(QUALITY_MODEL_PATH):
        model.load_state_dict(torch.load(QUALITY_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        print(f"Warning: Quality model not found at {QUALITY_MODEL_PATH}")
        return None

# Initialize model
quality_model = load_quality_model()

# Prediction function
def predict_quality(image_path):
    if quality_model is None:
        return "Model not loaded", 0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = quality_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_name = quality_classes[predicted.item()]
        accuracy = confidence.item() * 100  # Convert to percentage
    
    return class_name, accuracy

# Main route
@app.route('/')
def index():
    # Check if model is successfully loaded
    models_status = {
        'quality_model': quality_model is not None
    }
    return render_template('index.html', models_status=models_status)

# Route for handling uploads and classification
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('Không tìm thấy tệp', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn tệp', 'error')
        return redirect(url_for('index'))
    
    if file:
        # Ensure unique filename
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict quality if model is ready
        try:
            result, accuracy = predict_quality(filepath)
        except Exception as e:
            result = f"Lỗi phân loại: {str(e)}"
            accuracy = 0
        
        return render_template('result.html', 
                              filename=filename, 
                              result=result,
                              accuracy=accuracy)

# Route for only classification (JSON response for auto-capture)
@app.route('/classify_only', methods=['POST'])
def classify_only():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy tệp'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn tệp'}), 400
    
    if file:
        # Ensure unique filename
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict quality if model is ready
        try:
            result, accuracy = predict_quality(filepath)
            return jsonify({
                'filename': filename,
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'result': result,
                'accuracy': accuracy
            })
        except Exception as e:
            return jsonify({'error': f"Lỗi phân loại: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)