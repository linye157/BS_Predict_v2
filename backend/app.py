from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
import pickle
import tempfile
from werkzeug.utils import secure_filename

# Import API modules
from api.data_processing import DataProcessingAPI
from api.machine_learning import MachineLearningAPI
from api.stacking_ensemble import StackingEnsembleAPI
from api.auto_ml import AutoMLAPI
from api.visualization import VisualizationAPI
from api.report import ReportAPI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Initialize API modules
data_processing_api = DataProcessingAPI()
machine_learning_api = MachineLearningAPI()
stacking_ensemble_api = StackingEnsembleAPI()
auto_ml_api = AutoMLAPI()
visualization_api = VisualizationAPI()
report_api = ReportAPI()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes for Data Processing
@app.route('/api/data/load_default', methods=['GET'])
def load_default_data():
    return data_processing_api.load_default_data()

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'train')  # 'train' or 'test'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return data_processing_api.process_uploaded_file(filepath, file_type)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/data/download', methods=['GET'])
def download_data():
    data_type = request.args.get('type', 'train')  # 'train' or 'test'
    format_type = request.args.get('format', 'csv')  # 'csv' or 'excel'
    
    return data_processing_api.download_data(data_type, format_type)

@app.route('/api/data/preview', methods=['GET'])
def preview_data():
    data_type = request.args.get('type', 'train')  # 'train' or 'test'
    
    return data_processing_api.get_data_preview(data_type)

@app.route('/api/data/preprocess', methods=['POST'])
def preprocess_data():
    data = request.json
    preprocess_options = data.get('options', [])
    
    return data_processing_api.preprocess_data(preprocess_options, data)

# Routes for Machine Learning
@app.route('/api/ml/prepare', methods=['POST'])
def prepare_data():
    data = request.json
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    
    return machine_learning_api.prepare_data(test_size, random_state)

@app.route('/api/ml/train', methods=['POST'])
def train_model():
    data = request.json
    model_type = data.get('model_type')
    target_idx = data.get('target_idx', 0)
    params = data.get('params', {})
    
    if not model_type:
        return jsonify({'error': 'No model type specified'}), 400
    
    return machine_learning_api.train_model(model_type, target_idx, params)

@app.route('/api/ml/compare', methods=['GET'])
def compare_models():
    return machine_learning_api.compare_models()

@app.route('/api/ml/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model_name')
    input_data = data.get('data')
    
    if not model_name or not input_data:
        return jsonify({'error': 'Missing model name or input data'}), 400
    
    return machine_learning_api.predict(model_name, input_data)

# Routes for Stacking Ensemble
@app.route('/api/stacking/train', methods=['POST'])
def train_stacking_ensemble():
    data = request.json
    base_models = data.get('base_models', [])
    meta_model = data.get('meta_model', 'lr')
    target_idx = data.get('target_idx', 0)
    
    return stacking_ensemble_api.train_stacking_ensemble(base_models, meta_model, target_idx)

# Routes for Auto ML
@app.route('/api/automl/train', methods=['POST'])
def train_automl():
    data = request.json
    target_idx = data.get('target_idx', 0)
    time_limit = data.get('time_limit', 60)
    
    return auto_ml_api.train_automl(target_idx, time_limit)

# Routes for Visualization
@app.route('/api/visualization/data', methods=['POST'])
def visualize_data():
    data = request.json
    viz_type = data.get('viz_type')
    params = data.get('params', {})
    
    return visualization_api.visualize_data(viz_type, params)

@app.route('/api/visualization/model', methods=['POST'])
def visualize_model():
    data = request.json
    model_name = data.get('model_name')
    viz_type = data.get('viz_type')
    
    return visualization_api.visualize_model(model_name, viz_type)

@app.route('/api/visualization/results', methods=['POST'])
def visualize_results():
    data = request.json
    model_names = data.get('model_names', [])
    viz_type = data.get('viz_type')
    
    return visualization_api.visualize_results(model_names, viz_type)

# Routes for Report Generation
@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    data = request.json
    report_type = data.get('report_type', 'full')
    model_names = data.get('model_names', [])
    
    return report_api.generate_report(report_type, model_names)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 