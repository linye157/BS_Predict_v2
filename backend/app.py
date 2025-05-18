from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
import pickle
import tempfile
from werkzeug.utils import secure_filename
import traceback # 引入 traceback

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
def download_data_route(): # Renamed to avoid conflict with function name
    data_type = request.args.get('type', 'train')  # 'train' or 'test'
    format_type = request.args.get('format', 'csv')  # 'csv' or 'excel'
    
    return data_processing_api.download_data(data_type, format_type)

@app.route('/api/data/preview', methods=['GET'])
def preview_data():
    data_type = request.args.get('type', 'train')  # 'train' or 'test'
    
    return data_processing_api.get_data_preview(data_type)

@app.route('/api/data/preprocess', methods=['POST'])
def preprocess_data_route(): # Renamed to avoid conflict
    data = request.json
    preprocess_options = data.get('options', [])
    
    return data_processing_api.preprocess_data(preprocess_options, data)

# Routes for Machine Learning
@app.route('/api/ml/prepare', methods=['POST'])
def prepare_data_route(): # Renamed to avoid conflict
    data = request.json
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    
    if data_processing_api.train_data is None:
        return jsonify({
            'status': 'error',
            'message': 'No training data available'
        }), 404
    
    machine_learning_api.train_data = data_processing_api.train_data
    if data_processing_api.test_data is not None:
        machine_learning_api.test_data = data_processing_api.test_data
    
    return machine_learning_api.prepare_data(test_size, random_state)

@app.route('/api/ml/train', methods=['POST'])
def train_model_route(): # Renamed to avoid conflict
    data = request.json
    model_type = data.get('model_type')
    target_idx = data.get('target_idx', 0)
    params = data.get('params', {})
    
    if not model_type:
        return jsonify({'error': 'No model type specified'}), 400
    
    return machine_learning_api.train_model(model_type, target_idx, params)

@app.route('/api/ml/compare', methods=['GET'])
def compare_models_route(): # Renamed to avoid conflict
    return machine_learning_api.compare_models()

@app.route('/api/ml/predict', methods=['POST'])
def predict_route(): # Renamed to avoid conflict
    data = request.json
    model_name = data.get('model_name')
    input_data_req = data.get('data') # Renamed to avoid conflict with flask.data
    
    if not model_name or not input_data_req:
        return jsonify({'error': 'Missing model name or input data'}), 400
    
    return machine_learning_api.predict(model_name, input_data_req)

# Routes for Stacking Ensemble
@app.route('/api/stacking/train', methods=['POST'])
def train_stacking_ensemble_route(): # Renamed
    try:
        data = request.json
        base_models = data.get('base_models', [])
        meta_model = data.get('meta_model', 'lr')
        target_idx = data.get('target_idx', 0)
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)

        # 检查数据处理是否已完成
        if data_processing_api.train_data is None:
            return jsonify({
                'status': 'error',
                'message': '未找到训练数据。请先在数据处理部分加载数据。'
            }), 400

        # 检查基础模型列表是否为空
        if not base_models or len(base_models) < 2:
            return jsonify({
                'status': 'error',
                'message': '至少需要两个基础模型才能训练Stacking集成。'
            }), 400

        # 准备数据分割 - 从数据处理API直接获取数据，并进行训练/测试分割
        if machine_learning_api.X_train is None or machine_learning_api.y_train is None:
            # 如果机器学习API中没有准备好的数据，则从数据处理API获取并准备
            machine_learning_api.train_data = data_processing_api.train_data
            if data_processing_api.test_data is not None:
                machine_learning_api.test_data = data_processing_api.test_data
            
            # 调用机器学习API的数据准备方法
            prep_result = machine_learning_api.prepare_data(test_size, random_state)
            if isinstance(prep_result, tuple) and prep_result[1] >= 400:
                return prep_result  # 返回错误

        # 现在从机器学习API复制准备好的数据
        stacking_ensemble_api.X_train = machine_learning_api.X_train
        stacking_ensemble_api.y_train = machine_learning_api.y_train
        stacking_ensemble_api.X_test = machine_learning_api.X_test
        stacking_ensemble_api.y_test = machine_learning_api.y_test
        
        # 训练需要的基础模型（如果未训练）。兼容前端传入的模型名称既可能为 "rf" 这样的前缀，也可能已带有 "rf_target_0" 这样的后缀。
        normalized_base_model_names = []  # 保存标准化后的完整模型名称（带 target 后缀），后续传递给 StackingEnsembleAPI
        trained_model_count = 0  # 记录成功训练的模型数量

        for base_model_entry in base_models:
            # 判断是否已经包含目标索引后缀
            if base_model_entry.endswith(f"_target_{target_idx}"):
                model_prefix = base_model_entry.split('_target_')[0]
                model_full_name = base_model_entry
            else:
                model_prefix = base_model_entry
                model_full_name = f"{base_model_entry}_target_{target_idx}"

            # 若该模型尚未训练，则先训练
            if model_full_name not in machine_learning_api.models:
                print(f"训练基础模型: {model_prefix}，目标索引: {target_idx}")  # 调试输出
                train_result = machine_learning_api.train_model(model_prefix, target_idx, {})
                
                if isinstance(train_result, tuple) and train_result[1] >= 400:
                    return jsonify({
                        'status': 'error',
                        'message': f'训练基础模型 {model_prefix} 失败: {train_result[0].json.get("message", "未知错误")}'
                    }), 400
            
            # 验证模型是否已经成功加载/训练
            if model_full_name in machine_learning_api.models:
                normalized_base_model_names.append(model_full_name)
                trained_model_count += 1
            else:
                # 模型训练后仍不可用，可能是个问题
                return jsonify({
                    'status': 'error',
                    'message': f'基础模型 {model_prefix} 训练完成但在模型字典中未找到。'
                }), 500

        # 检查是否有足够的基础模型被成功训练
        if trained_model_count < 2:
            return jsonify({
                'status': 'error',
                'message': f'训练成功的基础模型数量不足(需要至少2个，当前只有{trained_model_count}个)。'
            }), 400

        # 更新 base_models 为标准化后的名称，确保后续流程一致
        base_models = normalized_base_model_names

        # 将训练好的模型传递给stacking API
        stacking_ensemble_api.trained_base_models = machine_learning_api.models
        stacking_ensemble_api.base_model_results = machine_learning_api.model_results
        
        # 调用stacking模型训练
        result = stacking_ensemble_api.train_stacking_ensemble(base_models, meta_model, target_idx)
        return result
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Stacking集成训练发生错误: {str(e)}\n{tb}")
        return jsonify({
            'status': 'error',
            'message': f'Stacking集成训练过程中发生错误: {str(e)}',
            'traceback': tb
        }), 500

# Routes for Auto ML
@app.route('/api/automl/train', methods=['POST'])
def train_automl_route(): # Renamed
    data = request.json
    target_idx = data.get('target_idx', 0)
    time_limit = data.get('time_limit', 60)
    
    # Pass necessary data from ML API to AutoML API
    if machine_learning_api.X_train is None or machine_learning_api.y_train is None:
         return jsonify({'status': 'error', 'message': 'Training data not prepared in ML API.'}), 400
    
    auto_ml_api.X_train = machine_learning_api.X_train
    auto_ml_api.y_train = machine_learning_api.y_train
    auto_ml_api.X_test = machine_learning_api.X_test
    auto_ml_api.y_test = machine_learning_api.y_test
    auto_ml_api.ml_api_instance = machine_learning_api # Pass ML API instance to save best model

    return auto_ml_api.train_automl(target_idx, time_limit)


# Routes for Visualization
@app.route('/api/visualization/data', methods=['POST'])
def visualize_data_route(): # Renamed
    data = request.json
    viz_type = data.get('viz_type')
    params = data.get('params', {})
    
    # Pass DataProcessingAPI instance to VisualizationAPI
    visualization_api.data_processing_api_instance = data_processing_api
    return visualization_api.visualize_data(viz_type, params)

@app.route('/api/visualization/model', methods=['POST'])
def visualize_model_route(): # Renamed
    data = request.json
    model_name = data.get('model_name')
    viz_type = data.get('viz_type')

    # Pass MachineLearningAPI instance to VisualizationAPI
    visualization_api.machine_learning_api_instance = machine_learning_api
    return visualization_api.visualize_model(model_name, viz_type)

@app.route('/api/visualization/results', methods=['POST'])
def visualize_results_route(): # Renamed
    data = request.json
    model_names = data.get('model_names', [])
    viz_type = data.get('viz_type')

    # Pass MachineLearningAPI instance to VisualizationAPI
    visualization_api.machine_learning_api_instance = machine_learning_api
    return visualization_api.visualize_results(model_names, viz_type)

# Routes for Report Generation
@app.route('/api/report/generate', methods=['POST'])
def generate_report_route(): # Renamed
    data = request.json
    report_type = data.get('report_type', 'full')
    model_names = data.get('model_names', [])

    # Pass API instances to ReportAPI
    report_api.machine_learning_api_instance = machine_learning_api
    report_api.data_processing_api_instance = data_processing_api
    return report_api.generate_report(report_type, model_names)

if __name__ == '__main__':
    app.run(debug=True, port=5000)