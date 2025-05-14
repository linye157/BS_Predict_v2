from flask import jsonify
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class StackingEnsembleAPI:
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.model_folder = Path('models')
        self.num_targets = 3
        
        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)
    
    def train_stacking_ensemble(self, base_models, meta_model='lr', target_idx=0):
        """Train a stacking ensemble model"""
        try:
            # Get data and models from MachineLearningAPI
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Check if training data is prepared
            if ml_api.X_train is None or ml_api.y_train is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Training data not prepared. Call prepare_data first.'
                }), 400
            
            # Check if base models are specified
            if not base_models:
                return jsonify({
                    'status': 'error',
                    'message': 'No base models specified'
                }), 400
            
            # Get training target
            y_train_target = ml_api.y_train.iloc[:, target_idx]
            y_test_target = ml_api.y_test.iloc[:, target_idx]
            
            # Load base models or use existing ones
            loaded_base_models = []
            for model_name in base_models:
                if model_name in ml_api.models:
                    loaded_base_models.append((model_name, ml_api.models[model_name]))
                else:
                    # Try to load from disk
                    model_path = self.model_folder / f"{model_name}.pkl"
                    if not model_path.exists():
                        return jsonify({
                            'status': 'error',
                            'message': f'Base model {model_name} not found'
                        }), 404
                    
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        loaded_base_models.append((model_name, model))
            
            # Create meta-model
            meta_model_instance = self._create_meta_model(meta_model)
            
            if meta_model_instance is None:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported meta-model type: {meta_model}'
                }), 400
            
            # Start timing
            start_time = time.time()
            
            # Train stacking ensemble using k-fold cross-validation
            k_folds = 5
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            # Initialize out-of-fold predictions for training data
            X_train_meta = np.zeros((ml_api.X_train.shape[0], len(loaded_base_models)))
            
            # Train base models on k-fold data and generate out-of-fold predictions
            for i, (model_name, model) in enumerate(loaded_base_models):
                # Clone model to avoid modifying the original
                if model_name.startswith('lr'):
                    fold_model = LinearRegression(fit_intercept=True)
                elif model_name.startswith('rf'):
                    fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name.startswith('gbr'):
                    fold_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_name.startswith('xgbr'):
                    fold_model = XGBRegressor(n_estimators=100, random_state=42)
                elif model_name.startswith('svr'):
                    fold_model = SVR(C=1.0, kernel='rbf')
                elif model_name.startswith('ann'):
                    fold_model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
                else:
                    # Default to RF if model type can't be determined
                    fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                # Generate out-of-fold predictions
                for train_idx, val_idx in kf.split(ml_api.X_train):
                    # Split data
                    X_fold_train = ml_api.X_train.iloc[train_idx]
                    y_fold_train = y_train_target.iloc[train_idx]
                    X_fold_val = ml_api.X_train.iloc[val_idx]
                    
                    # Train model on fold
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Generate predictions for validation fold
                    X_train_meta[val_idx, i] = fold_model.predict(X_fold_val)
            
            # Generate predictions for test data using full models
            X_test_meta = np.zeros((ml_api.X_test.shape[0], len(loaded_base_models)))
            for i, (model_name, model) in enumerate(loaded_base_models):
                X_test_meta[:, i] = model.predict(ml_api.X_test)
            
            # Train meta-model on out-of-fold predictions
            meta_model_instance.fit(X_train_meta, y_train_target)
            
            # Make predictions
            train_preds = meta_model_instance.predict(X_train_meta)
            test_preds = meta_model_instance.predict(X_test_meta)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_target, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test_target, test_preds))
            train_r2 = r2_score(y_train_target, train_preds)
            test_r2 = r2_score(y_test_target, test_preds)
            train_mae = mean_absolute_error(y_train_target, train_preds)
            test_mae = mean_absolute_error(y_test_target, test_preds)
            
            # End timing
            end_time = time.time()
            
            # Model name
            model_name = f"stacking_{meta_model}_target_{target_idx}"
            
            # Save ensemble model
            ensemble_model = {
                'base_models': [name for name, _ in loaded_base_models],
                'meta_model': meta_model_instance,
                'meta_model_type': meta_model
            }
            
            model_path = self.model_folder / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble_model, f)
            
            # Store model in memory
            self.base_models[model_name] = ensemble_model
            
            # Create prediction vs actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_target, test_preds, alpha=0.5)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted Values for {model_name}')
            
            # Add perfect prediction line
            min_val = min(y_test_target.min(), test_preds.min())
            max_val = max(y_test_target.max(), test_preds.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            # Compare stacking to base models
            comparison_data = {
                'model_names': [name for name, _ in loaded_base_models] + [model_name],
                'test_rmse': []
            }
            
            # Get test RMSE for base models
            for model_name, model in loaded_base_models:
                base_preds = model.predict(ml_api.X_test)
                base_rmse = np.sqrt(mean_squared_error(y_test_target, base_preds))
                comparison_data['test_rmse'].append(float(base_rmse))
            
            # Add stacking RMSE
            comparison_data['test_rmse'].append(float(test_rmse))
            
            # Create comparison plot
            plt.figure(figsize=(12, 6))
            plt.barh(comparison_data['model_names'], comparison_data['test_rmse'])
            plt.xlabel('Test RMSE (lower is better)')
            plt.title(f'Model Comparison - Stacking vs Base Models - Target {target_idx}')
            plt.tight_layout()
            
            # Convert comparison plot to base64 string
            comp_buffer = io.BytesIO()
            plt.savefig(comp_buffer, format='png')
            comp_buffer.seek(0)
            comparison_plot = base64.b64encode(comp_buffer.read()).decode('utf-8')
            plt.close()
            
            return jsonify({
                'status': 'success',
                'message': f'Stacking ensemble {model_name} trained successfully',
                'model_name': model_name,
                'metrics': {
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'training_time': end_time - start_time
                },
                'base_models': [name for name, _ in loaded_base_models],
                'meta_model_type': meta_model,
                'plot': plot_data,
                'comparison_plot': comparison_plot,
                'comparison_data': comparison_data
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error training stacking ensemble: {str(e)}'
            }), 500
    
    def _create_meta_model(self, meta_model_type):
        """Create a meta-model instance based on type"""
        if meta_model_type == 'lr':
            return LinearRegression()
        elif meta_model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif meta_model_type == 'gbr':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif meta_model_type == 'xgbr':
            return XGBRegressor(n_estimators=100, random_state=42)
        elif meta_model_type == 'svr':
            return SVR(C=1.0, kernel='rbf')
        else:
            return None
    
    def predict(self, model_name, input_data):
        """Make predictions using a stacking ensemble model"""
        try:
            # Check if model exists in memory
            if model_name not in self.base_models:
                # Try to load from disk
                model_path = self.model_folder / f"{model_name}.pkl"
                if not model_path.exists():
                    return jsonify({
                        'status': 'error',
                        'message': f'Model {model_name} not found'
                    }), 404
                
                with open(model_path, 'rb') as f:
                    self.base_models[model_name] = pickle.load(f)
            
            ensemble_model = self.base_models[model_name]
            base_models = ensemble_model['base_models']
            meta_model = ensemble_model['meta_model']
            
            # Load ML API to get base models and feature info
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Validate input data
            try:
                if isinstance(input_data, list):
                    if len(input_data) > 0 and isinstance(input_data[0], dict):
                        # List of dictionaries (records format)
                        input_df = pd.DataFrame(input_data)
                    else:
                        # List of values (assume single record)
                        input_df = pd.DataFrame([input_data], columns=ml_api.X_train.columns)
                elif isinstance(input_data, dict):
                    # Single dictionary
                    input_df = pd.DataFrame([input_data])
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid input data format'
                    }), 400
                
                # Ensure all required columns are present
                missing_cols = set(ml_api.X_train.columns) - set(input_df.columns)
                if missing_cols:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing columns in input data: {list(missing_cols)}'
                    }), 400
                
                # Select only the columns used in training and in the same order
                input_df = input_df[ml_api.X_train.columns]
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing input data: {str(e)}'
                }), 400
            
            # Generate meta-features using base models
            meta_features = np.zeros((input_df.shape[0], len(base_models)))
            
            for i, model_name in enumerate(base_models):
                if model_name in ml_api.models:
                    model = ml_api.models[model_name]
                else:
                    # Try to load model from disk
                    model_path = self.model_folder / f"{model_name}.pkl"
                    if not model_path.exists():
                        return jsonify({
                            'status': 'error',
                            'message': f'Base model {model_name} not found'
                        }), 404
                    
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                # Generate predictions for this base model
                meta_features[:, i] = model.predict(input_df)
            
            # Make predictions using meta-model
            predictions = meta_model.predict(meta_features)
            
            # Convert to list
            predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            
            return jsonify({
                'status': 'success',
                'predictions': predictions_list,
                'model_name': model_name
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error making predictions with stacking ensemble: {str(e)}'
            }), 500 