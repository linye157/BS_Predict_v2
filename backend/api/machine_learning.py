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
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

class MachineLearningAPI:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.num_targets = 3  # Default value for number of target columns
        self.model_folder = Path('models')
        
        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        try:
            from api.data_processing import DataProcessingAPI
            data_api = DataProcessingAPI()
            
            # Get the training data
            if data_api.train_data is None:
                return jsonify({
                    'status': 'error',
                    'message': 'No training data available'
                }), 404
            
            self.train_data = data_api.train_data
            
            # Get features and targets
            X = self.train_data.iloc[:, :-self.num_targets]
            y = self.train_data.iloc[:, -self.num_targets:]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Data prepared successfully',
                'data': {
                    'X_train_shape': [self.X_train.shape[0], self.X_train.shape[1]],
                    'X_test_shape': [self.X_test.shape[0], self.X_test.shape[1]],
                    'y_train_shape': [self.y_train.shape[0], self.y_train.shape[1]],
                    'y_test_shape': [self.y_test.shape[0], self.y_test.shape[1]],
                    'feature_names': self.X_train.columns.tolist(),
                    'target_names': self.y_train.columns.tolist()
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error preparing data: {str(e)}'
            }), 500
    
    def train_model(self, model_type, target_idx=0, params={}):
        """Train a machine learning model"""
        try:
            if self.X_train is None or self.y_train is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Training data not prepared. Call prepare_data first.'
                }), 400
            
            # Training target
            y_train_target = self.y_train.iloc[:, target_idx]
            y_test_target = self.y_test.iloc[:, target_idx]
            
            # Train model based on type
            model = None
            start_time = time.time()
            
            if model_type == 'lr':
                # Linear Regression
                fit_intercept = params.get('fit_intercept', True)
                normalize = params.get('normalize', False)
                
                model = LinearRegression(fit_intercept=fit_intercept)
                model.fit(self.X_train, y_train_target)
                
                # Feature importance is coefficients for linear regression
                feature_importance = model.coef_.tolist() if hasattr(model, 'coef_') else None
                
            elif model_type == 'rf':
                # Random Forest
                n_estimators = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', None)
                min_samples_split = params.get('min_samples_split', 2)
                min_samples_leaf = params.get('min_samples_leaf', 1)
                max_features = params.get('max_features', 'sqrt')
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(self.X_train, y_train_target)
                
                feature_importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
                
            elif model_type == 'gbr':
                # Gradient Boosting Regressor
                n_estimators = params.get('n_estimators', 100)
                learning_rate = params.get('learning_rate', 0.1)
                max_depth = params.get('max_depth', 3)
                min_samples_split = params.get('min_samples_split', 2)
                
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                model.fit(self.X_train, y_train_target)
                
                feature_importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
                
            elif model_type == 'xgbr':
                # XGBoost Regressor
                n_estimators = params.get('n_estimators', 100)
                learning_rate = params.get('learning_rate', 0.1)
                max_depth = params.get('max_depth', 3)
                subsample = params.get('subsample', 1.0)
                colsample_bytree = params.get('colsample_bytree', 1.0)
                
                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(self.X_train, y_train_target)
                
                feature_importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
                
            elif model_type == 'svr':
                # Support Vector Regressor
                C = params.get('C', 1.0)
                epsilon = params.get('epsilon', 0.1)
                kernel = params.get('kernel', 'rbf')
                gamma = params.get('gamma', 'scale')
                
                model = SVR(
                    C=C,
                    epsilon=epsilon,
                    kernel=kernel,
                    gamma=gamma
                )
                model.fit(self.X_train, y_train_target)
                
                # SVR doesn't have direct feature importances
                feature_importance = None
                
            elif model_type == 'ann':
                # Neural Network
                hidden_layer_sizes = params.get('hidden_layer_sizes', (100,))
                activation = params.get('activation', 'relu')
                solver = params.get('solver', 'adam')
                alpha = params.get('alpha', 0.0001)
                learning_rate = params.get('learning_rate', 'constant')
                max_iter = params.get('max_iter', 200)
                
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    random_state=42
                )
                model.fit(self.X_train, y_train_target)
                
                # Neural networks don't have direct feature importances
                feature_importance = None
            
            end_time = time.time()
            
            if model is None:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported model type: {model_type}'
                }), 400
            
            # Make predictions
            train_preds = model.predict(self.X_train)
            test_preds = model.predict(self.X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_target, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test_target, test_preds))
            train_r2 = r2_score(y_train_target, train_preds)
            test_r2 = r2_score(y_test_target, test_preds)
            train_mae = mean_absolute_error(y_train_target, train_preds)
            test_mae = mean_absolute_error(y_test_target, test_preds)
            
            # Model name
            model_name = f"{model_type}_target_{target_idx}"
            
            # Store model
            self.models[model_name] = model
            
            # Store results
            self.model_results[model_name] = {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'training_time': end_time - start_time,
                'target_idx': target_idx,
                'feature_importance': feature_importance,
                'feature_names': self.X_train.columns.tolist(),
                'params': params,
                'model_type': model_type
            }
            
            # Save model to disk
            model_path = self.model_folder / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
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
            
            return jsonify({
                'status': 'success',
                'message': f'Model {model_name} trained successfully',
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
                'feature_importance': {
                    'values': feature_importance,
                    'feature_names': self.X_train.columns.tolist()
                } if feature_importance is not None else None,
                'plot': plot_data
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error training model: {str(e)}'
            }), 500
    
    def compare_models(self):
        """Compare all trained models"""
        try:
            if not self.model_results:
                return jsonify({
                    'status': 'error',
                    'message': 'No models trained yet'
                }), 404
            
            # Group models by target index
            target_groups = {}
            for model_name, results in self.model_results.items():
                target_idx = results['target_idx']
                if target_idx not in target_groups:
                    target_groups[target_idx] = []
                
                target_groups[target_idx].append({
                    'model_name': model_name,
                    'metrics': {
                        'train_rmse': results['train_rmse'],
                        'test_rmse': results['test_rmse'],
                        'train_r2': results['train_r2'],
                        'test_r2': results['test_r2'],
                        'train_mae': results['train_mae'],
                        'test_mae': results['test_mae'],
                        'training_time': results['training_time']
                    }
                })
            
            # Create comparison plots for each target
            comparison_plots = {}
            for target_idx, models in target_groups.items():
                # Test RMSE comparison
                plt.figure(figsize=(12, 6))
                model_names = [model['model_name'] for model in models]
                test_rmses = [model['metrics']['test_rmse'] for model in models]
                
                # Sort by performance
                sorted_idx = np.argsort(test_rmses)
                sorted_model_names = [model_names[i] for i in sorted_idx]
                sorted_test_rmses = [test_rmses[i] for i in sorted_idx]
                
                plt.barh(sorted_model_names, sorted_test_rmses)
                plt.xlabel('Test RMSE (lower is better)')
                plt.title(f'Model Comparison - Target {target_idx}')
                plt.tight_layout()
                
                # Convert plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                comparison_plots[f"target_{target_idx}"] = plot_data
            
            return jsonify({
                'status': 'success',
                'target_groups': target_groups,
                'comparison_plots': comparison_plots
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error comparing models: {str(e)}'
            }), 500
    
    def predict(self, model_name, input_data):
        """Make predictions using a trained model"""
        try:
            # Check if model exists
            if model_name not in self.models:
                # Try to load from disk
                model_path = self.model_folder / f"{model_name}.pkl"
                if not model_path.exists():
                    return jsonify({
                        'status': 'error',
                        'message': f'Model {model_name} not found'
                    }), 404
                
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            # Get model
            model = self.models[model_name]
            
            # Convert input data to dataframe with correct columns
            try:
                if isinstance(input_data, list):
                    if len(input_data) > 0 and isinstance(input_data[0], dict):
                        # List of dictionaries (records format)
                        input_df = pd.DataFrame(input_data)
                    else:
                        # List of values (assume single record)
                        input_df = pd.DataFrame([input_data], columns=self.X_train.columns)
                elif isinstance(input_data, dict):
                    # Single dictionary
                    input_df = pd.DataFrame([input_data])
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid input data format'
                    }), 400
                
                # Ensure all required columns are present
                missing_cols = set(self.X_train.columns) - set(input_df.columns)
                if missing_cols:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing columns in input data: {list(missing_cols)}'
                    }), 400
                
                # Select only the columns used in training and in the same order
                input_df = input_df[self.X_train.columns]
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing input data: {str(e)}'
                }), 400
            
            # Make predictions
            predictions = model.predict(input_df)
            
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
                'message': f'Error making predictions: {str(e)}'
            }), 500 