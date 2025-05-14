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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

class AutoMLAPI:
    def __init__(self):
        self.model_folder = Path('models')
        self.best_model = None
        self.num_targets = 3
        
        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)
    
    def train_automl(self, target_idx=0, time_limit=60):
        """Train multiple models and select the best one"""
        try:
            # Get data from MachineLearningAPI
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Check if training data is prepared
            if ml_api.X_train is None or ml_api.y_train is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Training data not prepared. Call prepare_data first.'
                }), 400
            
            # Get training target
            y_train_target = ml_api.y_train.iloc[:, target_idx]
            y_test_target = ml_api.y_test.iloc[:, target_idx]
            
            # Start timing
            overall_start_time = time.time()
            
            # Define models to try
            models = [
                {
                    'name': 'lr',
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False]
                    }
                },
                {
                    'name': 'rf',
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                {
                    'name': 'gbr',
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }
                },
                {
                    'name': 'xgbr',
                    'model': XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                {
                    'name': 'svr',
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto', 0.1, 1]
                    }
                },
                {
                    'name': 'ann',
                    'model': MLPRegressor(random_state=42, max_iter=300),
                    'params': {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive']
                    }
                }
            ]
            
            # Train and evaluate models with time limit
            results = []
            remaining_time = time_limit
            
            for model_info in models:
                if remaining_time <= 0:
                    break
                    
                model_start_time = time.time()
                model_time_limit = min(remaining_time, time_limit / len(models) * 2)  # Allocate time proportionally
                
                # Perform randomized search with time limit
                search = RandomizedSearchCV(
                    model_info['model'],
                    model_info['params'],
                    n_iter=10,  # Limited iterations for quick search
                    cv=3,  # Use 3-fold CV for speed
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
                
                try:
                    # Set a timeout using time_limit
                    search.fit(ml_api.X_train, y_train_target)
                    
                    # Get best model
                    best_model = search.best_estimator_
                    
                    # Make predictions
                    train_preds = best_model.predict(ml_api.X_train)
                    test_preds = best_model.predict(ml_api.X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train_target, train_preds))
                    test_rmse = np.sqrt(mean_squared_error(y_test_target, test_preds))
                    train_r2 = r2_score(y_train_target, train_preds)
                    test_r2 = r2_score(y_test_target, test_preds)
                    train_mae = mean_absolute_error(y_train_target, train_preds)
                    test_mae = mean_absolute_error(y_test_target, test_preds)
                    
                    # Store results
                    results.append({
                        'name': model_info['name'],
                        'model': best_model,
                        'params': search.best_params_,
                        'metrics': {
                            'train_rmse': float(train_rmse),
                            'test_rmse': float(test_rmse),
                            'train_r2': float(train_r2),
                            'test_r2': float(test_r2),
                            'train_mae': float(train_mae),
                            'test_mae': float(test_mae)
                        }
                    })
                except Exception as e:
                    # If a model fails, skip it
                    print(f"Error training {model_info['name']}: {str(e)}")
                
                # Update remaining time
                model_elapsed_time = time.time() - model_start_time
                remaining_time -= model_elapsed_time
            
            # Find best model based on test_rmse
            if not results:
                return jsonify({
                    'status': 'error',
                    'message': 'No models were successfully trained'
                }), 500
            
            # Sort by test RMSE (lower is better)
            results.sort(key=lambda x: x['metrics']['test_rmse'])
            best_result = results[0]
            best_model = best_result['model']
            
            # Model name
            model_name = f"automl_{best_result['name']}_target_{target_idx}"
            
            # Save best model
            model_path = self.model_folder / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Create prediction vs actual plot
            plt.figure(figsize=(10, 6))
            test_preds = best_model.predict(ml_api.X_test)
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
            
            # Create comparison plot for all models
            plt.figure(figsize=(12, 6))
            model_names = [r['name'] for r in results]
            test_rmses = [r['metrics']['test_rmse'] for r in results]
            
            # Sort by performance
            sorted_idx = np.argsort(test_rmses)
            sorted_model_names = [model_names[i] for i in sorted_idx]
            sorted_test_rmses = [test_rmses[i] for i in sorted_idx]
            
            plt.barh(sorted_model_names, sorted_test_rmses)
            plt.xlabel('Test RMSE (lower is better)')
            plt.title('AutoML Model Comparison')
            plt.tight_layout()
            
            # Convert comparison plot to base64 string
            comp_buffer = io.BytesIO()
            plt.savefig(comp_buffer, format='png')
            comp_buffer.seek(0)
            comparison_plot = base64.b64encode(comp_buffer.read()).decode('utf-8')
            plt.close()
            
            # Feature importance if available
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_.tolist()
            elif hasattr(best_model, 'coef_'):
                feature_importance = best_model.coef_.tolist()
            
            # End timing
            overall_elapsed_time = time.time() - overall_start_time
            
            # Store in ML API for future use
            ml_api.models[model_name] = best_model
            ml_api.model_results[model_name] = {
                'train_rmse': best_result['metrics']['train_rmse'],
                'test_rmse': best_result['metrics']['test_rmse'],
                'train_r2': best_result['metrics']['train_r2'],
                'test_r2': best_result['metrics']['test_r2'],
                'train_mae': best_result['metrics']['train_mae'],
                'test_mae': best_result['metrics']['test_mae'],
                'training_time': overall_elapsed_time,
                'target_idx': target_idx,
                'feature_importance': feature_importance,
                'feature_names': ml_api.X_train.columns.tolist(),
                'params': best_result['params'],
                'model_type': best_result['name']
            }
            
            return jsonify({
                'status': 'success',
                'message': f'AutoML completed successfully. Best model: {model_name}',
                'model_name': model_name,
                'best_model_type': best_result['name'],
                'best_params': best_result['params'],
                'metrics': best_result['metrics'],
                'training_time': overall_elapsed_time,
                'all_models': [
                    {
                        'name': r['name'],
                        'params': r['params'],
                        'metrics': r['metrics']
                    } for r in results
                ],
                'feature_importance': {
                    'values': feature_importance,
                    'feature_names': ml_api.X_train.columns.tolist()
                } if feature_importance is not None else None,
                'plot': plot_data,
                'comparison_plot': comparison_plot
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error in AutoML: {str(e)}'
            }), 500 