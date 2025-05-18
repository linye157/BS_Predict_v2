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
        self.base_models = {}  # This will store the trained stacking ensemble itself
        self.meta_model = None
        self.model_folder = Path('models')
        self.num_targets = 3
        
        # Attributes to be populated from MachineLearningAPI
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trained_base_models = {} # To store base models passed from ML API

        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)
    
    def train_stacking_ensemble(self, base_model_names, meta_model_type='lr', target_idx=0): # Renamed parameters for clarity
        """Train a stacking ensemble model"""
        try:
            # Check if training data is prepared (now checking instance variables)
            if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Training data (X_train, y_train, X_test, y_test) not available in StackingEnsembleAPI. Ensure it is passed from MachineLearningAPI.'
                }), 400
            
            # Check if base models are specified
            if not base_model_names:
                return jsonify({
                    'status': 'error',
                    'message': 'No base model names specified'
                }), 400
            
            # Get training target
            y_train_target = self.y_train.iloc[:, target_idx]
            y_test_target = self.y_test.iloc[:, target_idx]
            
            # Load base models from the passed dictionary or try to load from disk
            loaded_base_models_for_stacking = []
            for model_name in base_model_names:
                if model_name in self.trained_base_models:
                    loaded_base_models_for_stacking.append((model_name, self.trained_base_models[model_name]))
                else:
                    # Try to load from disk if not found in the passed models (e.g., if app restarted or model trained in a previous session)
                    model_path = self.model_folder / f"{model_name}.pkl"
                    if not model_path.exists():
                        return jsonify({
                            'status': 'error',
                            'message': f'Base model {model_name} not found in trained models or on disk.'
                        }), 404
                    
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        loaded_base_models_for_stacking.append((model_name, model))
            
            if not loaded_base_models_for_stacking:
                 return jsonify({
                    'status': 'error',
                    'message': 'None of the specified base models could be loaded.'
                }), 400

            # Create meta-model
            meta_model_instance = self._create_meta_model(meta_model_type) # using renamed parameter
            
            if meta_model_instance is None:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported meta-model type: {meta_model_type}'
                }), 400
            
            # Start timing
            start_time = time.time()
            
            # Train stacking ensemble using k-fold cross-validation
            k_folds = 5
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            # Initialize out-of-fold predictions for training data
            X_train_meta = np.zeros((self.X_train.shape[0], len(loaded_base_models_for_stacking)))
            
            # Train base models on k-fold data and generate out-of-fold predictions
            for i, (model_name, model) in enumerate(loaded_base_models_for_stacking):
                # It's better to use the actual model instances passed rather than re-creating them here
                # However, for k-fold, we need to fit them on folds.
                # The original code re-created simplified models for folds. We'll keep that logic for now,
                # but ideally, it should clone the original model with its parameters.
                
                # Clone model to avoid modifying the original - Simplified recreation as per original logic
                # This part might need refinement to use actual parameters of loaded base models if they vary
                if model_name.startswith('lr'):
                    fold_model = LinearRegression(fit_intercept=True) # Consider params from original model
                elif model_name.startswith('rf'):
                    fold_model = RandomForestRegressor(n_estimators=100, random_state=42) # Consider params
                elif model_name.startswith('gbr'):
                    fold_model = GradientBoostingRegressor(n_estimators=100, random_state=42) # Consider params
                elif model_name.startswith('xgbr'):
                    fold_model = XGBRegressor(n_estimators=100, random_state=42) # Consider params
                elif model_name.startswith('svr'):
                    fold_model = SVR(C=1.0, kernel='rbf') # Consider params
                elif model_name.startswith('ann'):
                    fold_model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42) # Consider params
                else:
                    # Attempt to clone if it's a scikit-learn estimator
                    try:
                        from sklearn.base import clone
                        fold_model = clone(model)
                    except: # Fallback if clone fails or not a sklearn model with clone
                        return jsonify({
                            'status': 'error',
                            'message': f'Cannot determine type or clone base model {model_name} for k-fold. Ensure it is a known type or supports cloning.'
                        }), 400
                    
                # Generate out-of-fold predictions
                for train_idx_fold, val_idx_fold in kf.split(self.X_train): # Renamed variables
                    X_fold_train = self.X_train.iloc[train_idx_fold]
                    y_fold_train = y_train_target.iloc[train_idx_fold]
                    X_fold_val = self.X_train.iloc[val_idx_fold]
                    
                    fold_model.fit(X_fold_train, y_fold_train)
                    X_train_meta[val_idx_fold, i] = fold_model.predict(X_fold_val)
            
            # Generate predictions for test data using full (original) base models
            X_test_meta = np.zeros((self.X_test.shape[0], len(loaded_base_models_for_stacking)))
            for i, (model_name, model) in enumerate(loaded_base_models_for_stacking): # Use the original loaded models
                X_test_meta[:, i] = model.predict(self.X_test)
            
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
            
            end_time = time.time()
            
            # Ensemble Model name
            ensemble_model_name = f"stacking_{meta_model_type}_target_{target_idx}" # using renamed parameter
            
            # Save ensemble model (meta-model and names of base models)
            stacking_ensemble_details = {
                'base_model_names': [name for name, _ in loaded_base_models_for_stacking],
                'meta_model_instance': meta_model_instance, # Storing the fitted meta-model
                'meta_model_type': meta_model_type # using renamed parameter
            }
            
            model_path = self.model_folder / f"{ensemble_model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(stacking_ensemble_details, f)
            
            # Store trained ensemble in memory (perhaps using self.base_models as originally intended for ensembles)
            self.base_models[ensemble_model_name] = stacking_ensemble_details # Changed from self.base_models to self.trained_ensembles or similar would be clearer
                                                                             # but keeping self.base_models to align with predict method's expectation
            
            # Create prediction vs actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_target, test_preds, alpha=0.5)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted Values for {ensemble_model_name}')
            min_val = min(y_test_target.min(), test_preds.min())
            max_val = max(y_test_target.max(), test_preds.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            # Compare stacking to base models
            comparison_data = {
                'model_names': [name for name, _ in loaded_base_models_for_stacking] + [ensemble_model_name],
                'test_rmse': []
            }
            
            for model_name, model in loaded_base_models_for_stacking:
                base_preds = model.predict(self.X_test) # Use self.X_test
                base_rmse = np.sqrt(mean_squared_error(y_test_target, base_preds)) # Use y_test_target
                comparison_data['test_rmse'].append(float(base_rmse))
            comparison_data['test_rmse'].append(float(test_rmse))
            
            plt.figure(figsize=(12, 6))
            plt.barh(comparison_data['model_names'], comparison_data['test_rmse'])
            plt.xlabel('Test RMSE (lower is better)')
            plt.title(f'Model Comparison - Stacking vs Base Models - Target {target_idx}')
            plt.tight_layout()
            comp_buffer = io.BytesIO()
            plt.savefig(comp_buffer, format='png')
            comp_buffer.seek(0)
            comparison_plot = base64.b64encode(comp_buffer.read()).decode('utf-8')
            plt.close()
            
            return jsonify({
                'status': 'success',
                'message': f'Stacking ensemble {ensemble_model_name} trained successfully',
                'model_name': ensemble_model_name,
                'metrics': {
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'training_time': end_time - start_time
                },
                'base_models': [name for name, _ in loaded_base_models_for_stacking],
                'meta_model_type': meta_model_type, # using renamed parameter
                'plot': plot_data,
                'comparison_plot': comparison_plot,
                'comparison_data': comparison_data
            })
            
        except Exception as e:
            # Import traceback to get more detailed error information
            import traceback
            error_details = traceback.format_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error training stacking ensemble: {str(e)}',
                'details': error_details # Adding detailed traceback for debugging
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
    
    def predict(self, ensemble_model_name, input_data): # Renamed model_name to ensemble_model_name
        """Make predictions using a stacking ensemble model"""
        try:
            # Check if model exists in memory (self.base_models now stores ensemble details)
            if ensemble_model_name not in self.base_models:
                # Try to load from disk
                model_path = self.model_folder / f"{ensemble_model_name}.pkl"
                if not model_path.exists():
                    return jsonify({
                        'status': 'error',
                        'message': f'Stacking ensemble model {ensemble_model_name} not found in memory or on disk'
                    }), 404
                
                with open(model_path, 'rb') as f:
                    self.base_models[ensemble_model_name] = pickle.load(f) # Load ensemble details
            
            ensemble_details = self.base_models[ensemble_model_name]
            base_model_names_for_ensemble = ensemble_details['base_model_names'] # Corrected variable name
            meta_model_instance = ensemble_details['meta_model_instance'] # Corrected variable name
            
            # For input_df column consistency, we need X_train columns.
            # This implies StackingEnsembleAPI needs X_train, or at least its columns,
            # even for prediction if it's called independently.
            # The current structure sets self.X_train during training.
            # If predict is called in a new session/instance without training, self.X_train might be None.
            if self.X_train is None:
                 # Attempt to load X_train columns from a saved configuration or error out
                 # For now, let's assume X_train was set or error if not.
                 # A robust solution would involve saving/loading feature names with the model.
                return jsonify({
                    'status': 'error',
                    'message': 'X_train (for column names) not available in StackingEnsembleAPI. Train first or ensure data context is loaded.'
                }), 400

            # Validate input data
            try:
                input_df = pd.DataFrame() # Initialize input_df
                if isinstance(input_data, list):
                    if len(input_data) > 0 and isinstance(input_data[0], dict):
                        input_df = pd.DataFrame(input_data)
                    else:
                        # Ensure self.X_train.columns is available
                        if self.X_train is None or self.X_train.columns is None:
                             raise ValueError("X_train.columns is not available for creating DataFrame from list.")
                        input_df = pd.DataFrame([input_data], columns=self.X_train.columns)
                elif isinstance(input_data, dict):
                    input_df = pd.DataFrame([input_data])
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid input data format'}), 400
                
                # Ensure all required columns are present
                if self.X_train is None or self.X_train.columns is None:
                    raise ValueError("X_train.columns is not available for column validation.")

                missing_cols = set(self.X_train.columns) - set(input_df.columns)
                if missing_cols:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing columns in input data: {list(missing_cols)}. Required: {list(self.X_train.columns)}'
                    }), 400
                
                input_df = input_df[self.X_train.columns]
            except Exception as e:
                import traceback
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing input data for prediction: {str(e)}',
                    'details': traceback.format_exc()
                }), 400
            
            # Generate meta-features using base models
            # We need to load the actual base model instances.
            # The self.trained_base_models should be available or loaded here.
            # If predict is called in a new session, self.trained_base_models might be empty.
            # This part needs robust loading of base models.

            if not self.trained_base_models and not self.model_folder.exists():
                 return jsonify({'status': 'error', 'message': 'Trained base models not available and model folder not found.'}), 404

            meta_features = np.zeros((input_df.shape[0], len(base_model_names_for_ensemble)))
            
            for i, model_name_for_base in enumerate(base_model_names_for_ensemble): # Renamed variable
                base_model_instance = None # Initialize
                if model_name_for_base in self.trained_base_models:
                    base_model_instance = self.trained_base_models[model_name_for_base]
                else:
                    model_path = self.model_folder / f"{model_name_for_base}.pkl"
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            base_model_instance = pickle.load(f)
                            # Optionally cache it in self.trained_base_models if predict is called multiple times
                            if self.trained_base_models is None: self.trained_base_models = {} # Ensure dict exists
                            self.trained_base_models[model_name_for_base] = base_model_instance
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': f'Base model {model_name_for_base} for stacking ensemble not found in memory or on disk.'
                        }), 404
                
                if base_model_instance is None: # Should not happen if logic above is correct
                     return jsonify({'status': 'error', 'message': f'Failed to load base model {model_name_for_base}.'}), 500

                meta_features[:, i] = base_model_instance.predict(input_df)
            
            # Make predictions using meta-model
            predictions = meta_model_instance.predict(meta_features)
            predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else [predictions] # Ensure list
            
            return jsonify({
                'status': 'success',
                'predictions': predictions_list,
                'model_name': ensemble_model_name # Use the ensemble model name here
            })
            
        except Exception as e:
            import traceback
            return jsonify({
                'status': 'error',
                'message': f'Error making predictions with stacking ensemble: {str(e)}',
                'details': traceback.format_exc()
            }), 500 