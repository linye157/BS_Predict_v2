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
from sklearn.base import clone # 引入 clone
import traceback # 引入 traceback

class StackingEnsembleAPI:
    def __init__(self):
        self.base_models_ensemble = {}  # Changed name to avoid confusion, stores the trained stacking ensemble
        self.meta_model_instance = None # Changed name to avoid confusion, stores the meta model instance
        self.model_folder = Path('models')
        self.num_targets = 3
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trained_base_models = {} 
        self.base_model_results = {} # 新增: 用于存储基础模型的结果和参数

        os.makedirs(self.model_folder, exist_ok=True)

    def _create_model_from_config(self, model_name_prefix, params):
        """Helper to create a model instance with specific parameters."""
        params = params or {} # Ensure params is a dict

        # Common parameters that might be needed and should have defaults if not in params
        common_params = {'random_state': 42}
        
        current_params = {**common_params, **params} # Merge params, specific params override common ones

        if model_name_prefix == 'lr':
            # LinearRegression specific params, remove common ones if not applicable
            lr_params = {k: v for k, v in current_params.items() if k in ['fit_intercept', 'copy_X', 'n_jobs', 'positive']}
            return LinearRegression(**lr_params)
        elif model_name_prefix == 'rf':
            rf_params = {k: v for k, v in current_params.items() if k in ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'ccp_alpha', 'max_samples']}
            rf_params.setdefault('n_jobs', -1) # Ensure n_jobs is set
            return RandomForestRegressor(**rf_params)
        elif model_name_prefix == 'gbr':
            gbr_params = {k: v for k, v in current_params.items() if k in ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease', 'init', 'random_state', 'max_features', 'alpha', 'verbose', 'max_leaf_nodes', 'warm_start', 'validation_fraction', 'n_iter_no_change', 'tol', 'ccp_alpha']}
            return GradientBoostingRegressor(**gbr_params)
        elif model_name_prefix == 'xgbr' or model_name_prefix == 'xgb':
            # XGBRegressor has many parameters, ensure only valid ones are passed
            # This is a simplified example; you might need more robust param handling for XGB
            xgbr_params_keys = ['objective', 'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs']
            xgbr_params = {k: v for k, v in current_params.items() if k in xgbr_params_keys}
            xgbr_params.setdefault('n_jobs', -1)
            xgbr_params.setdefault('objective', 'reg:squarederror') # Common default
            return XGBRegressor(**xgbr_params)
        elif model_name_prefix == 'svr':
            svr_params = {k:v for k,v in current_params.items() if k in ['kernel', 'degree', 'gamma', 'coef0', 'tol', 'C', 'epsilon', 'shrinking', 'cache_size', 'verbose', 'max_iter']}
            return SVR(**svr_params)
        elif model_name_prefix == 'ann':
            ann_params = {k:v for k,v in current_params.items() if k in ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size', 'learning_rate', 'learning_rate_init', 'power_t', 'max_iter', 'shuffle', 'random_state', 'tol', 'verbose', 'warm_start', 'momentum', 'nesterovs_momentum', 'early_stopping', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon', 'n_iter_no_change', 'max_fun']}
            return MLPRegressor(**ann_params)
        else:
            print(f"Warning: Unknown model prefix {model_name_prefix} for param-based instantiation.")
            return None
    
    def _create_meta_model(self, meta_model_type):
        """Create a meta-model instance based on type"""
        # Add specific parameters for meta-models if needed
        if meta_model_type == 'lr':
            return LinearRegression()
        elif meta_model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif meta_model_type == 'gbr':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif meta_model_type == 'xgbr' or meta_model_type == 'xgb':
            return XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror')
        elif meta_model_type == 'svr':
            return SVR(C=1.0, kernel='rbf')
        else:
            print(f"Warning: Unsupported meta-model type: {meta_model_type}. Defaulting to LinearRegression.")
            return LinearRegression()

    def train_stacking_ensemble(self, base_model_names, meta_model_type='lr', target_idx=0):
        try:
            print(f"开始训练Stacking集成，基础模型: {base_model_names}, 元模型: {meta_model_type}, 目标索引: {target_idx}")
            
            if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
                print("错误: 训练/测试数据在StackingEnsembleAPI中不可用")
                return jsonify({'status': 'error', 'message': '训练/测试数据在StackingEnsembleAPI中不可用。请确保数据已正确加载和处理。'}), 400
            
            if not base_model_names:
                print("错误: 未指定基础模型")
                return jsonify({'status': 'error', 'message': '未指定基础模型'}), 400

            y_train_target = self.y_train.iloc[:, target_idx]
            y_test_target = self.y_test.iloc[:, target_idx]

            loaded_base_models_for_stacking = []
            
            print(f"准备加载 {len(base_model_names)} 个基础模型...")
            for model_name_from_frontend in base_model_names:
                print(f"处理模型: {model_name_from_frontend}")
                # 如果前端只传递了模型前缀（如 'rf'），则自动补全完整模型名称
                if not model_name_from_frontend.endswith(f"_target_{target_idx}"):
                    model_full_name = f"{model_name_from_frontend}_target_{target_idx}"
                    print(f"  补全模型名称: {model_full_name}")
                else:
                    model_full_name = model_name_from_frontend
                    print(f"  使用完整模型名称: {model_full_name}")

                if model_full_name in self.trained_base_models:
                    print(f"  从已训练模型字典中加载: {model_full_name}")
                    model_instance = self.trained_base_models[model_full_name]
                    loaded_base_models_for_stacking.append((model_full_name, model_instance))
                else:
                    model_path = self.model_folder / f"{model_full_name}.pkl"
                    print(f"  尝试从文件加载: {model_path}")
                    if model_path.exists():
                        try:
                            with open(model_path, 'rb') as f:
                                model_instance = pickle.load(f)
                            loaded_base_models_for_stacking.append((model_full_name, model_instance))
                            print(f"  成功从文件加载模型")
                        except Exception as load_err:
                            print(f"  从文件加载模型失败: {str(load_err)}")
                            return jsonify({'status': 'error', 'message': f'加载基础模型文件 {model_full_name} 失败: {str(load_err)}'}), 500
                    else:
                        print(f"  错误: 模型文件不存在: {model_path}")
                        return jsonify({'status': 'error', 'message': f'基础模型 {model_full_name} 未找到。请先训练此模型。'}), 404
            
            print(f"成功加载了 {len(loaded_base_models_for_stacking)} 个基础模型")
            if not loaded_base_models_for_stacking:
                 print("错误: 没有可用于堆叠的基础模型")
                 return jsonify({'status': 'error', 'message': '没有可用于堆叠的基础模型'}), 400

            self.meta_model_instance = self._create_meta_model(meta_model_type)
            if self.meta_model_instance is None: # Should be handled by _create_meta_model default
                print(f"错误: 不支持的元模型类型: {meta_model_type}")
                return jsonify({'status': 'error', 'message': f'不支持的元模型类型: {meta_model_type}'}), 400

            print("开始训练Stacking集成模型...")
            start_time = time.time()
            k_folds = 5
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            X_train_meta = np.zeros((self.X_train.shape[0], len(loaded_base_models_for_stacking)))
            
            print(f"使用 {k_folds} 折交叉验证生成元特征...")
            for i, (model_name, fitted_original_model) in enumerate(loaded_base_models_for_stacking):
                print(f"处理基础模型 {i+1}/{len(loaded_base_models_for_stacking)}: {model_name}")
                fold_model_for_cv = None
                
                # Get original params if available
                original_params = {}
                if model_name in self.base_model_results and 'params' in self.base_model_results[model_name]:
                    original_params = self.base_model_results[model_name].get('params', {})
                    print(f"  找到原始参数: {original_params}")

                model_prefix_for_cv = model_name.split('_target_')[0]
                print(f"  创建用于CV的模型实例: {model_prefix_for_cv}")
                fold_model_for_cv = self._create_model_from_config(model_prefix_for_cv, original_params)

                if fold_model_for_cv is None:
                    print(f"  无法通过配置创建模型，尝试克隆原始模型")
                    try:
                        fold_model_for_cv = clone(fitted_original_model)
                        print(f"  成功克隆原始模型")
                    except Exception as e_clone:
                        print(f"  克隆失败: {str(e_clone)}，尝试使用默认参数")
                        fold_model_for_cv = self._create_model_from_config(model_prefix_for_cv, {}) # Try with defaults
                        if fold_model_for_cv is None:
                            print(f"  错误: 无法为k折交叉验证实例化或克隆基础模型 {model_name}")
                            return jsonify({'status': 'error', 'message': f'无法为k折交叉验证实例化或克隆基础模型 {model_name}'}), 400
                
                oof_preds_for_model = np.zeros(self.X_train.shape[0])
                print(f"  开始 {k_folds} 折交叉验证")
                for fold_idx, (train_idx_fold, val_idx_fold) in enumerate(kf.split(self.X_train, y_train_target)):
                    print(f"    处理fold {fold_idx+1}/{k_folds}")
                    X_fold_train, y_fold_train_cv = self.X_train.iloc[train_idx_fold], y_train_target.iloc[train_idx_fold]
                    X_fold_val = self.X_train.iloc[val_idx_fold]
                    
                    try:
                        fold_model_for_cv.fit(X_fold_train, y_fold_train_cv)
                        oof_preds_for_model[val_idx_fold] = fold_model_for_cv.predict(X_fold_val)
                    except Exception as e_fold:
                        print(f"    错误: 训练或预测fold {fold_idx+1}失败: {str(e_fold)}")
                        return jsonify({'status': 'error', 'message': f'交叉验证中模型 {model_name} 在fold {fold_idx+1} 训练或预测失败: {str(e_fold)}'}), 500
                
                X_train_meta[:, i] = oof_preds_for_model
                print(f"  完成基础模型 {i+1} 的元特征生成")

            print("生成测试集元特征...")
            X_test_meta = np.zeros((self.X_test.shape[0], len(loaded_base_models_for_stacking)))
            for i, (model_name, original_model_instance) in enumerate(loaded_base_models_for_stacking):
                print(f"  生成模型 {model_name} 的测试集元特征")
                try:
                    X_test_meta[:, i] = original_model_instance.predict(self.X_test)
                except Exception as e_test:
                    print(f"  错误: 使用模型 {model_name} 生成测试集元特征失败: {str(e_test)}")
                    return jsonify({'status': 'error', 'message': f'使用模型 {model_name} 生成测试集元特征失败: {str(e_test)}'}), 500
            
            print("训练元模型...")
            try:
                self.meta_model_instance.fit(X_train_meta, y_train_target)
            except Exception as e_meta:
                print(f"错误: 元模型训练失败: {str(e_meta)}")
                return jsonify({'status': 'error', 'message': f'元模型训练失败: {str(e_meta)}'}), 500
            
            print("生成训练集和测试集预测...")
            try:
                train_preds = self.meta_model_instance.predict(X_train_meta)
                test_preds = self.meta_model_instance.predict(X_test_meta)
            except Exception as e_pred:
                print(f"错误: 生成预测失败: {str(e_pred)}")
                return jsonify({'status': 'error', 'message': f'生成预测失败: {str(e_pred)}'}), 500
            
            print("计算性能指标...")
            train_rmse = np.sqrt(mean_squared_error(y_train_target, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test_target, test_preds))
            train_r2 = r2_score(y_train_target, train_preds)
            test_r2 = r2_score(y_test_target, test_preds)
            train_mae = mean_absolute_error(y_train_target, train_preds)
            test_mae = mean_absolute_error(y_test_target, test_preds)
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Stacking集成训练完成，用时 {total_time:.2f} 秒")
            
            ensemble_model_name = f"stacking_{meta_model_type}_target_{target_idx}"
            
            stacking_ensemble_details = {
                'base_model_names': [name for name, _ in loaded_base_models_for_stacking],
                'meta_model_instance': self.meta_model_instance,
                'meta_model_type': meta_model_type,
                'target_idx': target_idx # Store target_idx for consistency
            }
            
            print(f"保存集成模型: {ensemble_model_name}")
            model_path = self.model_folder / f"{ensemble_model_name}.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(stacking_ensemble_details, f)
                print(f"模型成功保存到: {model_path}")
            except Exception as e_save:
                print(f"警告: 保存模型失败: {str(e_save)}")
                # 继续执行，但记录警告
            
            self.base_models_ensemble[ensemble_model_name] = stacking_ensemble_details
            
            print("生成可视化图表...")
            try:
                # 实际与预测值对比图
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
                
                # 性能对比数据
                comparison_data = {
                    'model_names': [name for name, _ in loaded_base_models_for_stacking] + [ensemble_model_name],
                    'test_rmse': []
                }
                
                for model_name, model in loaded_base_models_for_stacking:
                    base_preds = model.predict(self.X_test) 
                    base_rmse = np.sqrt(mean_squared_error(y_test_target, base_preds)) 
                    comparison_data['test_rmse'].append(float(base_rmse))
                comparison_data['test_rmse'].append(float(test_rmse))
                
                # 性能对比图
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
            except Exception as e_viz:
                print(f"警告: 生成可视化失败: {str(e_viz)}")
                plot_data = None
                comparison_plot = None
                comparison_data = {
                    'model_names': [name for name, _ in loaded_base_models_for_stacking] + [ensemble_model_name],
                    'test_rmse': []
                }
            
            print("返回训练结果")
            return jsonify({
                'status': 'success',
                'message': f'Stacking集成 {ensemble_model_name} 训练成功',
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
                'meta_model_type': meta_model_type,
                'plot': plot_data,
                'comparison_plot': comparison_plot,
                'comparison_data': comparison_data
            })
            
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Stacking集成训练发生错误: {str(e)}\n{tb_str}")
            return jsonify({
                'status': 'error',
                'message': f'训练Stacking集成时发生错误: {str(e)}',
                'details': tb_str
            }), 500
    
    def predict(self, ensemble_model_name, input_data):
        try:
            if ensemble_model_name not in self.base_models_ensemble:
                model_path = self.model_folder / f"{ensemble_model_name}.pkl"
                if not model_path.exists():
                    return jsonify({'status': 'error', 'message': f'Stacking ensemble model {ensemble_model_name} not found.'}), 404
                with open(model_path, 'rb') as f:
                    self.base_models_ensemble[ensemble_model_name] = pickle.load(f)
            
            ensemble_details = self.base_models_ensemble[ensemble_model_name]
            base_model_names_for_ensemble = ensemble_details['base_model_names']
            meta_model_instance_loaded = ensemble_details['meta_model_instance'] # Renamed for clarity
            
            # Attempt to get feature names from X_train if available, or from a saved configuration
            # This part assumes X_train was available during the session this API instance was active
            # or that feature names are saved/loaded with the model or app context.
            feature_columns = None
            if self.X_train is not None:
                feature_columns = self.X_train.columns.tolist()
            else:
                # Fallback: try to load from a base model if it's a scikit-learn model and has feature_names_in_
                # This is a heuristic and might not always work.
                if self.trained_base_models:
                    first_base_model_name = base_model_names_for_ensemble[0]
                    if first_base_model_name in self.trained_base_models and hasattr(self.trained_base_models[first_base_model_name], 'feature_names_in_'):
                        feature_columns = self.trained_base_models[first_base_model_name].feature_names_in_
                
                if feature_columns is None:
                     return jsonify({'status': 'error', 'message': 'Feature names for input data not determined. Train model in current session or ensure models are loaded with feature context.'}), 400


            try:
                input_df = None
                if isinstance(input_data, list):
                    if input_data and isinstance(input_data[0], dict):
                        input_df = pd.DataFrame(input_data)
                    else:
                        input_df = pd.DataFrame([input_data], columns=feature_columns)
                elif isinstance(input_data, dict):
                    input_df = pd.DataFrame([input_data])
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid input data format'}), 400
                
                missing_cols = set(feature_columns) - set(input_df.columns)
                if missing_cols:
                    return jsonify({'status': 'error', 'message': f'Missing columns: {list(missing_cols)}. Required: {list(feature_columns)}'}), 400
                
                input_df = input_df[feature_columns]
            except Exception as e_input:
                return jsonify({'status': 'error', 'message': f'Error processing input data for prediction: {str(e_input)}', 'details': traceback.format_exc()}), 400
            
            meta_features = np.zeros((input_df.shape[0], len(base_model_names_for_ensemble)))
            
            for i, model_name_for_base in enumerate(base_model_names_for_ensemble):
                base_model_instance = None
                if model_name_for_base in self.trained_base_models:
                    base_model_instance = self.trained_base_models[model_name_for_base]
                else:
                    model_path = self.model_folder / f"{model_name_for_base}.pkl"
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            base_model_instance = pickle.load(f)
                        if self.trained_base_models is None: self.trained_base_models = {}
                        self.trained_base_models[model_name_for_base] = base_model_instance
                    else:
                        return jsonify({'status': 'error', 'message': f'Base model {model_name_for_base} not found.'}), 404
                
                if base_model_instance is None:
                     return jsonify({'status': 'error', 'message': f'Failed to load base model {model_name_for_base}.'}), 500

                meta_features[:, i] = base_model_instance.predict(input_df)
            
            predictions = meta_model_instance_loaded.predict(meta_features)
            predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else [predictions]
            
            return jsonify({
                'status': 'success',
                'predictions': predictions_list,
                'model_name': ensemble_model_name
            })
            
        except Exception as e:
            tb_str = traceback.format_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error making predictions with stacking ensemble: {str(e)}',
                'details': tb_str
            }), 500