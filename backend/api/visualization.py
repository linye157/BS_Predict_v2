from flask import jsonify
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class VisualizationAPI:
    def __init__(self):
        self.num_targets = 3
        
    def visualize_data(self, viz_type, params):
        """Generate data visualizations"""
        try:
            # Get data from DataProcessingAPI
            from api.data_processing import DataProcessingAPI
            data_api = DataProcessingAPI()
            
            # Get appropriate data
            data_type = params.get('data_type', 'train')
            if data_type == 'train':
                if data_api.train_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No training data available'
                    }), 404
                data = data_api.train_data
            else:
                if data_api.test_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No testing data available'
                    }), 404
                data = data_api.test_data
            
            # Get feature and target columns
            feature_cols = data.columns[:-self.num_targets]
            target_cols = data.columns[-self.num_targets:]
            
            # Generate visualization based on type
            if viz_type == 'distribution':
                return self._generate_distribution_plot(data, params)
            elif viz_type == 'correlation':
                return self._generate_correlation_plot(data, params)
            elif viz_type == 'feature_target':
                return self._generate_feature_target_plot(data, params, feature_cols, target_cols)
            elif viz_type == 'dimensionality_reduction':
                return self._generate_dimension_reduction_plot(data, params, feature_cols, target_cols)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported visualization type: {viz_type}'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error generating visualization: {str(e)}'
            }), 500
    
    def _generate_distribution_plot(self, data, params):
        """Generate distribution plots for selected features"""
        # Get selected features
        selected_features = params.get('features', data.columns[:-self.num_targets][:5].tolist())
        plot_type = params.get('plot_type', 'histogram')
        
        if not selected_features:
            return jsonify({
                'status': 'error',
                'message': 'No features selected for visualization'
            }), 400
        
        # Create plots
        plots = {}
        
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'histogram':
                # Histogram with KDE
                sns.histplot(data[feature].dropna(), kde=True)
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
            
            elif plot_type == 'box':
                # Box plot
                sns.boxplot(y=data[feature].dropna())
                plt.title(f'Box Plot of {feature}')
                plt.ylabel(feature)
            
            elif plot_type == 'violin':
                # Violin plot
                sns.violinplot(y=data[feature].dropna())
                plt.title(f'Violin Plot of {feature}')
                plt.ylabel(feature)
                
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            plots[feature] = plot_data
        
        return jsonify({
            'status': 'success',
            'plots': plots
        })
    
    def _generate_correlation_plot(self, data, params):
        """Generate correlation plots"""
        corr_method = params.get('method', 'pearson')
        plot_type = params.get('plot_type', 'heatmap')
        
        # Compute correlation
        corr_matrix = data.corr(method=corr_method)
        
        plt.figure(figsize=(12, 10))
        
        if plot_type == 'heatmap':
            # Heatmap
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, linewidths=.5)
            plt.title(f'{corr_method.capitalize()} Correlation Heatmap')
        
        elif plot_type == 'clustermap':
            # Clustermap
            sns.clustermap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                          standard_scale=1)
            plt.title(f'{corr_method.capitalize()} Correlation Clustermap')
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'corr_data': corr_matrix.to_dict()
        })
    
    def _generate_feature_target_plot(self, data, params, feature_cols, target_cols):
        """Generate feature-target relationship plots"""
        selected_features = params.get('features', feature_cols[:3].tolist())
        selected_target = params.get('target', target_cols[0])
        plot_type = params.get('plot_type', 'scatter')
        
        if not selected_features:
            return jsonify({
                'status': 'error',
                'message': 'No features selected for visualization'
            }), 400
        
        # Create plots
        plots = {}
        
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'scatter':
                # Scatter plot
                plt.scatter(data[feature], data[selected_target], alpha=0.5)
                plt.title(f'{feature} vs {selected_target}')
                plt.xlabel(feature)
                plt.ylabel(selected_target)
                
                # Add trend line
                try:
                    z = np.polyfit(data[feature], data[selected_target], 1)
                    p = np.poly1d(z)
                    plt.plot(data[feature], p(data[feature]), "r--")
                except:
                    pass  # Skip trend line if it can't be computed
            
            elif plot_type == 'joint':
                # Create temporary figure for joint plot
                # We have to capture a seaborn plot which doesn't use the same matplotlib object
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                    g = sns.jointplot(x=feature, y=selected_target, data=data, kind='reg')
                    g.savefig(temp_file.name)
                    plt.close()
                    
                    # Read the saved plot
                    with open(temp_file.name, 'rb') as f:
                        plot_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    plots[feature] = plot_data
                    continue  # Skip the regular plot saving below
            
            # Convert plot to base64 string (for regular matplotlib plots)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            plots[feature] = plot_data
        
        return jsonify({
            'status': 'success',
            'plots': plots,
            'target': selected_target
        })
    
    def _generate_dimension_reduction_plot(self, data, params, feature_cols, target_cols):
        """Generate dimensionality reduction plots (PCA, t-SNE)"""
        method = params.get('method', 'pca')
        target_idx = params.get('target_idx', 0)
        
        # Get feature data
        X = data[feature_cols].values
        target = data[target_cols[target_idx]].values
        
        # Standardize features
        X_std = StandardScaler().fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        
        if method == 'pca':
            # PCA to 2 dimensions
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_std)
            
            # Plot
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap='viridis', alpha=0.7)
            plt.colorbar(label=target_cols[target_idx])
            plt.title(f'PCA Projection colored by {target_cols[target_idx]}')
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            
            # Additional information to return
            additional_info = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'feature_importance': pca.components_.tolist(),
                'feature_names': feature_cols.tolist()
            }
            
        elif method == 'tsne':
            # t-SNE to 2 dimensions
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_std)
            
            # Plot
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, cmap='viridis', alpha=0.7)
            plt.colorbar(label=target_cols[target_idx])
            plt.title(f't-SNE Projection colored by {target_cols[target_idx]}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # No additional information for t-SNE
            additional_info = {}
            
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported dimensionality reduction method: {method}'
            }), 400
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'method': method,
            'target': target_cols[target_idx],
            'additional_info': additional_info
        })
    
    def visualize_model(self, model_name, viz_type):
        """Generate model visualizations"""
        try:
            # Get models from MachineLearningAPI
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Check if model exists
            if not ml_api.model_results or model_name not in ml_api.model_results:
                return jsonify({
                    'status': 'error',
                    'message': f'Model {model_name} not found or no results available'
                }), 404
            
            # Get model results
            model_result = ml_api.model_results[model_name]
            
            # Generate visualization based on type
            if viz_type == 'feature_importance':
                return self._generate_feature_importance_plot(model_result)
            elif viz_type == 'learning_curve':
                return self._generate_learning_curve(model_name, model_result, ml_api)
            elif viz_type == 'prediction_error':
                return self._generate_prediction_error_plot(model_name, model_result, ml_api)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported model visualization type: {viz_type}'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error generating model visualization: {str(e)}'
            }), 500
    
    def _generate_feature_importance_plot(self, model_result):
        """Generate feature importance plot for a model"""
        feature_importance = model_result.get('feature_importance')
        feature_names = model_result.get('feature_names')
        
        if feature_importance is None or feature_names is None:
            return jsonify({
                'status': 'error',
                'message': 'No feature importance data available for this model'
            }), 404
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 20 features (or all if less than 20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'importance_data': importance_df.to_dict('records')
        })
    
    def _generate_learning_curve(self, model_name, model_result, ml_api):
        """Generate learning curve for a model"""
        # Learning curve requires retraining the model with different dataset sizes
        from sklearn.model_selection import learning_curve
        
        # Get target index
        target_idx = model_result.get('target_idx', 0)
        
        # Check if model is available
        if model_name not in ml_api.models:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_name} not available for generating learning curve'
            }), 404
        
        model = ml_api.models[model_name]
        
        # Get data
        X = ml_api.X_train
        y = ml_api.y_train.iloc[:, target_idx]
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Convert to RMSE (positive values)
        train_scores_rmse = np.sqrt(-train_scores)
        test_scores_rmse = np.sqrt(-test_scores)
        
        # Calculate mean and std
        train_mean = np.mean(train_scores_rmse, axis=1)
        train_std = np.std(train_scores_rmse, axis=1)
        test_mean = np.mean(test_scores_rmse, axis=1)
        test_std = np.std(test_scores_rmse, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training error')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation error')
        
        # Add error bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.xlabel('Training set size')
        plt.ylabel('RMSE')
        plt.title(f'Learning Curve for {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Prepare curve data
        curve_data = {
            'train_sizes': train_sizes.tolist(),
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'test_mean': test_mean.tolist(),
            'test_std': test_std.tolist()
        }
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'curve_data': curve_data
        })
    
    def _generate_prediction_error_plot(self, model_name, model_result, ml_api):
        """Generate prediction error plot for a model"""
        # Get target index
        target_idx = model_result.get('target_idx', 0)
        
        # Check if model is available
        if model_name not in ml_api.models:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_name} not available for generating prediction error plot'
            }), 404
        
        model = ml_api.models[model_name]
        
        # Get data
        X_test = ml_api.X_test
        y_test = ml_api.y_test.iloc[:, target_idx]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate errors
        errors = y_test - y_pred
        
        # Create error distribution plot
        plt.figure(figsize=(12, 6))
        
        # Plot error distribution
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        
        # Plot prediction vs actual
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Prepare error data
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2)))
        }
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'error_stats': error_stats
        })
    
    def visualize_results(self, model_names, viz_type):
        """Generate visualizations for comparing multiple models"""
        try:
            # Get models from MachineLearningAPI
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Verify models exist
            valid_models = []
            for model_name in model_names:
                if model_name in ml_api.model_results:
                    valid_models.append(model_name)
            
            if not valid_models:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid models found for comparison'
                }), 404
            
            # Generate visualization based on type
            if viz_type == 'metric_comparison':
                return self._generate_metric_comparison(valid_models, ml_api.model_results)
            elif viz_type == 'prediction_comparison':
                return self._generate_prediction_comparison(valid_models, ml_api)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported comparison visualization type: {viz_type}'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error generating results visualization: {str(e)}'
            }), 500
    
    def _generate_metric_comparison(self, model_names, model_results):
        """Generate comparison of metrics across models"""
        # Get metrics for each model
        comparison_data = []
        
        for model_name in model_names:
            result = model_results[model_name]
            comparison_data.append({
                'model_name': model_name,
                'train_rmse': result.get('train_rmse'),
                'test_rmse': result.get('test_rmse'),
                'train_r2': result.get('train_r2'),
                'test_r2': result.get('test_r2'),
                'train_mae': result.get('train_mae'),
                'test_mae': result.get('test_mae'),
                'training_time': result.get('training_time'),
                'target_idx': result.get('target_idx')
            })
        
        # Create DataFrame
        comp_df = pd.DataFrame(comparison_data)
        
        # Group by target_idx
        target_groups = comp_df.groupby('target_idx')
        
        # Generate plots for each target
        plots = {}
        
        for target_idx, group in target_groups:
            # Sort by test_rmse
            group_sorted = group.sort_values('test_rmse')
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Test RMSE comparison
            plt.subplot(2, 1, 1)
            sns.barplot(x='test_rmse', y='model_name', data=group_sorted)
            plt.title(f'Test RMSE Comparison - Target {target_idx}')
            plt.xlabel('RMSE (lower is better)')
            
            # Test R² comparison
            plt.subplot(2, 1, 2)
            sns.barplot(x='test_r2', y='model_name', data=group_sorted)
            plt.title(f'Test R² Comparison - Target {target_idx}')
            plt.xlabel('R² (higher is better)')
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            plots[f'target_{target_idx}'] = plot_data
        
        return jsonify({
            'status': 'success',
            'plots': plots,
            'comparison_data': comparison_data
        })
    
    def _generate_prediction_comparison(self, model_names, ml_api):
        """Generate comparison of predictions across models"""
        # Group models by target index
        target_groups = {}
        
        for model_name in model_names:
            result = ml_api.model_results[model_name]
            target_idx = result.get('target_idx', 0)
            
            if target_idx not in target_groups:
                target_groups[target_idx] = []
                
            target_groups[target_idx].append({
                'model_name': model_name,
                'model': ml_api.models.get(model_name)
            })
        
        # Generate plots for each target
        plots = {}
        
        for target_idx, models in target_groups.items():
            # Skip if no models available
            if not all(m['model'] is not None for m in models):
                continue
                
            # Get data
            X_test = ml_api.X_test
            y_test = ml_api.y_test.iloc[:, target_idx]
            
            # Make predictions for each model
            predictions = {}
            for model_info in models:
                predictions[model_info['model_name']] = model_info['model'].predict(X_test)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Scatter plot of predictions vs actual
            plt.subplot(1, 1, 1)
            
            # Plot actual vs predicted for each model
            for model_name, preds in predictions.items():
                plt.scatter(y_test, preds, alpha=0.6, label=model_name)
            
            # Add perfect prediction line
            min_val = y_test.min()
            max_val = y_test.max()
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect predictions')
            
            plt.title(f'Actual vs Predicted Comparison - Target {target_idx}')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.legend()
            plt.grid(True)
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            plots[f'target_{target_idx}'] = plot_data
        
        return jsonify({
            'status': 'success',
            'plots': plots
        }) 