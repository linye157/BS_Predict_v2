from flask import jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import json

class DataProcessingAPI:
    def __init__(self):
        self.train_data_path = Path("data/train_data.xlsx")
        self.test_data_path = Path("data/test_data.xlsx")
        self.train_data = None
        self.test_data = None
        self.num_targets = 3  # Default value for number of target columns
    
    def load_default_data(self):
        """Load the default training and testing data"""
        try:
            if self.train_data_path.exists():
                self.train_data = pd.read_excel(self.train_data_path)
                train_shape = self.train_data.shape
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Default train data file not found: {self.train_data_path}'
                }), 404
                
            if self.test_data_path.exists():
                self.test_data = pd.read_excel(self.test_data_path)
                test_shape = self.test_data.shape
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Default test data file not found: {self.test_data_path}'
                }), 404
            
            # Return success with data shapes
            return jsonify({
                'status': 'success',
                'message': 'Default data loaded successfully',
                'data': {
                    'train': {
                        'rows': train_shape[0],
                        'columns': train_shape[1]
                    },
                    'test': {
                        'rows': test_shape[0],
                        'columns': test_shape[1]
                    }
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error loading default data: {str(e)}'
            }), 500
    
    def process_uploaded_file(self, filepath, file_type):
        """Process uploaded file and store in memory"""
        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            else:
                data = pd.read_excel(filepath)
            
            # Store data based on type
            if file_type == 'train':
                self.train_data = data
            else:
                self.test_data = data
            
            # Return success with data shape
            return jsonify({
                'status': 'success',
                'message': f'{file_type.capitalize()} data uploaded successfully',
                'data': {
                    'rows': data.shape[0],
                    'columns': data.shape[1]
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error processing uploaded file: {str(e)}'
            }), 500
    
    def download_data(self, data_type, format_type):
        """Download data in specified format"""
        try:
            # Get the appropriate data
            if data_type == 'train':
                if self.train_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No training data available for download'
                    }), 404
                data = self.train_data
                filename = 'train_data'
            else:
                if self.test_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No testing data available for download'
                    }), 404
                data = self.test_data
                filename = 'test_data'
            
            # Create file in memory
            if format_type == 'csv':
                output = io.StringIO()
                data.to_csv(output, index=False)
                output.seek(0)
                
                # Create a bytes buffer from the string buffer
                bytes_output = io.BytesIO()
                bytes_output.write(output.getvalue().encode('utf-8'))
                bytes_output.seek(0)
                
                return send_file(
                    bytes_output,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'{filename}.csv'
                )
            else:  # Excel
                output = io.BytesIO()
                data.to_excel(output, index=False)
                output.seek(0)
                
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name=f'{filename}.xlsx'
                )
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error downloading data: {str(e)}'
            }), 500
    
    def get_data_preview(self, data_type):
        """Get a preview of the data"""
        try:
            # Get the appropriate data
            if data_type == 'train':
                if self.train_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No training data available'
                    }), 404
                data = self.train_data
            else:
                if self.test_data is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'No testing data available'
                    }), 404
                data = self.test_data
            
            # Generate preview data
            preview = {
                'head': data.head(10).to_dict('records'),
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'missing_values': {col: int(count) for col, count in data.isnull().sum().items() if count > 0},
                'summary': data.describe().to_dict()
            }
            
            return jsonify({
                'status': 'success',
                'data': preview
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error getting data preview: {str(e)}'
            }), 500
    
    def preprocess_data(self, preprocess_options, data):
        """Apply preprocessing to the data"""
        try:
            # Check if we need to preprocess train or test data or both
            data_targets = data.get('targets', ['train', 'test'])
            
            # Process each specified option
            results = {}
            
            for option in preprocess_options:
                if option == 'fill_missing_values':
                    fill_method = data.get('fill_method', 'mean')
                    fixed_value = data.get('fixed_value')
                    
                    for target in data_targets:
                        if target == 'train' and self.train_data is not None:
                            self.train_data = self._fill_missing_values(self.train_data, fill_method, fixed_value)
                            results[f'{target}_missing_filled'] = True
                        elif target == 'test' and self.test_data is not None:
                            self.test_data = self._fill_missing_values(self.test_data, fill_method, fixed_value)
                            results[f'{target}_missing_filled'] = True
                
                elif option == 'standardize':
                    if 'train' in data_targets and 'test' in data_targets and self.train_data is not None and self.test_data is not None:
                        self.train_data, self.test_data = self._standardize_features(self.train_data, self.test_data)
                        results['standardized'] = True
                    else:
                        results['standardized'] = False
                
                elif option == 'normalize':
                    if 'train' in data_targets and 'test' in data_targets and self.train_data is not None and self.test_data is not None:
                        self.train_data, self.test_data = self._normalize_features(self.train_data, self.test_data)
                        results['normalized'] = True
                    else:
                        results['normalized'] = False
                
                elif option == 'handle_outliers':
                    outlier_method = data.get('outlier_method', 'clip')
                    
                    for target in data_targets:
                        if target == 'train' and self.train_data is not None:
                            self.train_data = self._handle_outliers(self.train_data, outlier_method)
                            results[f'{target}_outliers_handled'] = True
                        elif target == 'test' and self.test_data is not None:
                            self.test_data = self._handle_outliers(self.test_data, outlier_method)
                            results[f'{target}_outliers_handled'] = True
            
            return jsonify({
                'status': 'success',
                'message': 'Preprocessing applied successfully',
                'results': results
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error during preprocessing: {str(e)}'
            }), 500
    
    def _fill_missing_values(self, df, method, fixed_value=None):
        """Fill missing values in dataframe"""
        # Create a copy to avoid modifying the original
        df_filled = df.copy()
        
        # Select only numeric columns
        numeric_cols = df_filled.select_dtypes(include=['int64', 'float64']).columns
        
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        elif method == 'fixed':
            if fixed_value is not None:
                imputer = SimpleImputer(strategy='constant', fill_value=fixed_value)
                df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        
        return df_filled
    
    def _standardize_features(self, train_df, test_df):
        """Standardize features (mean=0, std=1)"""
        # Get feature columns (exclude target columns)
        features = train_df.iloc[:, :-self.num_targets].columns
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data and transform both datasets
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        
        return train_df, test_df
    
    def _normalize_features(self, train_df, test_df):
        """Normalize features (min=0, max=1)"""
        # Get feature columns (exclude target columns)
        features = train_df.iloc[:, :-self.num_targets].columns
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Fit on training data and transform both datasets
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        
        return train_df, test_df
    
    def _handle_outliers(self, df, method):
        """Handle outliers in dataframe"""
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Select only numeric columns, excluding target columns
        numeric_cols = df_processed.iloc[:, :-self.num_targets].select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if method == 'clip':
                # Clip values outside of 3 standard deviations
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                df_processed[col] = df_processed[col].clip(lower=mean - 3*std, upper=mean + 3*std)
            
            elif method == 'remove':
                # Calculate z-scores and mark outliers (|z| > 3)
                z_scores = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
                df_processed = df_processed[(z_scores.abs() <= 3)]
        
        return df_processed 