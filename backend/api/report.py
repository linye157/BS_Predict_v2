from flask import jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime

# For generating reports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import xlsxwriter

class ReportAPI:
    def __init__(self):
        self.report_folder = Path('static/reports')
        
        # Ensure report folder exists
        os.makedirs(self.report_folder, exist_ok=True)
    
    def generate_report(self, report_type, model_names):
        """Generate a report in the specified format"""
        try:
            # Get models from MachineLearningAPI
            from api.machine_learning import MachineLearningAPI
            ml_api = MachineLearningAPI()
            
            # Check if models exist
            valid_models = []
            for model_name in model_names:
                if model_name in ml_api.model_results:
                    valid_models.append({
                        'name': model_name,
                        'results': ml_api.model_results[model_name]
                    })
            
            if not valid_models:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid models found for report generation'
                }), 404
            
            # Get data from DataProcessingAPI
            from api.data_processing import DataProcessingAPI
            data_api = DataProcessingAPI()
            
            # Generate report based on type
            if report_type == 'pdf':
                return self._generate_pdf_report(valid_models, ml_api, data_api)
            elif report_type == 'excel':
                return self._generate_excel_report(valid_models, ml_api, data_api)
            elif report_type == 'json':
                return self._generate_json_report(valid_models, ml_api, data_api)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported report type: {report_type}'
                }), 400
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error generating report: {str(e)}'
            }), 500
    
    def _generate_pdf_report(self, models, ml_api, data_api):
        """Generate a PDF report"""
        # Create a file-like buffer to receive PDF data
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Build the story (content)
        story = []
        
        # Title
        title = Paragraph(f"Machine Learning Model Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Data summary
        if data_api.train_data is not None:
            summary_text = f"Training data: {data_api.train_data.shape[0]} rows, {data_api.train_data.shape[1]} columns"
            story.append(Paragraph(summary_text, styles['Normal']))
            
        if data_api.test_data is not None:
            summary_text = f"Testing data: {data_api.test_data.shape[0]} rows, {data_api.test_data.shape[1]} columns"
            story.append(Paragraph(summary_text, styles['Normal']))
            
        story.append(Spacer(1, 12))
        
        # Model summaries
        story.append(Paragraph("Model Summaries", styles['Heading2']))
        
        for model_info in models:
            model_name = model_info['name']
            results = model_info['results']
            
            story.append(Paragraph(f"Model: {model_name}", styles['Heading3']))
            
            # Model metrics
            metrics_data = [
                ['Metric', 'Training', 'Testing'],
                ['RMSE', f"{results.get('train_rmse', 'N/A'):.4f}", f"{results.get('test_rmse', 'N/A'):.4f}"],
                ['R²', f"{results.get('train_r2', 'N/A'):.4f}", f"{results.get('test_r2', 'N/A'):.4f}"],
                ['MAE', f"{results.get('train_mae', 'N/A'):.4f}", f"{results.get('test_mae', 'N/A'):.4f}"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 12))
            
            # Model parameters
            params = results.get('params', {})
            if params:
                story.append(Paragraph("Model Parameters:", styles['Normal']))
                
                params_data = [['Parameter', 'Value']]
                for param, value in params.items():
                    params_data.append([param, str(value)])
                
                params_table = Table(params_data)
                params_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(params_table)
                story.append(Spacer(1, 12))
            
            # Feature importance if available
            if results.get('feature_importance') and results.get('feature_names'):
                # Create feature importance plot
                plt.figure(figsize=(8, 6))
                importance = results['feature_importance']
                feature_names = results['feature_names']
                
                # Create DataFrame for sorting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                
                # Sort by importance and take top 15
                importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                
                # Plot
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f'Top 15 Feature Importance - {model_name}')
                plt.tight_layout()
                
                # Save plot to buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                plt.close()
                
                # Add image to report
                story.append(Paragraph("Feature Importance:", styles['Normal']))
                story.append(Image(img_buffer, width=450, height=300))
                story.append(Spacer(1, 12))
        
        # Model comparison
        if len(models) > 1:
            story.append(Paragraph("Model Comparison", styles['Heading2']))
            
            # Create comparison table
            comp_data = [['Model', 'Test RMSE', 'Test R²', 'Test MAE', 'Training Time (s)']]
            
            for model_info in models:
                results = model_info['results']
                comp_data.append([
                    model_info['name'],
                    f"{results.get('test_rmse', 'N/A'):.4f}",
                    f"{results.get('test_r2', 'N/A'):.4f}",
                    f"{results.get('test_mae', 'N/A'):.4f}",
                    f"{results.get('training_time', 'N/A'):.2f}"
                ])
            
            comp_table = Table(comp_data)
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(comp_table)
            story.append(Spacer(1, 12))
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            
            model_names = [m['name'] for m in models]
            test_rmses = [m['results'].get('test_rmse', 0) for m in models]
            
            # Sort by performance
            sorted_idx = np.argsort(test_rmses)
            sorted_model_names = [model_names[i] for i in sorted_idx]
            sorted_test_rmses = [test_rmses[i] for i in sorted_idx]
            
            plt.barh(sorted_model_names, sorted_test_rmses)
            plt.xlabel('Test RMSE (lower is better)')
            plt.title('Model Comparison')
            plt.tight_layout()
            
            # Save plot to buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plt.close()
            
            # Add image to report
            story.append(Image(img_buffer, width=450, height=300))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_report_{timestamp}.pdf"
        
        # Save a copy to disk
        report_path = self.report_folder / filename
        with open(report_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Return the buffer for download
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    
    def _generate_excel_report(self, models, ml_api, data_api):
        """Generate an Excel report"""
        # Create a file-like buffer to receive Excel data
        buffer = io.BytesIO()
        
        # Create a workbook and add worksheets
        workbook = xlsxwriter.Workbook(buffer)
        
        # Add summary worksheet
        summary_sheet = workbook.add_worksheet('Summary')
        
        # Formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4F81BD',
            'font_color': 'white',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'border': 1
        })
        
        # Write summary header
        summary_sheet.write(0, 0, 'Model Report', header_format)
        summary_sheet.write(0, 1, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', header_format)
        
        # Write data summary
        row = 2
        summary_sheet.write(row, 0, 'Data Summary', header_format)
        row += 1
        
        if data_api.train_data is not None:
            summary_sheet.write(row, 0, 'Training data rows:', cell_format)
            summary_sheet.write(row, 1, data_api.train_data.shape[0], cell_format)
            row += 1
            summary_sheet.write(row, 0, 'Training data columns:', cell_format)
            summary_sheet.write(row, 1, data_api.train_data.shape[1], cell_format)
            row += 1
        
        if data_api.test_data is not None:
            summary_sheet.write(row, 0, 'Testing data rows:', cell_format)
            summary_sheet.write(row, 1, data_api.test_data.shape[0], cell_format)
            row += 1
            summary_sheet.write(row, 0, 'Testing data columns:', cell_format)
            summary_sheet.write(row, 1, data_api.test_data.shape[1], cell_format)
            row += 1
        
        # Model comparison table
        row += 2
        summary_sheet.write(row, 0, 'Model Comparison', header_format)
        row += 1
        
        # Headers
        summary_sheet.write(row, 0, 'Model', header_format)
        summary_sheet.write(row, 1, 'Test RMSE', header_format)
        summary_sheet.write(row, 2, 'Test R²', header_format)
        summary_sheet.write(row, 3, 'Test MAE', header_format)
        summary_sheet.write(row, 4, 'Training Time (s)', header_format)
        row += 1
        
        # Data
        for model_info in models:
            results = model_info['results']
            summary_sheet.write(row, 0, model_info['name'], cell_format)
            summary_sheet.write(row, 1, results.get('test_rmse', 'N/A'), cell_format)
            summary_sheet.write(row, 2, results.get('test_r2', 'N/A'), cell_format)
            summary_sheet.write(row, 3, results.get('test_mae', 'N/A'), cell_format)
            summary_sheet.write(row, 4, results.get('training_time', 'N/A'), cell_format)
            row += 1
        
        # Auto-adjust column widths
        summary_sheet.autofit()
        
        # Create detailed sheets for each model
        for model_info in models:
            model_name = model_info['name']
            results = model_info['results']
            
            # Create worksheet for model
            model_sheet = workbook.add_worksheet(model_name[:31])  # Excel worksheet names limited to 31 chars
            
            # Write model header
            model_sheet.write(0, 0, f'Model: {model_name}', header_format)
            model_sheet.write(0, 1, f'Type: {results.get("model_type", "Unknown")}', header_format)
            
            # Write metrics
            row = 2
            model_sheet.write(row, 0, 'Metrics', header_format)
            row += 1
            
            model_sheet.write(row, 0, 'Metric', header_format)
            model_sheet.write(row, 1, 'Training', header_format)
            model_sheet.write(row, 2, 'Testing', header_format)
            row += 1
            
            model_sheet.write(row, 0, 'RMSE', cell_format)
            model_sheet.write(row, 1, results.get('train_rmse', 'N/A'), cell_format)
            model_sheet.write(row, 2, results.get('test_rmse', 'N/A'), cell_format)
            row += 1
            
            model_sheet.write(row, 0, 'R²', cell_format)
            model_sheet.write(row, 1, results.get('train_r2', 'N/A'), cell_format)
            model_sheet.write(row, 2, results.get('test_r2', 'N/A'), cell_format)
            row += 1
            
            model_sheet.write(row, 0, 'MAE', cell_format)
            model_sheet.write(row, 1, results.get('train_mae', 'N/A'), cell_format)
            model_sheet.write(row, 2, results.get('test_mae', 'N/A'), cell_format)
            row += 2
            
            # Write parameters
            params = results.get('params', {})
            if params:
                model_sheet.write(row, 0, 'Parameters', header_format)
                row += 1
                
                model_sheet.write(row, 0, 'Parameter', header_format)
                model_sheet.write(row, 1, 'Value', header_format)
                row += 1
                
                for param, value in params.items():
                    model_sheet.write(row, 0, param, cell_format)
                    model_sheet.write(row, 1, str(value), cell_format)
                    row += 1
                
                row += 1
            
            # Write feature importance if available
            if results.get('feature_importance') and results.get('feature_names'):
                model_sheet.write(row, 0, 'Feature Importance', header_format)
                row += 1
                
                model_sheet.write(row, 0, 'Feature', header_format)
                model_sheet.write(row, 1, 'Importance', header_format)
                row += 1
                
                importance = results['feature_importance']
                feature_names = results['feature_names']
                
                # Create DataFrame for sorting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                for _, row_data in importance_df.iterrows():
                    model_sheet.write(row, 0, row_data['Feature'], cell_format)
                    model_sheet.write(row, 1, row_data['Importance'], cell_format)
                    row += 1
            
            # Auto-adjust column widths
            model_sheet.autofit()
        
        # Close workbook
        workbook.close()
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_report_{timestamp}.xlsx"
        
        # Save a copy to disk
        buffer.seek(0)
        report_path = self.report_folder / filename
        with open(report_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Return the buffer for download
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    def _generate_json_report(self, models, ml_api, data_api):
        """Generate a JSON report"""
        # Create report data
        report_data = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {},
            'models': {}
        }
        
        # Add data summary
        if data_api.train_data is not None:
            report_data['data_summary']['train_data'] = {
                'rows': data_api.train_data.shape[0],
                'columns': data_api.train_data.shape[1]
            }
        
        if data_api.test_data is not None:
            report_data['data_summary']['test_data'] = {
                'rows': data_api.test_data.shape[0],
                'columns': data_api.test_data.shape[1]
            }
        
        # Add model data
        for model_info in models:
            model_name = model_info['name']
            results = model_info['results']
            
            # Convert numpy types to Python types for JSON serialization
            model_data = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    model_data[key] = float(value)
                elif isinstance(value, np.ndarray):
                    model_data[key] = value.tolist()
                else:
                    model_data[key] = value
            
            report_data['models'][model_name] = model_data
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_report_{timestamp}.json"
        
        # Save to disk
        report_path = self.report_folder / filename
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create a buffer for download
        buffer = io.BytesIO()
        buffer.write(json.dumps(report_data, indent=2).encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        ) 