import axios from 'axios'

// Base URL for API calls
const API_BASE_URL = 'http://localhost:5000/api'

// Data Processing API
export function loadDefaultData() {
  return axios.get(`${API_BASE_URL}/data/load_default`)
}

export function uploadData(file, type) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('type', type)
  
  return axios.post(`${API_BASE_URL}/data/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export function downloadData(type, format) {
  return axios.get(`${API_BASE_URL}/data/download`, {
    params: { type, format },
    responseType: 'blob'
  })
}

export function getDataPreview(type) {
  return axios.get(`${API_BASE_URL}/data/preview`, {
    params: { type }
  })
}

export function preprocessData(options, data) {
  return axios.post(`${API_BASE_URL}/data/preprocess`, {
    options,
    ...data
  })
}

// Machine Learning API
export function prepareData(data) {
  return axios.post(`${API_BASE_URL}/ml/prepare`, data)
}

export function trainModel(modelType, targetIdx, params) {
  return axios.post(`${API_BASE_URL}/ml/train`, {
    model_type: modelType,
    target_idx: targetIdx,
    params
  })
}

export function compareModels() {
  return axios.get(`${API_BASE_URL}/ml/compare`)
}

export function predictModel(modelName, data) {
  return axios.post(`${API_BASE_URL}/ml/predict`, {
    model_name: modelName,
    data
  })
}

// Stacking Ensemble API
export function trainStackingEnsemble(baseModels, metaModel, targetIdx) {
  return axios.post(`${API_BASE_URL}/stacking/train`, {
    base_models: baseModels,
    meta_model: metaModel,
    target_idx: targetIdx
  })
}

// Auto ML API
export function trainAutoML(targetIdx, timeLimit) {
  return axios.post(`${API_BASE_URL}/automl/train`, {
    target_idx: targetIdx,
    time_limit: timeLimit
  })
}

// Visualization API
export function visualizeData(vizType, params) {
  return axios.post(`${API_BASE_URL}/visualization/data`, {
    viz_type: vizType,
    params
  })
}

export function visualizeModel(modelName, vizType) {
  return axios.post(`${API_BASE_URL}/visualization/model`, {
    model_name: modelName,
    viz_type: vizType
  })
}

export function visualizeResults(modelNames, vizType) {
  return axios.post(`${API_BASE_URL}/visualization/results`, {
    model_names: modelNames,
    viz_type: vizType
  })
}

// Report API
export function generateReport(reportType, modelNames) {
  return axios.post(`${API_BASE_URL}/report/generate`, {
    report_type: reportType,
    model_names: modelNames
  }, {
    responseType: 'blob'
  })
} 