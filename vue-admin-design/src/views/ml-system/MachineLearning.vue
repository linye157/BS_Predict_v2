<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>数据准备</span>
      </div>
      <el-form :model="prepareForm" label-width="120px">
        <el-form-item label="测试集比例">
          <el-slider v-model="prepareForm.test_size" :min="0.1" :max="0.5" :step="0.05" :format-tooltip="formatPercentage"></el-slider>
        </el-form-item>
        <el-form-item label="随机种子">
          <el-input-number v-model="prepareForm.random_state" :min="1" :max="1000" :step="1"></el-input-number>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="prepareData" :loading="preparing">准备数据</el-button>
        </el-form-item>
      </el-form>

      <div v-if="dataPrepared" class="data-prepared-info">
        <el-alert title="数据已准备完成" type="success" :closable="false"></el-alert>
        <el-descriptions title="数据分割信息" :column="3" border>
          <el-descriptions-item label="训练集特征">{{ dataInfo.X_train_shape[0] }} 行 × {{ dataInfo.X_train_shape[1] }} 列</el-descriptions-item>
          <el-descriptions-item label="训练集目标">{{ dataInfo.y_train_shape[0] }} 行 × {{ dataInfo.y_train_shape[1] }} 列</el-descriptions-item>
          <el-descriptions-item label="测试集特征">{{ dataInfo.X_test_shape[0] }} 行 × {{ dataInfo.X_test_shape[1] }} 列</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-card>

    <el-card class="box-card" style="margin-top: 20px" v-if="dataPrepared">
      <div slot="header" class="clearfix">
        <span>模型训练</span>
      </div>
      
      <el-tabs v-model="activeModelTab">
        <el-tab-pane label="线性回归 (LR)" name="lr">
          <el-form :model="lrForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="lrForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="拟合截距">
              <el-switch v-model="lrForm.params.fit_intercept"></el-switch>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainModel('lr', lrForm.target_idx, lrForm.params)" :loading="training">
                训练线性回归模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="随机森林 (RF)" name="rf">
          <el-form :model="rfForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="rfForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="树的数量">
              <el-slider v-model="rfForm.params.n_estimators" :min="10" :max="300" :step="10" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最大深度">
              <el-slider v-model="rfForm.params.max_depth" :min="1" :max="50" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最小分裂样本数">
              <el-slider v-model="rfForm.params.min_samples_split" :min="2" :max="20" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最小叶子结点样本数">
              <el-slider v-model="rfForm.params.min_samples_leaf" :min="1" :max="20" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最大特征数">
              <el-select v-model="rfForm.params.max_features">
                <el-option label="sqrt" value="sqrt"></el-option>
                <el-option label="log2" value="log2"></el-option>
                <el-option label="所有特征" value="auto"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainModel('rf', rfForm.target_idx, rfForm.params)" :loading="training">
                训练随机森林模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="梯度提升树 (GBR)" name="gbr">
          <el-form :model="gbrForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="gbrForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="树的数量">
              <el-slider v-model="gbrForm.params.n_estimators" :min="10" :max="300" :step="10" show-input></el-slider>
            </el-form-item>
            <el-form-item label="学习率">
              <el-slider v-model="gbrForm.params.learning_rate" :min="0.01" :max="0.5" :step="0.01" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最大深度">
              <el-slider v-model="gbrForm.params.max_depth" :min="1" :max="20" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最小分裂样本数">
              <el-slider v-model="gbrForm.params.min_samples_split" :min="2" :max="20" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainModel('gbr', gbrForm.target_idx, gbrForm.params)" :loading="training">
                训练梯度提升树模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="XGBoost (XGBR)" name="xgbr">
          <el-form :model="xgbrForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="xgbrForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="树的数量">
              <el-slider v-model="xgbrForm.params.n_estimators" :min="10" :max="300" :step="10" show-input></el-slider>
            </el-form-item>
            <el-form-item label="学习率">
              <el-slider v-model="xgbrForm.params.learning_rate" :min="0.01" :max="0.5" :step="0.01" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最大深度">
              <el-slider v-model="xgbrForm.params.max_depth" :min="1" :max="20" :step="1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="子采样比例">
              <el-slider v-model="xgbrForm.params.subsample" :min="0.5" :max="1" :step="0.05" show-input></el-slider>
            </el-form-item>
            <el-form-item label="特征采样比例">
              <el-slider v-model="xgbrForm.params.colsample_bytree" :min="0.5" :max="1" :step="0.05" show-input></el-slider>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainModel('xgbr', xgbrForm.target_idx, xgbrForm.params)" :loading="training">
                训练XGBoost模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="支持向量机 (SVR)" name="svr">
          <el-form :model="svrForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="svrForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="正则化参数 (C)">
              <el-slider v-model="svrForm.params.C" :min="0.1" :max="10" :step="0.1" show-input></el-slider>
            </el-form-item>
            <el-form-item label="Epsilon">
              <el-slider v-model="svrForm.params.epsilon" :min="0.01" :max="1" :step="0.01" show-input></el-slider>
            </el-form-item>
            <el-form-item label="核函数">
              <el-select v-model="svrForm.params.kernel">
                <el-option label="线性核" value="linear"></el-option>
                <el-option label="多项式核" value="poly"></el-option>
                <el-option label="RBF核" value="rbf"></el-option>
                <el-option label="Sigmoid核" value="sigmoid"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="Gamma">
              <el-select v-model="svrForm.params.gamma">
                <el-option label="scale" value="scale"></el-option>
                <el-option label="auto" value="auto"></el-option>
                <el-option label="0.1" value="0.1"></el-option>
                <el-option label="0.01" value="0.01"></el-option>
                <el-option label="0.001" value="0.001"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainModel('svr', svrForm.target_idx, svrForm.params)" :loading="training">
                训练SVR模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="神经网络 (ANN)" name="ann">
          <el-form :model="annForm" label-width="120px">
            <el-form-item label="目标变量">
              <el-select v-model="annForm.target_idx" placeholder="选择要预测的目标">
                <el-option v-for="(target, index) in targetNames" :key="index" :label="target" :value="index"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="隐藏层结构">
              <el-select v-model="annForm.params.hidden_layer_sizes_option">
                <el-option label="[100]" value="[100]"></el-option>
                <el-option label="[50, 50]" value="[50, 50]"></el-option>
                <el-option label="[100, 50]" value="[100, 50]"></el-option>
                <el-option label="[100, 100, 50]" value="[100, 100, 50]"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="激活函数">
              <el-select v-model="annForm.params.activation">
                <el-option label="ReLU" value="relu"></el-option>
                <el-option label="Tanh" value="tanh"></el-option>
                <el-option label="Sigmoid" value="logistic"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="优化器">
              <el-select v-model="annForm.params.solver">
                <el-option label="Adam" value="adam"></el-option>
                <el-option label="SGD" value="sgd"></el-option>
                <el-option label="LBFGS" value="lbfgs"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="正则化系数 (Alpha)">
              <el-slider v-model="annForm.params.alpha" :min="0.0001" :max="0.01" :step="0.0001" show-input></el-slider>
            </el-form-item>
            <el-form-item label="最大迭代次数">
              <el-slider v-model="annForm.params.max_iter" :min="100" :max="1000" :step="100" show-input></el-slider>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="trainNeuralNetwork" :loading="training">
                训练神经网络模型
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
      </el-tabs>
    </el-card>

    <el-card class="box-card" style="margin-top: 20px" v-if="modelResults">
      <div slot="header" class="clearfix">
        <span>模型结果</span>
      </div>
      
      <el-descriptions title="模型性能指标" :column="3" border>
        <el-descriptions-item label="模型名称">{{ modelResults.model_name }}</el-descriptions-item>
        <el-descriptions-item label="训练RMSE">{{ modelResults.metrics.train_rmse.toFixed(4) }}</el-descriptions-item>
        <el-descriptions-item label="测试RMSE">{{ modelResults.metrics.test_rmse.toFixed(4) }}</el-descriptions-item>
        <el-descriptions-item label="训练R²">{{ modelResults.metrics.train_r2.toFixed(4) }}</el-descriptions-item>
        <el-descriptions-item label="测试R²">{{ modelResults.metrics.test_r2.toFixed(4) }}</el-descriptions-item>
        <el-descriptions-item label="训练时间">{{ modelResults.metrics.training_time.toFixed(2) }}秒</el-descriptions-item>
      </el-descriptions>
      
      <el-row style="margin-top: 20px" v-if="modelResults.plot">
        <el-col :span="24">
          <h4>预测 vs 实际值</h4>
          <div class="plot-container">
            <img :src="'data:image/png;base64,' + modelResults.plot" alt="Prediction Plot" class="prediction-plot">
          </div>
        </el-col>
      </el-row>
      
      <el-row style="margin-top: 20px" v-if="modelResults.feature_importance && modelResults.feature_importance.values">
        <el-col :span="24">
          <h4>特征重要性</h4>
          <el-table
            :data="getFeatureImportance()"
            style="width: 100%"
            border
            max-height="400"
            :default-sort="{prop: 'importance', order: 'descending'}">
            <el-table-column prop="feature" label="特征" width="250"></el-table-column>
            <el-table-column prop="importance" label="重要性" sortable>
              <template slot-scope="scope">
                {{ scope.row.importance.toFixed(4) }}
                <el-progress :percentage="scope.row.percentage" :color="importanceColor(scope.row.percentage)"></el-progress>
              </template>
            </el-table-column>
          </el-table>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="box-card" style="margin-top: 20px" v-if="dataPrepared">
      <div slot="header" class="clearfix">
        <span>模型比较</span>
      </div>
      
      <el-button type="primary" @click="compareModels" :loading="comparing" :disabled="!modelsAvailable">
        比较所有训练的模型
      </el-button>
      
      <div v-if="comparisonResults">
        <div v-for="(plot, key) in comparisonResults.comparison_plots" :key="key" class="plot-container">
          <h4>目标 {{ key.replace('target_', '') }} 模型对比</h4>
          <img :src="'data:image/png;base64,' + plot" alt="Comparison Plot" class="comparison-plot">
        </div>
        
        <div v-for="(targetGroup, idx) in comparisonResults.target_groups" :key="'group'+idx">
          <h4>目标 {{ getTargetName(idx) }} 的模型比较</h4>
          <el-table
            :data="targetGroup"
            style="width: 100%"
            border
            :default-sort="{prop: 'metrics.test_rmse', order: 'ascending'}">
            <el-table-column prop="model_name" label="模型" width="250"></el-table-column>
            <el-table-column prop="metrics.test_rmse" label="测试RMSE" sortable>
              <template slot-scope="scope">
                {{ scope.row.metrics.test_rmse.toFixed(4) }}
              </template>
            </el-table-column>
            <el-table-column prop="metrics.test_r2" label="测试R²" sortable>
              <template slot-scope="scope">
                {{ scope.row.metrics.test_r2.toFixed(4) }}
              </template>
            </el-table-column>
            <el-table-column prop="metrics.training_time" label="训练时间 (秒)" sortable>
              <template slot-scope="scope">
                {{ scope.row.metrics.training_time.toFixed(2) }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import { 
  prepareData,
  trainModel,
  compareModels
} from '@/api/ml-api'

export default {
  name: 'MachineLearning',
  data() {
    return {
      // Forms
      prepareForm: {
        test_size: 0.2,
        random_state: 42
      },
      lrForm: {
        target_idx: 0,
        params: {
          fit_intercept: true
        }
      },
      rfForm: {
        target_idx: 0,
        params: {
          n_estimators: 100,
          max_depth: 10,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: 'sqrt'
        }
      },
      gbrForm: {
        target_idx: 0,
        params: {
          n_estimators: 100,
          learning_rate: 0.1,
          max_depth: 3,
          min_samples_split: 2
        }
      },
      xgbrForm: {
        target_idx: 0,
        params: {
          n_estimators: 100,
          learning_rate: 0.1,
          max_depth: 3,
          subsample: 1.0,
          colsample_bytree: 1.0
        }
      },
      svrForm: {
        target_idx: 0,
        params: {
          C: 1.0,
          epsilon: 0.1,
          kernel: 'rbf',
          gamma: 'scale'
        }
      },
      annForm: {
        target_idx: 0,
        params: {
          hidden_layer_sizes_option: '[100]',
          activation: 'relu',
          solver: 'adam',
          alpha: 0.0001,
          learning_rate: 'constant',
          max_iter: 200
        }
      },
      
      // States
      preparing: false,
      dataPrepared: false,
      training: false,
      comparing: false,
      
      // Data
      dataInfo: null,
      targetNames: [],
      featureNames: [],
      
      // Model tabs
      activeModelTab: 'lr',
      
      // Results
      modelResults: null,
      comparisonResults: null,
      trainedModels: [],
      modelsAvailable: false
    }
  },
  methods: {
    formatPercentage(val) {
      return val * 100 + '%'
    },
    
    // Prepare data for model training
    prepareData() {
      this.preparing = true
      
      prepareData(this.prepareForm)
        .then(response => {
          if (response.data.status === 'success') {
            this.dataPrepared = true
            this.dataInfo = response.data.data
            this.targetNames = response.data.data.target_names
            this.featureNames = response.data.data.feature_names
            this.$message.success('数据准备成功')
          } else {
            this.$message.error(response.data.message || '数据准备失败')
          }
        })
        .catch(error => {
          this.$message.error('数据准备出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.preparing = false
        })
    },
    
    // Train a model
    trainModel(modelType, targetIdx, params) {
      this.training = true
      
      trainModel(modelType, targetIdx, params)
        .then(response => {
          if (response.data.status === 'success') {
            this.modelResults = response.data
            this.$message.success(`${this.getModelTypeName(modelType)}模型训练成功`)
            
            // Add to trained models list if not already in the list
            const modelName = response.data.model_name
            if (!this.trainedModels.includes(modelName)) {
              this.trainedModels.push(modelName)
            }
            this.modelsAvailable = this.trainedModels.length > 0
          } else {
            this.$message.error(response.data.message || '模型训练失败')
          }
        })
        .catch(error => {
          this.$message.error('模型训练出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.training = false
        })
    },
    
    // Train neural network (special case for converting hidden_layer_sizes)
    trainNeuralNetwork() {
      // Parse the hidden layer sizes option
      const hidden_layer_sizes = JSON.parse(this.annForm.params.hidden_layer_sizes_option)
      
      // Create a copy of the params without the option field
      const params = { ...this.annForm.params }
      delete params.hidden_layer_sizes_option
      
      // Add the parsed hidden layer sizes
      params.hidden_layer_sizes = hidden_layer_sizes
      
      // Train the model
      this.trainModel('ann', this.annForm.target_idx, params)
    },
    
    // Compare all trained models
    compareModels() {
      if (!this.modelsAvailable) {
        this.$message.warning('请先训练至少一个模型')
        return
      }
      
      this.comparing = true
      
      compareModels()
        .then(response => {
          if (response.data.status === 'success') {
            this.comparisonResults = response.data
            this.$message.success('模型比较成功')
          } else {
            this.$message.error(response.data.message || '模型比较失败')
          }
        })
        .catch(error => {
          this.$message.error('模型比较出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.comparing = false
        })
    },
    
    // Get feature importance data for table
    getFeatureImportance() {
      if (!this.modelResults || !this.modelResults.feature_importance) {
        return []
      }
      
      const values = this.modelResults.feature_importance.values
      const features = this.modelResults.feature_importance.feature_names
      
      // If no values or features, return empty array
      if (!values || !features) {
        return []
      }
      
      // Create array of feature-importance pairs
      const data = features.map((feature, index) => {
        return {
          feature,
          importance: values[index]
        }
      })
      
      // Sort by importance (descending)
      data.sort((a, b) => b.importance - a.importance)
      
      // Calculate percentage (for progress bar)
      const maxImportance = Math.max(...values)
      data.forEach(item => {
        item.percentage = (item.importance / maxImportance) * 100
      })
      
      return data
    },
    
    // Get color for importance progress bar
    importanceColor(percentage) {
      if (percentage > 70) return '#67C23A'  // Green
      if (percentage > 30) return '#E6A23C'  // Orange
      return '#F56C6C'  // Red
    },
    
    // Get human-readable model type name
    getModelTypeName(modelType) {
      const modelNames = {
        'lr': '线性回归',
        'rf': '随机森林',
        'gbr': '梯度提升树',
        'xgbr': 'XGBoost',
        'svr': '支持向量机',
        'ann': '神经网络'
      }
      return modelNames[modelType] || modelType
    },
    
    // Get target name by index
    getTargetName(index) {
      return this.targetNames[index] || `目标 ${index}`
    }
  }
}
</script>

<style scoped>
.app-container {
  padding: 20px;
}

.box-card {
  margin-bottom: 20px;
}

.data-prepared-info {
  margin-top: 20px;
}

.plot-container {
  margin: 20px 0;
  text-align: center;
}

.prediction-plot,
.comparison-plot {
  max-width: 100%;
  border: 1px solid #EBEEF5;
  border-radius: 4px;
}
</style> 