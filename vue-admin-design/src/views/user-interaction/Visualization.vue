<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>可视化分析</span>
      </div>
      
      <el-tabs v-model="activeTab">
        <el-tab-pane label="数据可视化" name="data">
          <el-form :model="dataVizForm" label-width="120px">
            <el-form-item label="可视化类型">
              <el-select v-model="dataVizForm.vizType">
                <el-option label="特征分布图" value="distribution"></el-option>
                <el-option label="相关性热力图" value="correlation"></el-option>
                <el-option label="特征-目标关系图" value="feature_target"></el-option>
                <el-option label="降维分析图" value="dimension_reduction"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="参数设置" v-if="dataVizForm.vizType === 'distribution'">
              <el-select v-model="dataVizForm.params.features" multiple placeholder="选择要显示的特征">
                <el-option 
                  v-for="feature in availableFeatures" 
                  :key="feature.value" 
                  :label="feature.label" 
                  :value="feature.value">
                </el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item v-if="dataVizForm.vizType === 'feature_target'">
              <el-row>
                <el-col :span="12">
                  <el-form-item label="特征">
                    <el-select v-model="dataVizForm.params.feature" placeholder="选择特征">
                      <el-option 
                        v-for="feature in availableFeatures" 
                        :key="feature.value" 
                        :label="feature.label" 
                        :value="feature.value">
                      </el-option>
                    </el-select>
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="目标">
                    <el-select v-model="dataVizForm.params.target" placeholder="选择目标">
                      <el-option label="目标 0" value="0"></el-option>
                      <el-option label="目标 1" value="1"></el-option>
                      <el-option label="目标 2" value="2"></el-option>
                    </el-select>
                  </el-form-item>
                </el-col>
              </el-row>
            </el-form-item>
            
            <el-form-item v-if="dataVizForm.vizType === 'dimension_reduction'">
              <el-select v-model="dataVizForm.params.method" placeholder="降维方法">
                <el-option label="PCA" value="pca"></el-option>
                <el-option label="t-SNE" value="tsne"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" @click="generateDataVisualization">生成可视化</el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="模型可视化" name="model">
          <el-form :model="modelVizForm" label-width="120px">
            <el-form-item label="模型">
              <el-select v-model="modelVizForm.modelName">
                <el-option label="线性回归(LR)" value="lr"></el-option>
                <el-option label="随机森林(RF)" value="rf"></el-option>
                <el-option label="梯度提升回归(GBR)" value="gbr"></el-option>
                <el-option label="XGBoost回归(XGB)" value="xgb"></el-option>
                <el-option label="支持向量回归(SVR)" value="svr"></el-option>
                <el-option label="神经网络(ANN)" value="ann"></el-option>
                <el-option label="Stacking集成" value="stacking"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="可视化类型">
              <el-select v-model="modelVizForm.vizType">
                <el-option label="特征重要性" value="feature_importance"></el-option>
                <el-option label="学习曲线" value="learning_curve"></el-option>
                <el-option label="预测误差图" value="prediction_error"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" @click="generateModelVisualization">生成可视化</el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="结果可视化" name="results">
          <el-form :model="resultsVizForm" label-width="120px">
            <el-form-item label="模型选择">
              <el-checkbox-group v-model="resultsVizForm.modelNames">
                <el-checkbox label="lr">线性回归(LR)</el-checkbox>
                <el-checkbox label="rf">随机森林(RF)</el-checkbox>
                <el-checkbox label="gbr">梯度提升回归(GBR)</el-checkbox>
                <el-checkbox label="xgb">XGBoost回归(XGB)</el-checkbox>
                <el-checkbox label="svr">支持向量回归(SVR)</el-checkbox>
                <el-checkbox label="ann">神经网络(ANN)</el-checkbox>
                <el-checkbox label="stacking">Stacking集成</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
            
            <el-form-item label="可视化类型">
              <el-select v-model="resultsVizForm.vizType">
                <el-option label="指标比较" value="metric_comparison"></el-option>
                <el-option label="预测比较" value="prediction_comparison"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" @click="generateResultsVisualization" :disabled="resultsVizForm.modelNames.length === 0">生成可视化</el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>
      
      <div v-if="loading" class="loading-container">
        <el-progress :percentage="progress" />
        <p>正在生成可视化...</p>
      </div>
      
      <div v-if="visualization" class="visualization-container">
        <img :src="visualization" class="visualization-image" />
        <el-button type="primary" size="small" @click="downloadVisualization">下载图片</el-button>
      </div>
    </el-card>
  </div>
</template>

<script>
import { visualizeData, visualizeModel, visualizeResults } from '@/api/ml-api'

export default {
  name: 'Visualization',
  data() {
    return {
      activeTab: 'data',
      availableFeatures: Array.from({ length: 65 }, (_, i) => ({
        label: `特征 ${i+1}`,
        value: i
      })),
      dataVizForm: {
        vizType: 'distribution',
        params: {
          features: [0, 1, 2, 3, 4],
          feature: 0,
          target: 0,
          method: 'pca'
        }
      },
      modelVizForm: {
        modelName: 'rf',
        vizType: 'feature_importance'
      },
      resultsVizForm: {
        modelNames: ['lr', 'rf', 'xgb'],
        vizType: 'metric_comparison'
      },
      loading: false,
      progress: 0,
      visualization: null
    }
  },
  methods: {
    async generateDataVisualization() {
      this.loading = true
      this.progress = 0
      this.visualization = null
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 95) {
          this.progress += 5
        }
      }, 200)
      
      try {
        const response = await visualizeData(
          this.dataVizForm.vizType,
          this.dataVizForm.params
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.visualization = 'data:image/png;base64,' + response.data.image
      } catch (error) {
        this.$message.error('生成数据可视化失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
    },
    
    async generateModelVisualization() {
      this.loading = true
      this.progress = 0
      this.visualization = null
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 95) {
          this.progress += 5
        }
      }, 200)
      
      try {
        const response = await visualizeModel(
          this.modelVizForm.modelName,
          this.modelVizForm.vizType
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.visualization = 'data:image/png;base64,' + response.data.image
      } catch (error) {
        this.$message.error('生成模型可视化失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
    },
    
    async generateResultsVisualization() {
      if (this.resultsVizForm.modelNames.length === 0) {
        this.$message.warning('请至少选择一个模型')
        return
      }
      
      this.loading = true
      this.progress = 0
      this.visualization = null
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 95) {
          this.progress += 5
        }
      }, 200)
      
      try {
        const response = await visualizeResults(
          this.resultsVizForm.modelNames,
          this.resultsVizForm.vizType
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.visualization = 'data:image/png;base64,' + response.data.image
      } catch (error) {
        this.$message.error('生成结果可视化失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
    },
    
    downloadVisualization() {
      if (!this.visualization) return
      
      const a = document.createElement('a')
      a.href = this.visualization
      a.download = `visualization_${Date.now()}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }
}
</script>

<style scoped>
.loading-container {
  text-align: center;
  margin: 20px 0;
}

.visualization-container {
  margin-top: 20px;
  text-align: center;
}

.visualization-image {
  max-width: 100%;
  max-height: 600px;
  margin-bottom: 10px;
}
</style> 