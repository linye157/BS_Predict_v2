<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>机器学习Stacking集成</span>
      </div>
      <div>
        <el-form :model="stackingForm" label-width="140px">
          <el-form-item label="一级模型">
            <el-checkbox-group v-model="stackingForm.baseModels">
              <el-checkbox label="lr">线性回归(LR)</el-checkbox>
              <el-checkbox label="rf">随机森林(RF)</el-checkbox>
              <el-checkbox label="gbr">梯度提升回归(GBR)</el-checkbox>
              <el-checkbox label="xgb">XGBoost回归(XGB)</el-checkbox>
              <el-checkbox label="svr">支持向量回归(SVR)</el-checkbox>
              <el-checkbox label="ann">神经网络(ANN)</el-checkbox>
            </el-checkbox-group>
          </el-form-item>
          
          <el-form-item label="二级模型">
            <el-select v-model="stackingForm.metaModel">
              <el-option label="线性回归(LR)" value="lr"></el-option>
              <el-option label="随机森林(RF)" value="rf"></el-option>
              <el-option label="梯度提升回归(GBR)" value="gbr"></el-option>
              <el-option label="XGBoost回归(XGB)" value="xgb"></el-option>
              <el-option label="支持向量回归(SVR)" value="svr"></el-option>
            </el-select>
          </el-form-item>
          
          <el-form-item label="目标特征索引">
            <el-input-number v-model="stackingForm.targetIdx" :min="0" :max="2" />
          </el-form-item>
          
          <el-form-item>
            <el-button type="primary" @click="trainStacking" :disabled="stackingForm.baseModels.length < 2">训练Stacking集成</el-button>
          </el-form-item>
        </el-form>

        <div v-if="loading" class="loading-container">
          <el-progress :percentage="progress" />
          <p>正在训练Stacking集成模型...</p>
        </div>

        <div v-if="result">
          <h3>训练结果</h3>
          <el-tabs>
            <el-tab-pane label="总体性能">
              <el-table :data="resultTable" border>
                <el-table-column prop="model" label="模型" />
                <el-table-column prop="r2" label="R²得分" />
                <el-table-column prop="mse" label="均方误差(MSE)" />
                <el-table-column prop="mae" label="平均绝对误差(MAE)" />
              </el-table>
            </el-tab-pane>
            <el-tab-pane label="一级模型性能">
              <el-table :data="baseModelResults" border>
                <el-table-column prop="model" label="模型" />
                <el-table-column prop="r2" label="R²得分" />
                <el-table-column prop="mse" label="均方误差(MSE)" />
                <el-table-column prop="mae" label="平均绝对误差(MAE)" />
              </el-table>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import { trainStackingEnsemble } from '@/api/ml-api'

export default {
  name: 'StackingEnsemble',
  data() {
    return {
      stackingForm: {
        baseModels: ['lr', 'rf', 'xgb'],
        metaModel: 'lr',
        targetIdx: 0
      },
      loading: false,
      progress: 0,
      result: null,
      resultTable: [],
      baseModelResults: []
    }
  },
  methods: {
    async trainStacking() {
      if (this.stackingForm.baseModels.length < 2) {
        this.$message.warning('请至少选择两个一级模型')
        return
      }
      
      this.loading = true
      this.progress = 0
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 95) {
          this.progress += 2
        }
      }, 500)
      
      try {
        const response = await trainStackingEnsemble(
          this.stackingForm.baseModels,
          this.stackingForm.metaModel,
          this.stackingForm.targetIdx
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.result = response.data
        
        // 设置总体结果
        this.resultTable = [
          {
            model: 'Stacking集成',
            r2: this.result.ensemble_metrics.r2.toFixed(4),
            mse: this.result.ensemble_metrics.mse.toFixed(4),
            mae: this.result.ensemble_metrics.mae.toFixed(4)
          }
        ]
        
        // 设置基础模型结果
        this.baseModelResults = Object.keys(this.result.base_models_metrics).map(model => {
          const metrics = this.result.base_models_metrics[model]
          return {
            model: this.getModelName(model),
            r2: metrics.r2.toFixed(4),
            mse: metrics.mse.toFixed(4),
            mae: metrics.mae.toFixed(4)
          }
        })
      } catch (error) {
        this.$message.error('Stacking集成训练失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
    },
    
    getModelName(modelKey) {
      const modelNames = {
        'lr': '线性回归(LR)',
        'rf': '随机森林(RF)',
        'gbr': '梯度提升回归(GBR)',
        'xgb': 'XGBoost回归(XGB)',
        'svr': '支持向量回归(SVR)',
        'ann': '神经网络(ANN)'
      }
      
      return modelNames[modelKey] || modelKey
    }
  }
}
</script>

<style scoped>
.loading-container {
  text-align: center;
  margin: 20px 0;
}
</style> 