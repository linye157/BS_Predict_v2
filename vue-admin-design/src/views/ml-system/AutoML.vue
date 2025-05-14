<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>自动化机器学习</span>
      </div>
      <div>
        <el-form :model="autoMLForm" label-width="120px">
          <el-form-item label="目标特征索引">
            <el-input-number v-model="autoMLForm.targetIdx" :min="0" :max="2" />
          </el-form-item>
          <el-form-item label="时间限制(秒)">
            <el-input-number v-model="autoMLForm.timeLimit" :min="10" :max="3600" />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="trainAutoML">开始训练</el-button>
          </el-form-item>
        </el-form>

        <div v-if="loading" class="loading-container">
          <el-progress type="circle" :percentage="progress" />
          <p>正在进行自动机器学习训练...</p>
        </div>

        <div v-if="result">
          <h3>训练结果</h3>
          <el-table :data="resultTable" border>
            <el-table-column prop="model" label="最佳模型" />
            <el-table-column prop="r2" label="R²得分" />
            <el-table-column prop="mse" label="均方误差(MSE)" />
            <el-table-column prop="mae" label="平均绝对误差(MAE)" />
          </el-table>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import { trainAutoML } from '@/api/ml-api'

export default {
  name: 'AutoML',
  data() {
    return {
      autoMLForm: {
        targetIdx: 0,
        timeLimit: 60
      },
      loading: false,
      progress: 0,
      result: null,
      resultTable: []
    }
  },
  methods: {
    async trainAutoML() {
      this.loading = true
      this.progress = 0
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 95) {
          this.progress += 5
        }
      }, 1000)
      
      try {
        const response = await trainAutoML(
          this.autoMLForm.targetIdx,
          this.autoMLForm.timeLimit
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.result = response.data
        this.resultTable = [
          {
            model: this.result.best_model,
            r2: this.result.metrics.r2.toFixed(4),
            mse: this.result.metrics.mse.toFixed(4),
            mae: this.result.metrics.mae.toFixed(4)
          }
        ]
      } catch (error) {
        this.$message.error('自动机器学习训练失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
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