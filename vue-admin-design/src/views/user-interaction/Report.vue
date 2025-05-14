<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>报表生成</span>
      </div>
      
      <el-form :model="reportForm" label-width="120px">
        <el-form-item label="报表类型">
          <el-select v-model="reportForm.reportType">
            <el-option label="PDF 报表" value="pdf"></el-option>
            <el-option label="Excel 报表" value="excel"></el-option>
            <el-option label="JSON 报表" value="json"></el-option>
          </el-select>
        </el-form-item>
        
        <el-form-item label="包含模型">
          <el-checkbox-group v-model="reportForm.modelNames">
            <el-checkbox label="lr">线性回归(LR)</el-checkbox>
            <el-checkbox label="rf">随机森林(RF)</el-checkbox>
            <el-checkbox label="gbr">梯度提升回归(GBR)</el-checkbox>
            <el-checkbox label="xgb">XGBoost回归(XGB)</el-checkbox>
            <el-checkbox label="svr">支持向量回归(SVR)</el-checkbox>
            <el-checkbox label="ann">神经网络(ANN)</el-checkbox>
            <el-checkbox label="stacking">Stacking集成</el-checkbox>
            <el-checkbox label="automl">AutoML</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        
        <el-form-item label="报表内容">
          <el-checkbox-group v-model="reportForm.sections">
            <el-checkbox label="data_summary">数据概要</el-checkbox>
            <el-checkbox label="model_performance">模型性能</el-checkbox>
            <el-checkbox label="feature_importance">特征重要性</el-checkbox>
            <el-checkbox label="predictions">预测结果</el-checkbox>
            <el-checkbox label="visualizations">可视化</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        
        <el-form-item>
          <el-button type="primary" @click="generateReport" :disabled="reportForm.modelNames.length === 0">生成报表</el-button>
        </el-form-item>
      </el-form>
      
      <div v-if="loading" class="loading-container">
        <el-progress :percentage="progress" />
        <p>正在生成报表...</p>
      </div>
      
      <div v-if="reportUrl" class="report-container">
        <h3>报表已生成</h3>
        <p v-if="reportForm.reportType === 'pdf'">
          <el-button type="primary" @click="downloadReport">下载 PDF 报表</el-button>
          <el-button @click="previewReport">预览 PDF</el-button>
        </p>
        <p v-else-if="reportForm.reportType === 'excel'">
          <el-button type="primary" @click="downloadReport">下载 Excel 报表</el-button>
        </p>
        <p v-else-if="reportForm.reportType === 'json'">
          <el-button type="primary" @click="downloadReport">下载 JSON 报表</el-button>
          <el-button @click="viewJsonReport">查看 JSON</el-button>
        </p>
      </div>
      
      <!-- PDF预览对话框 -->
      <el-dialog title="PDF 预览" :visible.sync="pdfPreviewVisible" fullscreen>
        <iframe :src="reportUrl" style="width: 100%; height: calc(100vh - 150px);"></iframe>
      </el-dialog>
      
      <!-- JSON预览对话框 -->
      <el-dialog title="JSON 预览" :visible.sync="jsonPreviewVisible" fullscreen>
        <pre v-if="jsonContent" class="json-content">{{ jsonContent }}</pre>
      </el-dialog>
      
      <!-- 报表历史记录 -->
      <div class="history-container">
        <h3>报表历史记录</h3>
        <el-table :data="reportHistory" border style="width: 100%">
          <el-table-column prop="date" label="生成日期" width="180"></el-table-column>
          <el-table-column prop="type" label="报表类型" width="120"></el-table-column>
          <el-table-column prop="models" label="包含模型"></el-table-column>
          <el-table-column label="操作" width="200">
            <template slot-scope="scope">
              <el-button size="mini" @click="downloadHistoryReport(scope.row)">下载</el-button>
              <el-button size="mini" type="danger" @click="deleteHistoryReport(scope.$index)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>
  </div>
</template>

<script>
import { generateReport } from '@/api/ml-api'

export default {
  name: 'Report',
  data() {
    return {
      reportForm: {
        reportType: 'pdf',
        modelNames: ['lr', 'rf', 'xgb'],
        sections: ['data_summary', 'model_performance', 'predictions', 'visualizations']
      },
      loading: false,
      progress: 0,
      reportUrl: null,
      pdfPreviewVisible: false,
      jsonPreviewVisible: false,
      jsonContent: null,
      reportHistory: [
        {
          date: '2023-05-10 14:30',
          type: 'PDF',
          models: '线性回归, 随机森林, XGBoost'
        },
        {
          date: '2023-05-09 10:15',
          type: 'Excel',
          models: '全部模型'
        },
        {
          date: '2023-05-08 16:45',
          type: 'JSON',
          models: 'Stacking集成, AutoML'
        }
      ]
    }
  },
  methods: {
    async generateReport() {
      if (this.reportForm.modelNames.length === 0) {
        this.$message.warning('请至少选择一个模型')
        return
      }
      
      this.loading = true
      this.progress = 0
      this.reportUrl = null
      
      // 模拟进度
      const interval = setInterval(() => {
        if (this.progress < 90) {
          this.progress += 2
        }
      }, 300)
      
      try {
        const response = await generateReport(
          this.reportForm.reportType,
          this.reportForm.modelNames
        )
        
        clearInterval(interval)
        this.progress = 100
        
        this.reportUrl = response.data.report_url
        
        // 添加到历史记录
        const modelLabels = {
          'lr': '线性回归',
          'rf': '随机森林',
          'gbr': 'GBR',
          'xgb': 'XGBoost',
          'svr': 'SVR',
          'ann': '神经网络',
          'stacking': 'Stacking集成',
          'automl': 'AutoML'
        }
        
        const modelNames = this.reportForm.modelNames.map(m => modelLabels[m] || m).join(', ')
        const now = new Date()
        const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
        
        this.reportHistory.unshift({
          date: dateStr,
          type: this.reportForm.reportType.toUpperCase(),
          models: modelNames
        })
        
        this.$message.success('报表生成成功')
      } catch (error) {
        this.$message.error('报表生成失败')
        console.error(error)
      } finally {
        clearInterval(interval)
        this.loading = false
      }
    },
    
    downloadReport() {
      if (!this.reportUrl) return
      
      const a = document.createElement('a')
      a.href = this.reportUrl
      a.download = `report_${Date.now()}.${this.reportForm.reportType}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    },
    
    previewReport() {
      if (!this.reportUrl) return
      this.pdfPreviewVisible = true
    },
    
    async viewJsonReport() {
      if (!this.reportUrl) return
      
      try {
        const response = await fetch(this.reportUrl)
        const jsonData = await response.json()
        this.jsonContent = JSON.stringify(jsonData, null, 2)
        this.jsonPreviewVisible = true
      } catch (error) {
        this.$message.error('无法加载JSON数据')
        console.error(error)
      }
    },
    
    downloadHistoryReport(report) {
      // 这里实际使用中应该通过API获取历史报表URL
      this.$message.info(`下载报表: ${report.date}`)
    },
    
    deleteHistoryReport(index) {
      this.$confirm('确定要删除此报表记录吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.reportHistory.splice(index, 1)
        this.$message.success('删除成功')
      }).catch(() => {
        // 取消删除
      })
    }
  }
}
</script>

<style scoped>
.loading-container {
  text-align: center;
  margin: 20px 0;
}

.report-container {
  margin: 20px 0;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.history-container {
  margin-top: 30px;
}

.json-content {
  background-color: #f5f5f5;
  padding: 15px;
  border-radius: 4px;
  overflow: auto;
  max-height: calc(100vh - 200px);
  white-space: pre-wrap;
  font-family: monospace;
}
</style> 