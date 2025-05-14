<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>数据加载</span>
      </div>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-button type="primary" @click="loadDefaultData" :loading="loadingDefault">
            加载默认数据
          </el-button>
          <span v-if="defaultDataLoaded" style="margin-left: 15px; color: #67C23A;">
            <i class="el-icon-success"></i> 默认数据已加载
          </span>
        </el-col>
        <el-col :span="12" align="right">
          <el-button type="success" @click="clearAllData">
            清除所有数据
          </el-button>
        </el-col>
      </el-row>

      <el-divider content-position="left">上传数据</el-divider>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-upload
            class="upload-demo"
            action="#"
            :auto-upload="false"
            :on-change="handleTrainFileChange"
            :file-list="trainFileList"
            :limit="1">
            <el-button slot="trigger" size="small" type="primary">选择训练数据</el-button>
            <el-button style="margin-left: 10px;" size="small" type="success" @click="uploadTrainFile" :loading="uploadingTrain">上传到服务器</el-button>
            <div slot="tip" class="el-upload__tip">支持 xlsx 或 csv 文件（训练数据）</div>
          </el-upload>
        </el-col>
        <el-col :span="12">
          <el-upload
            class="upload-demo"
            action="#"
            :auto-upload="false"
            :on-change="handleTestFileChange"
            :file-list="testFileList"
            :limit="1">
            <el-button slot="trigger" size="small" type="primary">选择测试数据</el-button>
            <el-button style="margin-left: 10px;" size="small" type="success" @click="uploadTestFile" :loading="uploadingTest">上传到服务器</el-button>
            <div slot="tip" class="el-upload__tip">支持 xlsx 或 csv 文件（测试数据）</div>
          </el-upload>
        </el-col>
      </el-row>

      <el-divider content-position="left">下载数据</el-divider>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-button type="primary" @click="downloadData('train', 'csv')" :disabled="!dataLoaded">下载训练数据 (CSV)</el-button>
          <el-button type="primary" @click="downloadData('train', 'excel')" :disabled="!dataLoaded">下载训练数据 (Excel)</el-button>
        </el-col>
        <el-col :span="12">
          <el-button type="primary" @click="downloadData('test', 'csv')" :disabled="!dataLoaded">下载测试数据 (CSV)</el-button>
          <el-button type="primary" @click="downloadData('test', 'excel')" :disabled="!dataLoaded">下载测试数据 (Excel)</el-button>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="box-card" style="margin-top: 20px">
      <div slot="header" class="clearfix">
        <span>数据预览</span>
      </div>
      <el-tabs v-model="activeTab" @tab-click="handleTabClick">
        <el-tab-pane label="训练数据" name="train">
          <div v-if="trainDataPreview">
            <el-row>
              <el-col :span="24">
                <div class="data-info">
                  <p><strong>形状：</strong> {{ trainDataPreview.shape[0] }} 行, {{ trainDataPreview.shape[1] }} 列</p>
                </div>
              </el-col>
            </el-row>
            <el-table
              :data="trainDataPreview.head"
              style="width: 100%"
              border
              max-height="400"
              v-if="trainDataPreview.head && trainDataPreview.head.length > 0">
              <el-table-column
                v-for="column in Object.keys(trainDataPreview.head[0]).slice(0, 10)"
                :key="column"
                :prop="column"
                :label="column"
                min-width="120">
              </el-table-column>
              <el-table-column
                v-if="Object.keys(trainDataPreview.head[0]).length > 10"
                label="更多..."
                min-width="80"
                fixed="right">
                <template slot-scope="scope">
                  <el-button type="text" @click="showMoreColumns(scope.row)">查看更多</el-button>
                </template>
              </el-table-column>
            </el-table>
            <div v-else class="empty-data">
              <i class="el-icon-warning"></i>
              <p>无训练数据可预览</p>
            </div>
          </div>
          <div v-else class="loading-data">
            <p>请先加载训练数据</p>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="测试数据" name="test">
          <div v-if="testDataPreview">
            <el-row>
              <el-col :span="24">
                <div class="data-info">
                  <p><strong>形状：</strong> {{ testDataPreview.shape[0] }} 行, {{ testDataPreview.shape[1] }} 列</p>
                </div>
              </el-col>
            </el-row>
            <el-table
              :data="testDataPreview.head"
              style="width: 100%"
              border
              max-height="400"
              v-if="testDataPreview.head && testDataPreview.head.length > 0">
              <el-table-column
                v-for="column in Object.keys(testDataPreview.head[0]).slice(0, 10)"
                :key="column"
                :prop="column"
                :label="column"
                min-width="120">
              </el-table-column>
              <el-table-column
                v-if="Object.keys(testDataPreview.head[0]).length > 10"
                label="更多..."
                min-width="80"
                fixed="right">
                <template slot-scope="scope">
                  <el-button type="text" @click="showMoreColumns(scope.row)">查看更多</el-button>
                </template>
              </el-table-column>
            </el-table>
            <div v-else class="empty-data">
              <i class="el-icon-warning"></i>
              <p>无测试数据可预览</p>
            </div>
          </div>
          <div v-else class="loading-data">
            <p>请先加载测试数据</p>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-card class="box-card" style="margin-top: 20px">
      <div slot="header" class="clearfix">
        <span>数据预处理</span>
      </div>
      <el-form :model="preprocessForm" label-width="120px">
        <el-form-item label="预处理选项">
          <el-checkbox-group v-model="preprocessForm.options">
            <el-checkbox label="fill_missing_values">填充缺失值</el-checkbox>
            <el-checkbox label="standardize">特征标准化</el-checkbox>
            <el-checkbox label="normalize">特征归一化</el-checkbox>
            <el-checkbox label="handle_outliers">异常值处理</el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item label="应用于" v-if="preprocessForm.options.length > 0">
          <el-checkbox-group v-model="preprocessForm.targets">
            <el-checkbox label="train">训练数据</el-checkbox>
            <el-checkbox label="test">测试数据</el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item label="填充方法" v-if="preprocessForm.options.includes('fill_missing_values')">
          <el-select v-model="preprocessForm.fill_method" placeholder="选择填充方法">
            <el-option label="均值填充" value="mean"></el-option>
            <el-option label="中位数填充" value="median"></el-option>
            <el-option label="众数填充" value="mode"></el-option>
            <el-option label="固定值填充" value="fixed"></el-option>
          </el-select>
          <el-input-number 
            v-if="preprocessForm.fill_method === 'fixed'" 
            v-model="preprocessForm.fixed_value" 
            :precision="2" 
            :step="0.1" 
            placeholder="填充值" 
            style="margin-left: 10px;">
          </el-input-number>
        </el-form-item>

        <el-form-item label="异常值处理" v-if="preprocessForm.options.includes('handle_outliers')">
          <el-select v-model="preprocessForm.outlier_method" placeholder="选择异常值处理方法">
            <el-option label="剪裁" value="clip"></el-option>
            <el-option label="删除" value="remove"></el-option>
          </el-select>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="applyPreprocessing" :loading="preprocessing" :disabled="!dataLoaded || preprocessForm.options.length === 0 || preprocessForm.targets.length === 0">
            应用预处理
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- More Columns Dialog -->
    <el-dialog title="所有列数据" :visible.sync="dialogVisible" width="80%">
      <el-table :data="[dialogRow]" border>
        <el-table-column
          v-for="column in dialogColumns"
          :key="column"
          :prop="column"
          :label="column"
          min-width="120">
        </el-table-column>
      </el-table>
    </el-dialog>
  </div>
</template>

<script>
import { 
  loadDefaultData, 
  uploadData, 
  downloadData, 
  getDataPreview, 
  preprocessData 
} from '@/api/ml-api'

export default {
  name: 'DataProcessing',
  data() {
    return {
      // Loading states
      loadingDefault: false,
      uploadingTrain: false,
      uploadingTest: false,
      preprocessing: false,
      
      // Data states
      defaultDataLoaded: false,
      dataLoaded: false,
      
      // File upload
      trainFileList: [],
      testFileList: [],
      
      // Data preview
      activeTab: 'train',
      trainDataPreview: null,
      testDataPreview: null,
      
      // Preprocessing form
      preprocessForm: {
        options: [],
        targets: ['train', 'test'],
        fill_method: 'mean',
        fixed_value: 0,
        outlier_method: 'clip'
      },
      
      // Dialog for more columns
      dialogVisible: false,
      dialogRow: {},
      dialogColumns: []
    }
  },
  mounted() {
    // Check if data is already loaded
    this.checkDataLoaded()
  },
  methods: {
    // Load default data
    loadDefaultData() {
      this.loadingDefault = true
      loadDefaultData()
        .then(response => {
          if (response.data.status === 'success') {
            this.$message.success('默认数据加载成功')
            this.defaultDataLoaded = true
            this.dataLoaded = true
            this.fetchDataPreview()
          } else {
            this.$message.error(response.data.message || '加载失败')
          }
        })
        .catch(error => {
          this.$message.error('加载默认数据时出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.loadingDefault = false
        })
    },
    
    // Clear all data
    clearAllData() {
      this.$confirm('确定要清除所有已加载的数据吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.defaultDataLoaded = false
        this.dataLoaded = false
        this.trainDataPreview = null
        this.testDataPreview = null
        this.trainFileList = []
        this.testFileList = []
        this.$message.success('数据已清除')
      }).catch(() => {})
    },
    
    // Handle file selection
    handleTrainFileChange(file, fileList) {
      this.trainFileList = fileList.slice(-1)
    },
    
    handleTestFileChange(file, fileList) {
      this.testFileList = fileList.slice(-1)
    },
    
    // Upload files
    uploadTrainFile() {
      if (this.trainFileList.length === 0) {
        this.$message.warning('请先选择训练数据文件')
        return
      }
      
      this.uploadingTrain = true
      uploadData(this.trainFileList[0].raw, 'train')
        .then(response => {
          if (response.data.status === 'success') {
            this.$message.success('训练数据上传成功')
            this.dataLoaded = true
            this.fetchDataPreview('train')
          } else {
            this.$message.error(response.data.message || '上传失败')
          }
        })
        .catch(error => {
          this.$message.error('上传训练数据时出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.uploadingTrain = false
        })
    },
    
    uploadTestFile() {
      if (this.testFileList.length === 0) {
        this.$message.warning('请先选择测试数据文件')
        return
      }
      
      this.uploadingTest = true
      uploadData(this.testFileList[0].raw, 'test')
        .then(response => {
          if (response.data.status === 'success') {
            this.$message.success('测试数据上传成功')
            this.dataLoaded = true
            this.fetchDataPreview('test')
          } else {
            this.$message.error(response.data.message || '上传失败')
          }
        })
        .catch(error => {
          this.$message.error('上传测试数据时出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.uploadingTest = false
        })
    },
    
    // Download data
    downloadData(type, format) {
      downloadData(type, format)
        .then(response => {
          const blob = new Blob([response.data])
          const link = document.createElement('a')
          const fileName = `${type}_data.${format === 'excel' ? 'xlsx' : 'csv'}`
          
          link.href = URL.createObjectURL(blob)
          link.download = fileName
          link.click()
          URL.revokeObjectURL(link.href)
          
          this.$message.success(`${type === 'train' ? '训练' : '测试'}数据下载成功`)
        })
        .catch(error => {
          this.$message.error(`下载${type === 'train' ? '训练' : '测试'}数据失败: ` + (error.response?.data?.message || error.message))
        })
    },
    
    // Data preview
    checkDataLoaded() {
      this.fetchDataPreview()
    },
    
    fetchDataPreview(specificType = null) {
      const types = specificType ? [specificType] : ['train', 'test']
      
      types.forEach(type => {
        getDataPreview(type)
          .then(response => {
            if (response.data.status === 'success') {
              if (type === 'train') {
                this.trainDataPreview = response.data.data
              } else {
                this.testDataPreview = response.data.data
              }
              this.dataLoaded = true
            }
          })
          .catch(error => {
            console.error(`获取${type}数据预览失败:`, error)
            // Don't show error message for preview - it's expected to fail if no data is loaded
          })
      })
    },
    
    handleTabClick(tab) {
      this.activeTab = tab.name
    },
    
    showMoreColumns(row) {
      this.dialogRow = row
      this.dialogColumns = Object.keys(row)
      this.dialogVisible = true
    },
    
    // Preprocessing
    applyPreprocessing() {
      if (!this.dataLoaded) {
        this.$message.warning('请先加载数据')
        return
      }
      
      if (this.preprocessForm.options.length === 0) {
        this.$message.warning('请选择至少一个预处理选项')
        return
      }
      
      if (this.preprocessForm.targets.length === 0) {
        this.$message.warning('请选择要应用预处理的数据集')
        return
      }
      
      this.preprocessing = true
      
      const data = {
        options: this.preprocessForm.options,
        targets: this.preprocessForm.targets
      }
      
      // Add additional parameters based on selected options
      if (this.preprocessForm.options.includes('fill_missing_values')) {
        data.fill_method = this.preprocessForm.fill_method
        if (this.preprocessForm.fill_method === 'fixed') {
          data.fixed_value = this.preprocessForm.fixed_value
        }
      }
      
      if (this.preprocessForm.options.includes('handle_outliers')) {
        data.outlier_method = this.preprocessForm.outlier_method
      }
      
      preprocessData(this.preprocessForm.options, data)
        .then(response => {
          if (response.data.status === 'success') {
            this.$message.success('预处理应用成功')
            this.fetchDataPreview()  // Refresh data preview
          } else {
            this.$message.error(response.data.message || '预处理失败')
          }
        })
        .catch(error => {
          this.$message.error('应用预处理时出错: ' + (error.response?.data?.message || error.message))
        })
        .finally(() => {
          this.preprocessing = false
        })
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

.data-info {
  margin-bottom: 15px;
  background-color: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
}

.empty-data {
  text-align: center;
  padding: 30px;
  color: #909399;
}

.empty-data i {
  font-size: 48px;
  margin-bottom: 10px;
}

.loading-data {
  text-align: center;
  padding: 30px;
  color: #909399;
}

.el-divider {
  margin: 20px 0;
}
</style> 