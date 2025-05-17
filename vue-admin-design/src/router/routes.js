import Layout from '../layout'

/**
 * hidden: true                  如果设置为 true，该项菜单将不会显示在菜单栏中(默认为 false)
 * meta : {
    title: 'title'               菜单名
    icon: 'icon-name'            图标名
    fixed: true                  如果设置为 true，该项 tag 将一直存在 tag 栏中(默认为 false)
  }
 * */

export const asyncRoutes = [
  {
    path: '/ml-system',
    name: 'MLSystem',
    component: Layout,
    redirect: '/ml-system/data-processing',
    meta: {
      title: '机器学习子系统',
      icon: 'el-icon-s-platform'
    },
    children: [
      {
        path: 'data-processing',
        name: 'DataProcessing',
        component: () => import('../views/ml-system/DataProcessing'),
        meta: {
          title: '数据处理',
        }
      },
      {
        path: 'machine-learning',
        name: 'MachineLearning',
        component: () => import('../views/ml-system/MachineLearning'),
        meta: {
          title: '机器学习',
        }
      },
      {
        path: 'stacking-ensemble',
        name: 'StackingEnsemble',
        component: () => import('../views/ml-system/StackingEnsemble'),
        meta: {
          title: '机器学习Stacking集成',
        }
      },
      {
        path: 'auto-ml',
        name: 'AutoML',
        component: () => import('../views/ml-system/AutoML'),
        meta: {
          title: '自动化机器学习',
        }
      }
    ]
  },
  {
    path: '/user-interaction',
    name: 'UserInteraction',
    component: Layout,
    redirect: '/user-interaction/visualization',
    meta: {
      title: '用户交互子系统',
      icon: 'el-icon-s-data'
    },
    children: [
      {
        path: 'visualization',
        name: 'Visualization',
        component: () => import('../views/user-interaction/Visualization'),
        meta: {
          title: '可视化分析',
        }
      },
      {
        path: 'report',
        name: 'Report',
        component: () => import('../views/user-interaction/Report'),
        meta: {
          title: '报表',
        }
      }
    ]
  }
]
