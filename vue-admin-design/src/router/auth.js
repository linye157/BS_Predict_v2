import router from './index'
import { getToken } from '../utils/cookie'

// 白名单列表
const whiteList = []

router.beforeEach((to, from, next) => {
  // Since login page has been removed, we'll just let all routes pass through
  next()
})
