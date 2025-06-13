import request from '../request'

export function login(data) {
  return request({
    url: '/user/login',
    method: 'post',
    data
  })
}

export function getUserInfo(params) {
  return request({
    url: '/user/info',
    method: 'get',
    params
  })
}

export function download(params) {
  return request({
    url: '/download',
    method: 'get',
    params
  })
}