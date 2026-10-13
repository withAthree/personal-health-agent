/**
 * HTTP 层：普通 JSON 接口统一走 axios 实例 `http`。
 *
 * 聊天接口的 SSE 无法用 axios 读流式分块，仍在 `@/lib/api` 里用 fetch 手写。
 * 本模块负责三件事：
 * 1. 请求自动带上 Bearer 令牌；
 * 2. 把后端错误统一转成带 `status` / `code` 的 `ApiError`，调用方只需读 `message`；
 * 3. 收到 401 时清掉本地登录态，界面随之回到登录页。
 */
import axios from 'axios'

import { clearAuth, getToken } from '@/lib/auth'

/** 后端返回的业务错误。`code` 是后端给的机器可读错误码，没有时为 `null`。 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string | null

  constructor(status: number, message: string, code: string | null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

/**
 * 业务接口共用的 axios 实例；`/api/v1` 由 Vite dev server 代理到后端。
 *
 * 刻意不设默认 `Content-Type`：axios 1.x 在请求头已是 `application/json` 时会把 FormData
 * 序列化成 JSON（`axios/lib/defaults/index.js` 的 transformRequest），上传接口会因此 422。
 * 不设时 axios 按 payload 推断——普通对象给 `application/json`，FormData 交给浏览器补 boundary。
 */
export const http = axios.create({
  baseURL: '/api/v1',
  timeout: 15_000,
})

http.interceptors.request.use((config) => {
  const token = getToken()
  if (token !== null) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 错误在这里统一转成 ApiError，业务代码 catch 到的永远是这个类型
http.interceptors.response.use(
  (response) => response,
  (error: unknown) => Promise.reject(toApiError(error)),
)

function readDetail(payload: unknown): unknown {
  if (typeof payload === 'object' && payload !== null) {
    return (payload as { detail?: unknown }).detail
  }
  return undefined
}

function detailToMessage(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim() !== '') {
    return detail
  }
  // FastAPI 的请求体验证错误：detail 是 { loc, msg, type } 数组
  if (Array.isArray(detail)) {
    for (const item of detail as readonly unknown[]) {
      if (typeof item === 'object' && item !== null) {
        const { msg } = item as { msg?: unknown }
        if (typeof msg === 'string' && msg !== '') {
          return msg
        }
      }
    }
    return null
  }
  if (typeof detail === 'object' && detail !== null) {
    const { message } = detail as { message?: unknown }
    if (typeof message === 'string' && message !== '') {
      return message
    }
  }
  return null
}

function detailToCode(detail: unknown): string | null {
  if (typeof detail === 'object' && detail !== null) {
    const { code } = detail as { code?: unknown }
    if (typeof code === 'string' && code !== '') {
      return code
    }
  }
  return null
}

function networkMessage(code: string | undefined): string {
  if (code === 'ECONNABORTED' || code === 'ETIMEDOUT') {
    return '请求超时，请稍后重试'
  }
  return '无法连接后端服务，请确认后端已启动'
}

function buildApiError(status: number, payload: unknown): ApiError {
  // 401 说明令牌无效、过期或已被吊销，顺手清掉本地登录态
  if (status === 401) {
    clearAuth()
  }
  const detail = readDetail(payload)
  return new ApiError(
    status,
    detailToMessage(detail) ?? `请求失败（HTTP ${String(status)}）`,
    detailToCode(detail),
  )
}

/** axios 抛出的错误 → `ApiError`。网络层失败没有状态码，统一记为 0。 */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) {
    return error
  }
  if (axios.isAxiosError<unknown>(error)) {
    const response = error.response
    if (response === undefined) {
      return new ApiError(0, networkMessage(error.code), null)
    }
    return buildApiError(response.status, response.data)
  }
  return new ApiError(0, error instanceof Error ? error.message : '请求失败，请稍后重试', null)
}

/** fetch 的失败响应 → `ApiError`（SSE 接口用，axios 不参与）。 */
export async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    // 非 JSON 响应体（如网关的 502 页面）没有可展示的信息
    payload = undefined
  }
  return buildApiError(response.status, payload)
}
