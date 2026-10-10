/**
 * 前端登录态：访问令牌、过期时间与用户信息。
 *
 * 存在 localStorage，所以在令牌有效期内（48 小时）刷新页面或重开浏览器都免登录。
 * 状态变化通过订阅通知 React（`useSyncExternalStore`），
 * 这样 api 层在收到 401 时清掉登录态，界面会自动回到登录页。
 */

export interface AuthUser {
  id: string
  phone: string
  nickname: string
}

export interface AuthState {
  token: string
  /** ISO 8601 字符串，与后端 `expires_at` 字段一致。 */
  expiresAt: string
  user: AuthUser
}

const STORAGE_KEY = 'pha.auth'

function isAuthUser(value: unknown): value is AuthUser {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.phone === 'string' &&
    typeof candidate.nickname === 'string'
  )
}

function parseStored(raw: string): AuthState | null {
  try {
    const parsed = JSON.parse(raw) as unknown
    if (typeof parsed !== 'object' || parsed === null) {
      return null
    }
    const candidate = parsed as Record<string, unknown>
    if (
      typeof candidate.token !== 'string' ||
      candidate.token === '' ||
      typeof candidate.expiresAt !== 'string' ||
      !isAuthUser(candidate.user)
    ) {
      return null
    }
    return { token: candidate.token, expiresAt: candidate.expiresAt, user: candidate.user }
  } catch {
    // 存储内容被外部改坏时按未登录处理，不抛给调用方
    return null
  }
}

function isExpired(expiresAt: string): boolean {
  const timestamp = Date.parse(expiresAt)
  // 解析不出时间时按已过期处理，避免拿着坏令牌反复收到 401
  return Number.isNaN(timestamp) || timestamp <= Date.now()
}

function readStored(): AuthState | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw === null) {
      return null
    }
    const state = parseStored(raw)
    if (state === null || isExpired(state.expiresAt)) {
      window.localStorage.removeItem(STORAGE_KEY)
      return null
    }
    return state
  } catch {
    // 隐私模式等场景下 localStorage 可能不可用，退化成「每次都要登录」
    return null
  }
}

let current: AuthState | null = readStored()

const listeners = new Set<() => void>()

function emit(): void {
  for (const listener of listeners) {
    listener()
  }
}

export function subscribeAuth(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** 供 `useSyncExternalStore` 读取；未变化时返回同一引用。 */
export function getAuthSnapshot(): AuthState | null {
  return current
}

/**
 * 取当前令牌。
 *
 * 每次取用都重新判断过期：页面长时间开着时，本地过期也能立刻登出，
 * 而不是等后端返回 401。
 */
export function getToken(): string | null {
  if (current !== null && isExpired(current.expiresAt)) {
    clearAuth()
  }
  return current?.token ?? null
}

export function saveAuth(state: AuthState): void {
  current = state
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // 写不进去（如无痕模式配额限制）时仅保留内存态，本次会话仍然可用
  }
  emit()
}

export function clearAuth(): void {
  if (current === null) {
    return
  }
  current = null
  try {
    window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // 同上，清理失败不影响内存态已被重置
  }
  emit()
}
