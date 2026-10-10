/**
 * 最近一次会话 ID 的本地缓存。
 *
 * 作用与 `@/lib/auth` 里的令牌一致：刷新页面（或 48 小时内重新打开）时
 * 自动回到上一次的对话，而不是每次都从空白开始。
 *
 * 注意职责边界：这里只记「用哪个会话」，历史消息永远从后端账本拉取，
 * 不在浏览器里存第二份对话内容。
 */

const STORAGE_KEY = 'pha.last_session'

function storage(): Storage | null {
  try {
    return window.localStorage
  } catch {
    // 无痕模式等场景下 localStorage 不可用，退化为「不记忆」
    return null
  }
}

/** 读取上次会话 ID；没有或不可用时返回 null。 */
export function readLastSession(): string | null {
  const store = storage()
  if (store === null) {
    return null
  }
  try {
    const value = store.getItem(STORAGE_KEY)
    return value === null || value === '' ? null : value
  } catch {
    return null
  }
}

/** 记住当前会话 ID。 */
export function saveLastSession(sessionId: string): void {
  const store = storage()
  if (store === null) {
    return
  }
  try {
    store.setItem(STORAGE_KEY, sessionId)
  } catch {
    // 配额满或被禁用时忽略：只是少一次自动恢复
  }
}

/** 清除记忆（点「新对话」时调用）。 */
export function clearLastSession(): void {
  const store = storage()
  if (store === null) {
    return
  }
  try {
    store.removeItem(STORAGE_KEY)
  } catch {
    // 同上
  }
}
