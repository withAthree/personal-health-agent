/**
 * 登录态的 React 入口。
 *
 * 状态本身存在 `@/lib/auth` 的模块级 store 里（localStorage + 过期判断），
 * 因为 api 层在 React 之外也需要读令牌；这里用 Context 把它交给组件树，
 * 组件只依赖 `useAuth()`，不必关心底层实现。
 */
import { createContext, useContext, useEffect, useMemo, useSyncExternalStore } from 'react'
import type { ReactNode } from 'react'

import { fetchMe, logout, type AuthTokenResponse } from '@/lib/api'
import { getAuthSnapshot, saveAuth, subscribeAuth, type AuthState } from '@/lib/auth'

interface AuthContextValue {
  /** 未登录时为 null。 */
  auth: AuthState | null
  /** 保存登录结果（注册成功也走这里，后端会直接签发令牌）。 */
  signIn: (result: AuthTokenResponse) => void
  /** 退出登录：吊销服务端令牌并清掉本地登录态，不会抛错。 */
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const auth = useSyncExternalStore(subscribeAuth, getAuthSnapshot)
  const token = auth?.token ?? null

  // 启动时校验一次令牌：已被服务端吊销或用户已删除时，api 层收到 401 会清掉登录态，
  // 界面随之回到登录页。网络异常则保持当前状态，不误登出。
  useEffect(() => {
    if (token === null) {
      return
    }
    fetchMe().catch(() => {
      // 401 已在 http 层处理；其余错误（如后端未启动）不打扰用户
    })
  }, [token])

  const value = useMemo<AuthContextValue>(
    () => ({
      auth,
      signIn: (result) => {
        saveAuth({
          token: result.access_token,
          expiresAt: result.expires_at,
          user: result.user,
        })
      },
      signOut: async () => {
        try {
          await logout()
        } catch {
          // 服务端吊销失败也要退出：本地登录态已由 api 层清掉
        }
      },
    }),
    [auth],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (value === null) {
    throw new Error('useAuth 必须在 <AuthProvider> 内部使用')
  }
  return value
}
