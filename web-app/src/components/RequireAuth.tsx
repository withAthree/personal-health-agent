import type { ReactNode } from 'react'
import { Navigate } from 'react-router'

import { useAuth } from '@/contexts/auth-context'

/**
 * 需要登录才能进入的页面。
 *
 * 未登录（或令牌已过期）时跳登录页，不做「回跳原页面」：
 * 重新登录后统一进对话页。
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { auth } = useAuth()

  if (auth === null) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}
