import type { ReactNode } from 'react'
import { Navigate } from 'react-router'

import { useAuth } from '@/contexts/auth-context'

/** 只给未登录用户看的页面（登录页）；已登录时直接进对话页。 */
export function GuestOnly({ children }: { children: ReactNode }) {
  const { auth } = useAuth()

  if (auth !== null) {
    return <Navigate to="/chat" replace />
  }

  return <>{children}</>
}
