import { Navigate, type RouteObject } from 'react-router'

import { AppLayout } from '@/components/AppLayout'
import { GuestOnly } from '@/components/GuestOnly'
import { RequireAuth } from '@/components/RequireAuth'
import { ChatPage } from '@/pages/ChatPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { LoginPage } from '@/pages/LoginPage'

/** 路由表；由 `src/main.tsx` 交给 `createBrowserRouter`。 */
export const routes: RouteObject[] = [
  {
    path: '/login',
    element: (
      <GuestOnly>
        <LoginPage />
      </GuestOnly>
    ),
  },
  {
    path: '/',
    element: (
      <RequireAuth>
        <AppLayout />
      </RequireAuth>
    ),
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: 'chat', element: <ChatPage /> },
      { path: 'dashboard', element: <DashboardPage /> },
      // 未知路径回到对话页
      { path: '*', element: <Navigate to="/chat" replace /> },
    ],
  },
]
