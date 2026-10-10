import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router'

import { AuthProvider } from '@/contexts/auth-context'
import { routes } from '@/routes'
import '@/index.css'

const container = document.getElementById('root')

if (!container) {
  throw new Error('未找到 #root 挂载点，请检查 index.html')
}

createRoot(container).render(
  <StrictMode>
    <AuthProvider>
      <RouterProvider router={createBrowserRouter(routes)} />
    </AuthProvider>
  </StrictMode>,
)
