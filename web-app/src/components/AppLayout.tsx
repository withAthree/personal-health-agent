import { HeartPulse, LayoutDashboard, LogOut, MessageSquare } from 'lucide-react'
import { NavLink, Outlet } from 'react-router'

import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { useAuth } from '@/contexts/auth-context'
import { cn } from '@/lib/utils'

const appTitle = import.meta.env.VITE_APP_TITLE ?? 'Personal Health Agent'

const tabs: readonly { to: string; label: string; icon: typeof MessageSquare }[] = [
  { to: '/chat', label: '对话', icon: MessageSquare },
  { to: '/dashboard', label: '健康看板', icon: LayoutDashboard },
]

/** 登录后的整体框架：顶栏 + 页签导航 + 子路由出口。 */
export function AppLayout() {
  const { auth, signOut } = useAuth()
  const nickname = auth?.user.nickname ?? '未登录'

  return (
    <div className="flex min-h-dvh flex-col bg-muted/30 text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-6 py-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground">
            <HeartPulse className="size-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold tracking-tight">{appTitle}</h1>
            <p className="truncate text-xs text-muted-foreground">
              基于多 Agent 架构的个人健康智能助手
            </p>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="ml-auto gap-2" aria-label="账号菜单">
                <Avatar className="size-6">
                  <AvatarFallback>{nickname.slice(0, 1)}</AvatarFallback>
                </Avatar>
                <span className="hidden max-w-32 truncate sm:inline">{nickname}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuLabel className="truncate font-normal">{nickname}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                onSelect={() => {
                  void signOut()
                }}
              >
                <LogOut />
                退出登录
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <nav className="border-b bg-background">
        <div className="mx-auto flex w-full max-w-7xl gap-1 px-6">
          {tabs.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-1.5 border-b-2 px-3 py-2.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'border-primary text-foreground'
                    : 'border-transparent text-muted-foreground hover:text-foreground',
                )
              }
            >
              <tab.icon className="size-4" aria-hidden="true" />
              {tab.label}
            </NavLink>
          ))}
        </div>
      </nav>

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">
        <Outlet />
      </main>

      <Separator />
      <footer className="mx-auto w-full max-w-7xl px-6 py-3 text-xs text-muted-foreground">
        健康助手提供的是一般性健康信息，不能替代医生的诊断与处方
      </footer>
    </div>
  )
}
