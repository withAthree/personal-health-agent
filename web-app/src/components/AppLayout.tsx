import { HeartPulse, LayoutDashboard, Loader2, LogOut, MessageSquare, Plus } from 'lucide-react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router'

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
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { useAuth } from '@/contexts/auth-context'
import { useChatSession } from '@/contexts/chat-session-context'
import { cn } from '@/lib/utils'

const appTitle = import.meta.env.VITE_APP_TITLE ?? 'Personal Health Agent'

/** 会话列表骨架屏的行数；只用于首屏占位，不做真实分页。 */
const SKELETON_ROWS = 5

/**
 * 登录后的整体框架：左侧边栏（品牌 + 对话记录列表 + 用户昵称）与右侧主内容区。
 *
 * 侧边栏由 `ChatSessionProvider` 提供当前会话与会话列表；点击列表项即切换会话，
 * 对话页读同一个 `sessionId` 把对应的账本贴回来。
 */
export function AppLayout() {
  const { auth, signOut } = useAuth()
  const { sessionId, openSession, startNewChat, sessions, sessionsLoading } = useChatSession()
  const navigate = useNavigate()
  const location = useLocation()
  const nickname = auth?.user.nickname ?? '未登录'

  /** 会话相关操作都落在对话页，从别的页面点进来要先切回去。 */
  function goToChat(): void {
    if (location.pathname !== '/chat') {
      void navigate('/chat')
    }
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-background text-foreground">
      <aside className="flex w-72 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground">
        {/* 品牌行。标题取自环境变量，这里按设计稿只显示缩写，全名挂在 title 上 */}
        <div className="flex h-14 shrink-0 items-center gap-2 border-b px-3">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground">
            <HeartPulse className="size-4" aria-hidden="true" />
          </span>
          <span className="truncate text-sm font-semibold tracking-tight" title={appTitle}>
            PHA
          </span>
          <NavLink
            to="/dashboard"
            className="ml-auto"
            aria-label="健康看板"
            title="健康看板"
          >
            {({ isActive }) => (
              <span
                className={cn(
                  'grid size-8 place-items-center rounded-lg transition-colors',
                  isActive
                    ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                    : 'text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
                )}
              >
                <LayoutDashboard className="size-4" aria-hidden="true" />
              </span>
            )}
          </NavLink>
        </div>

        {/* 对话记录列表 */}
        <div className="flex min-h-0 flex-1 flex-col gap-1 px-3 py-3">
          <div className="flex shrink-0 items-center px-1 pb-1">
            <span className="text-xs font-medium text-muted-foreground">对话记录</span>
            <Button
              variant="ghost"
              size="icon-sm"
              className="ml-auto"
              onClick={() => {
                startNewChat()
                goToChat()
              }}
              aria-label="新对话"
              title="新对话"
            >
              <Plus />
            </Button>
          </div>

          {/* min-h-0 不能省：Radix 的 ScrollArea root 自身 overflow 仍是 visible，
              在 column flex 容器里 min-height:auto 会解析成内容高度，把侧边栏撑破 */}
          <ScrollArea className="min-h-0 flex-1">
            <nav className="flex flex-col gap-0.5 pr-1" aria-label="对话记录">
              {sessionsLoading ? (
                <div className="flex flex-col gap-1 px-2 py-1">
                  {Array.from({ length: SKELETON_ROWS }, (_, index) => (
                    <Skeleton key={index} className="h-7 w-full" />
                  ))}
                  <span className="sr-only">正在读取对话记录…</span>
                </div>
              ) : sessions.length === 0 ? (
                <div className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-muted-foreground">
                  <Loader2 className="size-3.5 shrink-0" aria-hidden="true" />
                  还没有对话记录
                </div>
              ) : (
                sessions.map((session) => {
                  const active = session.id === sessionId
                  const title = session.title ?? '新对话'
                  return (
                    <button
                      key={session.id}
                      type="button"
                      onClick={() => {
                        openSession(session.id)
                        goToChat()
                      }}
                      title={title}
                      aria-current={active ? 'true' : undefined}
                      className={cn(
                        'flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition-colors',
                        active
                          ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
                          : 'text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
                      )}
                    >
                      <MessageSquare className="size-3.5 shrink-0 opacity-70" aria-hidden="true" />
                      <span className="truncate">{title}</span>
                    </button>
                  )
                })
              )}
            </nav>
          </ScrollArea>
        </div>

        {/* 用户昵称 */}
        <div className="shrink-0 border-t p-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="h-auto w-full justify-start gap-2 px-2 py-1.5"
                aria-label="账号菜单"
              >
                <Avatar className="size-7">
                  <AvatarFallback>{nickname.slice(0, 1)}</AvatarFallback>
                </Avatar>
                <span className="min-w-0 flex-1 truncate text-left text-sm">{nickname}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="w-56">
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
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </div>
        <footer className="shrink-0 border-t px-4 py-2 text-[11px] text-muted-foreground">
          健康助手提供的是一般性健康信息，不能替代医生的诊断与处方
        </footer>
      </main>
    </div>
  )
}
