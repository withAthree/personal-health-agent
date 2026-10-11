import { useRef, useState } from 'react'
import {
  Ellipsis,
  HeartPulse,
  LayoutDashboard,
  Loader2,
  LogOut,
  MessageSquare,
  Pencil,
  Pin,
  PinOff,
  Plus,
  Trash2,
} from 'lucide-react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
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
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { useAuth } from '@/contexts/auth-context'
import { useChatSession } from '@/contexts/chat-session-context'
import { SESSION_TITLE_MAX_LEN, type ChatSessionSummary } from '@/lib/api'
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
  const {
    sessionId,
    openSession,
    startNewChat,
    sessions,
    sessionsLoading,
    renameSession,
    pinSession,
    removeSession,
  } = useChatSession()
  const navigate = useNavigate()
  const location = useLocation()
  const nickname = auth?.user.nickname ?? '未登录'

  // 正在改名的会话与它的草稿；一次只改一个
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null)
  // 本次改名是否已经收尾（提交过或取消过）。Esc 与提交之后还可能补一次 blur，
  // 用这个标记挡住它：否则刚取消就被提交回去，或者提交完再发一遍同样的请求
  const renameSettled = useRef(false)
  // 菜单里的动作有的会把焦点交给别处（改名输入框、删除确认框）。这时不能让 Radix
  // 在菜单关闭时把焦点抢回触发按钮，否则刚挂上去的控件会立刻失焦
  const handoffFocus = useRef(false)
  // 等待确认删除的会话；非 null 时确认框是打开的
  const [pendingDelete, setPendingDelete] = useState<ChatSessionSummary | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  /** 会话相关操作都落在对话页，从别的页面点进来要先切回去。 */
  function goToChat(): void {
    if (location.pathname !== '/chat') {
      void navigate('/chat')
    }
  }

  /**
   * 列表操作的统一入口。
   *
   * 失败时只把原因显示在列表上方：这些操作不是乐观更新，后端没改成功时列表
   * 本来就没变，不需要回滚。
   */
  async function runSessionAction(action: () => Promise<void>): Promise<void> {
    setActionError(null)
    try {
      await action()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : '操作失败，请稍后重试')
    }
  }

  function beginRename(session: ChatSessionSummary): void {
    renameSettled.current = false
    handoffFocus.current = true
    setRenaming({ id: session.id, title: session.title ?? '新对话' })
  }

  function cancelRename(): void {
    renameSettled.current = true
    setRenaming(null)
  }

  /** 提交改名。Enter 与失焦都走这里，重复调用是幂等的（标题相同就不再发请求）。 */
  async function commitRename(): Promise<void> {
    if (renaming === null || renameSettled.current) {
      return
    }
    renameSettled.current = true
    const { id } = renaming
    const title = renaming.title.trim()
    setRenaming(null)
    // 清空或没改就不发请求：后端会拒绝空标题，前端提前挡掉
    if (title === '' || title === (sessions.find((item) => item.id === id)?.title ?? '新对话')) {
      return
    }
    await runSessionAction(() => renameSession(id, title))
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

          {/* 列表操作的失败原因。放在列表上方而不是弹 toast：侧边栏本来就是
              常驻的，用户点完按钮目光还在这里 */}
          {actionError !== null && (
            <p role="alert" className="shrink-0 px-1 pb-1 text-xs text-destructive">
              {actionError}
            </p>
          )}

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
                  if (renaming?.id === session.id) {
                    return (
                      <Input
                        key={session.id}
                        autoFocus
                        value={renaming.title}
                        maxLength={SESSION_TITLE_MAX_LEN}
                        aria-label="会话标题"
                        onChange={(event) => {
                          setRenaming({ id: session.id, title: event.target.value })
                        }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            event.preventDefault()
                            void commitRename()
                          } else if (event.key === 'Escape') {
                            event.preventDefault()
                            cancelRename()
                          }
                        }}
                        onBlur={() => {
                          void commitRename()
                        }}
                        className="h-7 px-2 text-sm"
                      />
                    )
                  }
                  return (
                    // 会话行是「一个按钮 + 一个悬浮操作按钮」，按钮不能嵌套，
                    // 所以外面包一层相对定位的容器
                    <div key={session.id} className="group/session relative">
                      <button
                        type="button"
                        onClick={() => {
                          openSession(session.id)
                          goToChat()
                        }}
                        title={title}
                        aria-current={active ? 'true' : undefined}
                        className={cn(
                          'flex w-full items-center gap-2 rounded-lg py-1.5 pr-7 pl-2 text-left text-sm transition-colors',
                          active
                            ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
                            : 'text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
                        )}
                      >
                        <MessageSquare
                          className="size-3.5 shrink-0 opacity-70"
                          aria-hidden="true"
                        />
                        <span className="truncate">{title}</span>
                        {session.is_pinned && (
                          <>
                            <Pin className="size-3 shrink-0 opacity-60" aria-hidden="true" />
                            <span className="sr-only">已置顶</span>
                          </>
                        )}
                      </button>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-xs"
                            aria-label={`${title} 的会话操作`}
                            title="会话操作"
                            className="absolute top-1/2 right-0.5 -translate-y-1/2 opacity-0 group-hover/session:opacity-100 focus-visible:opacity-100 data-open:opacity-100"
                          >
                            <Ellipsis />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="w-36"
                          onCloseAutoFocus={(event) => {
                            if (handoffFocus.current) {
                              handoffFocus.current = false
                              event.preventDefault()
                            }
                          }}
                        >
                          <DropdownMenuItem onSelect={() => beginRename(session)}>
                            <Pencil />
                            重命名
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onSelect={() => {
                              void runSessionAction(() =>
                                pinSession(session.id, !session.is_pinned),
                              )
                            }}
                          >
                            {session.is_pinned ? <PinOff /> : <Pin />}
                            {session.is_pinned ? '取消置顶' : '置顶'}
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            variant="destructive"
                            onSelect={() => {
                              handoffFocus.current = true
                              setPendingDelete(session)
                            }}
                          >
                            <Trash2 />
                            删除
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
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

        {/* 删除确认。删会话会连消息一起带走，所以必须过一道确认 */}
        <AlertDialog
          open={pendingDelete !== null}
          onOpenChange={(open) => {
            if (!open) {
              setPendingDelete(null)
            }
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>删除这个对话？</AlertDialogTitle>
              <AlertDialogDescription>
                对话「{pendingDelete?.title ?? '新对话'}」里的消息会全部删除，无法恢复。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                onClick={() => {
                  const target = pendingDelete
                  setPendingDelete(null)
                  if (target !== null) {
                    void runSessionAction(() => removeSession(target.id))
                  }
                }}
              >
                删除
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
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
