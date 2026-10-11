/**
 * 「当前会话 + 会话列表」的 React 入口。
 *
 * 侧边栏（`AppLayout`）要展示并切换会话，对话页（`ChatPage`）要读写当前会话，
 * 两者共用同一份状态，所以放到 Context 里，而不是留在对话页的局部 state。
 *
 * 职责边界与 `@/lib/session` 一致：这里只记「用哪个会话」，历史消息永远从
 * 后端账本按需拉取，不在浏览器里存第二份对话内容。
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

import { fetchSessions, type ChatSessionSummary } from '@/lib/api'
import { clearLastSession, readLastSession, saveLastSession } from '@/lib/session'

interface ChatSessionContextValue {
  /** 当前会话 ID；null 表示还没落库的新对话。 */
  sessionId: string | null
  /** 打开一个已有会话（会话列表点击）。 */
  openSession: (id: string) => void
  /** 开始新对话：清空当前会话记忆。 */
  startNewChat: () => void
  /**
   * 「新对话」计数，每次 `startNewChat` 自增。
   *
   * 对话页靠它清空界面：`sessionId` 本来就是 null 时（第一条消息还没拿到
   * `session` 事件）它不会变化，只靠 `sessionId` 无法触发清理。
   */
  newChatToken: number
  /** 会话列表，最近更新的在前。 */
  sessions: ChatSessionSummary[]
  /** 会话列表是否正在拉取。 */
  sessionsLoading: boolean
  /** 重新拉取会话列表（新建会话、一轮结束、标题变化后调用）。 */
  refreshSessions: () => Promise<void>
}

const ChatSessionContext = createContext<ChatSessionContextValue | null>(null)

/** 拉一次会话列表；失败时按「没有记录」处理（列表只是导航入口，不该打断对话）。 */
async function loadSessions(): Promise<ChatSessionSummary[]> {
  try {
    return await fetchSessions()
  } catch {
    return []
  }
}

export function ChatSessionProvider({ children }: { children: ReactNode }) {
  // 初始值直接读本地记忆：刷新页面（或 48 小时内重新打开）时回到上次的会话
  const [sessionId, setSessionIdState] = useState<string | null>(() => readLastSession())
  const [newChatToken, setNewChatToken] = useState(0)
  // null 表示还没拉到过：首屏据此显示骨架屏，而不另开一个 loading 状态
  const [sessions, setSessions] = useState<ChatSessionSummary[] | null>(null)

  const openSession = useCallback((id: string) => {
    setSessionIdState(id)
    saveLastSession(id)
  }, [])

  const startNewChat = useCallback(() => {
    setSessionIdState(null)
    clearLastSession()
    setNewChatToken((token) => token + 1)
  }, [])

  const refreshSessions = useCallback(async () => {
    setSessions(await loadSessions())
  }, [])

  // 首屏拉一次。setSessions 放在异步回调里，而不是 effect 体里直接调：后者会同步
  // 触发一轮额外渲染（react-hooks 的 set-state-in-effect 就是拦这个的）
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const list = await loadSessions()
      if (!cancelled) {
        setSessions(list)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const value = useMemo<ChatSessionContextValue>(
    () => ({
      sessionId,
      openSession,
      startNewChat,
      newChatToken,
      sessions: sessions ?? [],
      sessionsLoading: sessions === null,
      refreshSessions,
    }),
    [sessionId, openSession, startNewChat, newChatToken, sessions, refreshSessions],
  )

  return <ChatSessionContext.Provider value={value}>{children}</ChatSessionContext.Provider>
}

export function useChatSession(): ChatSessionContextValue {
  const value = useContext(ChatSessionContext)
  if (value === null) {
    throw new Error('useChatSession 必须在 <ChatSessionProvider> 内部使用')
  }
  return value
}
