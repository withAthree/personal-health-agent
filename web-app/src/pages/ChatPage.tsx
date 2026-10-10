import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { Bot, Plus, SendHorizontal, Sparkles } from 'lucide-react'
import { useForm, useWatch } from 'react-hook-form'
import { z } from 'zod'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Field, FieldDescription, FieldError, FieldGroup } from '@/components/ui/field'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { MarkdownMessage } from '@/components/MarkdownMessage'
import {
  fetchSessionMessages,
  fetchSessionSummary,
  streamChat,
  type ChatMessage,
} from '@/lib/api'
import { clearLastSession, readLastSession, saveLastSession } from '@/lib/session'
import { cn } from '@/lib/utils'

interface Bubble {
  id: number
  role: 'user' | 'assistant'
  content: string
  agentSource: string | null
  pending: boolean
}

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === 'AbortError'

const composerSchema = z.object({
  message: z.string().trim().min(1, '请输入内容'),
})

type ComposerValues = z.infer<typeof composerSchema>

/** 账本里 role 为 system 的行不参与界面渲染（当前后端也不会返回）。 */
function isVisibleMessage(
  message: ChatMessage,
): message is ChatMessage & { role: 'user' | 'assistant' } {
  return message.role !== 'system'
}

/** Radix 的 ScrollArea 不对外暴露 viewport，只能从 root 里查出来。 */
function findViewport(root: HTMLElement | null): HTMLElement | null {
  return (
    root?.querySelector<HTMLElement>(
      '[data-radix-scroll-area-viewport], [data-slot="scroll-area-viewport"]',
    ) ?? null
  )
}

function BubbleRow({ bubble }: { bubble: Bubble }) {
  const isUser = bubble.role === 'user'

  return (
    <div className={cn('flex gap-2.5', isUser ? 'justify-end' : 'justify-start')}>
      {isUser ? null : (
        <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-secondary text-secondary-foreground">
          <Bot className="size-4" aria-hidden="true" />
        </span>
      )}
      <div className="max-w-[85%] space-y-1">
        <div
          className={cn(
            'rounded-2xl px-4 py-2.5 text-sm',
            isUser
              ? 'rounded-br-sm bg-primary whitespace-pre-wrap text-primary-foreground'
              : 'rounded-bl-sm bg-muted text-foreground',
          )}
        >
          {bubble.pending && bubble.content === '' ? (
            <>
              {/* 气泡底色是 bg-muted，Skeleton 默认的 bg-accent 在同色系上看不出来 */}
              <Skeleton className="h-4 w-24 bg-foreground/15" />
              <span className="sr-only">正在思考…</span>
            </>
          ) : isUser ? (
            bubble.content
          ) : (
            <MarkdownMessage content={bubble.content} />
          )}
        </div>
        {!isUser && bubble.agentSource !== null ? (
          <Badge variant="outline" className="text-[11px] text-muted-foreground">
            来源：{bubble.agentSource}
          </Badge>
        ) : null}
      </div>
    </div>
  )
}

/** 压缩分隔线：告诉用户「上面的内容模型已经换成摘要了，但原文还在」。 */
function CompactionDivider({ count }: { count: number }) {
  return (
    <div
      className="flex items-center gap-3 py-1"
      role="separator"
      aria-label={`以上 ${String(count)} 条对话已被压缩`}
    >
      <span className="h-px flex-1 bg-border" />
      <span className="shrink-0 text-[11px] text-muted-foreground">
        以上 {count} 条已压成摘要（只影响模型，原文保留）
      </span>
      <span className="h-px flex-1 bg-border" />
    </div>
  )
}

/** 把账本消息转成界面气泡。 */
function toBubbles(messages: ChatMessage[]): Bubble[] {
  return messages.filter(isVisibleMessage).map((message, index) => ({
    id: index + 1,
    role: message.role,
    content: message.content,
    agentSource: message.agent_source,
    pending: false,
  }))
}

export function ChatPage() {
  const [bubbles, setBubbles] = useState<Bubble[]>([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [compactedCount, setCompactedCount] = useState(0)
  // 初始值直接读本地记忆：有会话才需要在首屏拉历史，避免在 effect 里同步 setState
  const [loadingHistory, setLoadingHistory] = useState(() => readLastSession() !== null)
  // TODO(临时): 摘要查看块只用于开发期验收压缩效果，验收后连同状态一起删除
  const [summaryOpen, setSummaryOpen] = useState(false)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [summary, setSummary] = useState<string | null>(null)

  const nextIdRef = useRef(1)
  const abortRef = useRef<AbortController | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  // 是否停在底部：停在底部才跟随新内容，用户自己往上翻时不再把他拽回来
  const stickRef = useRef(true)

  const composer = useForm<ComposerValues>({
    resolver: zodResolver(composerSchema),
    defaultValues: { message: '' },
  })
  // useWatch 走订阅，比 watch() 更贴合 React Compiler
  const message = useWatch({ control: composer.control, name: 'message' })
  const messageError = composer.formState.errors.message
  const hasMessages = !loadingHistory && bubbles.length > 0

  // 打开页面时恢复上一次会话：内容从账本拉，而不是存浏览器里
  useEffect(() => {
    const stored = readLastSession()
    if (stored === null) {
      return
    }

    let cancelled = false
    void (async () => {
      try {
        const ledger = await fetchSessionMessages(stored)
        if (cancelled) {
          return
        }
        const restored = toBubbles(ledger.messages)
        setBubbles(restored)
        nextIdRef.current = restored.length + 1
        setCompactedCount(ledger.compacted_message_count)
        setSessionId(stored)
      } catch (err) {
        if (cancelled) {
          return
        }
        // 会话已被删除或不属于当前用户：忘掉它，退回空白对话
        clearLastSession()
        setError(err instanceof Error ? err.message : '读取历史对话失败')
      } finally {
        if (!cancelled) {
          setLoadingHistory(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  // 监听滚动，判断用户是否还停在底部；手动往上翻后就不再自动跟随
  useEffect(() => {
    const viewport = findViewport(listRef.current)
    if (viewport === null) {
      return
    }
    const onScroll = (): void => {
      stickRef.current = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 32
    }
    viewport.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      viewport.removeEventListener('scroll', onScroll)
    }
  }, [hasMessages])

  // 新内容到达（含流式分片）时跟随到底部。不用 scrollIntoView：它会连外层页面
  // 一起滚，而且 Radix 的 viewport 不对外暴露 ref，只能从 ScrollArea 里找出来直接设
  // scrollTop。
  useEffect(() => {
    if (!stickRef.current) {
      return
    }
    const viewport = findViewport(listRef.current)
    if (viewport !== null) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [bubbles])

  function patchBubble(id: number, patch: Partial<Bubble>): void {
    setBubbles((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }

  /** 每轮结束后同步压缩标记：分隔线位置由后端的账本决定。 */
  async function refreshCompactionMarker(id: string): Promise<void> {
    try {
      const ledger = await fetchSessionMessages(id)
      setCompactedCount(ledger.compacted_message_count)
    } catch {
      // 分隔线只是展示，拉不到就保持原样
    }
  }

  /** 【临时】读取模型当前上下文里的摘要。 */
  async function toggleSummary(): Promise<void> {
    if (sessionId === null) {
      return
    }
    if (summaryOpen) {
      setSummaryOpen(false)
      return
    }
    setSummaryOpen(true)
    setSummaryLoading(true)
    try {
      setSummary(await fetchSessionSummary(sessionId))
    } catch (err) {
      setSummary(null)
      setError(err instanceof Error ? err.message : '读取摘要失败')
    } finally {
      setSummaryLoading(false)
    }
  }

  async function send(rawText: string): Promise<void> {
    const text = rawText.trim()
    if (text === '' || sending) {
      return
    }

    setError(null)
    setSending(true)
    // 自己发消息时无条件回到最新位置，哪怕刚才在往上翻
    stickRef.current = true

    const userId = nextIdRef.current
    const replyId = userId + 1
    nextIdRef.current = userId + 2

    setBubbles((prev) => [
      ...prev,
      { id: userId, role: 'user', content: text, agentSource: null, pending: false },
      { id: replyId, role: 'assistant', content: '', agentSource: null, pending: true },
    ])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const body = sessionId === null ? { message: text } : { message: text, session_id: sessionId }
      for await (const event of streamChat(body, controller.signal)) {
        if (event.type === 'session') {
          setSessionId(event.session_id)
          saveLastSession(event.session_id)
        } else if (event.type === 'chunk') {
          setBubbles((prev) =>
            prev.map((item) =>
              item.id === replyId
                ? { ...item, content: item.content + event.content, agentSource: event.agent_source }
                : item,
            ),
          )
        } else if (event.type === 'done') {
          void refreshCompactionMarker(event.session_id)
        } else if (event.type === 'error') {
          setError(event.content)
        }
      }
    } catch (err) {
      if (!isAbortError(err)) {
        setError(err instanceof Error ? err.message : '请求失败，请检查后端服务是否已启动')
      }
    } finally {
      abortRef.current = null
      setSending(false)
      // 没收到任何内容时不留空气泡
      setBubbles((prev) => prev.filter((item) => item.id !== replyId || item.content !== ''))
      patchBubble(replyId, { pending: false })
    }
  }

  function streamMessage(values: ComposerValues): void {
    // 回复期间按 Enter 不应该吞掉已经输入的内容
    if (sending) {
      return
    }
    composer.reset()
    void send(values.message)
  }

  // 经事件回调再交给 handleSubmit：避免在 render 期间把含 ref 的闭包传给未知函数
  function onFormSubmit(event: FormEvent<HTMLFormElement>): void {
    void composer.handleSubmit(streamMessage)(event)
  }

  function startNewChat(): void {
    abortRef.current?.abort()
    abortRef.current = null
    stickRef.current = true
    setBubbles([])
    setSessionId(null)
    setError(null)
    setSending(false)
    setCompactedCount(0)
    setSummary(null)
    setSummaryOpen(false)
    clearLastSession()
  }

  return (
    <Card className="flex h-[calc(100dvh-14rem)] min-h-96 flex-col gap-0 py-0">
      <CardHeader className="border-b px-5 py-4">
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="size-4 text-muted-foreground" aria-hidden="true" />
          对话
        </CardTitle>
        <CardDescription>
          向健康助手提问，回复会流式逐字显示
          {sessionId === null ? '' : `（会话 ${sessionId.slice(0, 8)}）`}
        </CardDescription>
        <CardAction>
          <Button
            variant="outline"
            size="sm"
            onClick={startNewChat}
            disabled={bubbles.length === 0}
          >
            <Plus />
            新对话
          </Button>
        </CardAction>
      </CardHeader>

      {/* TODO(临时): 开发期用来肉眼确认摘要内容，压缩验收后连同 fetchSessionSummary 一起删除 */}
      {sessionId === null ? null : (
        <div className="border-b bg-muted/30 px-5 py-2">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              临时
            </Badge>
            <span className="text-xs text-muted-foreground">模型当前上下文里的摘要</span>
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto h-7"
              disabled={summaryLoading}
              onClick={() => void toggleSummary()}
            >
              {summaryLoading ? '读取中…' : summaryOpen ? '收起' : '查看'}
            </Button>
          </div>
          {summaryOpen ? (
            <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-background p-3 text-[11px] whitespace-pre-wrap text-muted-foreground">
              {summary ?? '还没有生成摘要：消息条数尚未达到压缩阈值'}
            </pre>
          ) : null}
        </div>
      )}

      {loadingHistory ? (
        <div className="grid flex-1 place-items-center px-5 py-5">
          <Skeleton className="h-4 w-40" />
          <span className="sr-only">正在读取历史对话…</span>
        </div>
      ) : bubbles.length === 0 ? (
        <div className="grid flex-1 place-items-center px-5 py-5 text-center">
          <div>
            <p className="text-sm font-medium">还没有对话</p>
            <p className="mt-1 text-xs text-muted-foreground">
              在下面输入你的健康问题，例如「最近总是睡不好怎么办？」
            </p>
          </div>
        </div>
      ) : (
        /* min-h-0 不能省：Radix 的 ScrollArea root 只有 position:relative（自身
           overflow 仍是 visible），在 column flex 容器里 min-height:auto 会解析成
           内容高度，导致它撑破卡片（卡片是 overflow-hidden）而不是内部滚动 */
        <ScrollArea ref={listRef} className="min-h-0 flex-1">
          <div className="flex flex-col gap-4 px-5 py-5">
            {bubbles.map((bubble, index) => (
              <div key={bubble.id} className="flex flex-col gap-4">
                <BubbleRow bubble={bubble} />
                {compactedCount > 0 && index + 1 === compactedCount ? (
                  <CompactionDivider count={compactedCount} />
                ) : null}
              </div>
            ))}
          </div>
        </ScrollArea>
      )}

      {error !== null ? (
        <div className="px-5 pb-3">
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      ) : null}

      <CardFooter className="border-t px-5 py-4">
        <form noValidate className="w-full" onSubmit={onFormSubmit}>
          <FieldGroup className="gap-2">
            <Field data-invalid={messageError !== undefined}>
              <div className="flex items-end gap-3">
                <Textarea
                  rows={2}
                  placeholder="输入你的问题…"
                  className="max-h-40 min-h-14 flex-1 resize-none"
                  aria-invalid={messageError !== undefined}
                  {...composer.register('message')}
                  onKeyDown={(event) => {
                    // 中文输入法组词期间的 Enter 不应触发发送
                    if (
                      event.key === 'Enter' &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault()
                      // requestSubmit 会走同一个 onSubmit，避免出现第二条提交路径
                      event.currentTarget.form?.requestSubmit()
                    }
                  }}
                />
                <Button
                  type="submit"
                  size="lg"
                  className="h-14 shrink-0 px-5"
                  disabled={sending || message.trim() === ''}
                >
                  <SendHorizontal />
                  {sending ? '回复中…' : '发送'}
                </Button>
              </div>
              <FieldError errors={[messageError]} />
              <FieldDescription>Enter 发送，Shift + Enter 换行</FieldDescription>
            </Field>
          </FieldGroup>
        </form>
      </CardFooter>
    </Card>
  )
}
