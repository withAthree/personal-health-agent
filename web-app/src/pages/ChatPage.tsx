import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { Bot, Plus, RotateCcw, SendHorizontal, Sparkles, Square } from 'lucide-react'
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
import { ThinkingBlock } from '@/components/ThinkingBlock'
import {
  fetchSessionMessages,
  fetchSessionSummary,
  fetchStreamState,
  streamChat,
  type ChatMessage,
  type StreamBlockKind,
  type StreamBlockStatus,
  type StreamState,
  type StreamStatus,
} from '@/lib/api'
import { clearLastSession, readLastSession, saveLastSession } from '@/lib/session'
import { cn } from '@/lib/utils'

interface Block {
  /** 后端的 `block_id`；流式分片按它归位（`index` 在重试后会从头复用）。 */
  id: string
  kind: StreamBlockKind
  text: string
  status: StreamBlockStatus
}

interface Bubble {
  id: number
  role: 'user' | 'assistant'
  content: string
  agentSource: string | null
  pending: boolean
  /** 回复被用户主动终止：界面标「已终止」，并给这条提问一个重试入口。 */
  interrupted: boolean
  /** 回复失败（模型报错、网络中断）：气泡里就是错误文案，并给重试入口。 */
  failed: boolean
  /**
   * 本轮的过程块（含正文块）。
   *
   * 旧消息只有 `content` 字符串，这里为空数组；`content` 始终是「这一轮的答案
   * 文本」，过程块只负责展示，重试与空回复清理都只看 `content`。
   */
  blocks: Block[]
  /** 本轮标识，来自 `run-start`；历史消息为 null。 */
  runId: string | null
  /** 本轮开始的本地时刻，用来算「已深度思考 N 秒」。 */
  runStartedAt: number | null
  /** 思考阶段耗时（毫秒）；还没进入正文时为 null。 */
  thinkingMs: number | null
  /** 回放时服务端记下的本轮状态。 */
  runStatus: StreamStatus | null
}

interface DeltaItem {
  /** 目标气泡 id。 */
  id: number
  /** 目标过程块的 `block_id`。 */
  blockId: string
  delta: string
}

/**
 * 把攒下的分片一次性合并进对应气泡。
 *
 * 思考分片实测能到每秒几十片（1015 片 / 1510 字），逐片 `setState` 会把主线程吃满，
 * 所以分片先进 ref 缓冲，由 `requestAnimationFrame` 每帧合并一次。
 */
function applyDeltas(bubbles: Bubble[], items: DeltaItem[]): Bubble[] {
  return bubbles.map((bubble) => {
    const mine = items.filter((item) => item.id === bubble.id)
    if (mine.length === 0) {
      return bubble
    }

    const byBlock = new Map<string, string>()
    for (const item of mine) {
      byBlock.set(item.blockId, (byBlock.get(item.blockId) ?? '') + item.delta)
    }

    let appended = ''
    const blocks = bubble.blocks.map((block) => {
      const delta = byBlock.get(block.id)
      if (delta === undefined) {
        return block
      }
      if (block.kind === 'text') {
        appended += delta
      }
      return { ...block, text: block.text + delta }
    })

    return { ...bubble, blocks, content: bubble.content + appended }
  })
}

/** 本轮开始至今的毫秒数；拿不到起点时返回 null。 */
function elapsedSince(startedAt: number | null): number | null {
  return startedAt === null ? null : Date.now() - startedAt
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

function BubbleRow({
  bubble,
  onRetry,
}: {
  bubble: Bubble
  /** 该轮没有正常答完时的重试回调；为 null 表示不给重试入口。 */
  onRetry: (() => void) | null
}) {
  const isUser = bubble.role === 'user'
  const thinkingBlocks = bubble.blocks.filter((block) => block.kind === 'thinking')
  const thinkingText = thinkingBlocks.map((block) => block.text).join('')
  const thinkingRunning = thinkingBlocks.some((block) => block.status === 'running')
  // 还在思考、正文一个字都还没来时，不显示空气泡：思考面板本身就是「正在处理」
  // 的反馈，也就不再需要原来的骨架屏
  const showAnswer =
    isUser ||
    bubble.pending === false ||
    bubble.content !== '' ||
    bubble.blocks.some((block) => block.kind === 'text')

  return (
    <div className={cn('flex gap-2.5', isUser ? 'justify-end' : 'justify-start')}>
      {isUser ? null : (
        <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-secondary text-secondary-foreground">
          <Bot className="size-4" aria-hidden="true" />
        </span>
      )}
      <div className="max-w-[85%] space-y-1">
        {isUser ? null : (
          <ThinkingBlock
            key={bubble.runId ?? 'history'}
            text={thinkingText}
            running={thinkingRunning}
            thinkingMs={bubble.thinkingMs}
          />
        )}
        {showAnswer ? (
          <div
            className={cn(
              'w-fit rounded-2xl px-4 py-2.5 text-sm',
              isUser
                ? 'rounded-br-sm bg-primary whitespace-pre-wrap text-primary-foreground'
                : 'rounded-bl-sm bg-muted text-foreground',
              // 失败回复：把错误文案当作一条回复显示在对话里，而不是另弹一条提示
              bubble.failed && 'bg-destructive/10 text-destructive',
            )}
          >
            {bubble.pending && bubble.content === '' ? (
              <>
                {/* 气泡底色是 bg-muted，Skeleton 默认的 bg-accent 在同色系上看不出来 */}
                <Skeleton className="h-4 w-24 bg-foreground/15" />
                <span className="sr-only">正在思考…</span>
              </>
            ) : isUser || bubble.failed ? (
              <span className="whitespace-pre-wrap">{bubble.content}</span>
            ) : (
              <MarkdownMessage content={bubble.content} />
            )}
          </div>
        ) : null}
        {isUser ? (
          onRetry === null ? null : (
            <div className="flex justify-end">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 gap-1 px-2 text-[11px] text-muted-foreground"
                onClick={onRetry}
                title="重新回答这个问题"
              >
                <RotateCcw className="size-3" aria-hidden="true" />
                重试
              </Button>
            </div>
          )
        ) : bubble.failed ? (
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[11px] text-destructive">
              失败
            </Badge>
            {onRetry === null ? null : (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 gap-1 px-2 text-[11px] text-muted-foreground"
                onClick={onRetry}
                title="重新回答这个问题"
              >
                <RotateCcw className="size-3" aria-hidden="true" />
                重试
              </Button>
            )}
          </div>
        ) : bubble.interrupted ? (
          <Badge variant="outline" className="text-[11px] text-muted-foreground">
            已终止
          </Badge>
        ) : bubble.agentSource !== null ? (
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
    interrupted: message.interrupted,
    failed: false,
    // 历史只有最终正文，没有过程块：这里不造假的正文块，只留空数组
    blocks: [],
    runId: null,
    runStartedAt: null,
    thinkingMs: null,
    runStatus: null,
  }))
}

/**
 * 把服务端保留的这轮过程块贴回会话末尾。
 *
 * 正文始终以账本为准，这里只补「模型当时的思考」。思考到一半就断了时账本里没有
 * 回复行，那就单独接一个气泡，让它还能挂上「已终止」和重试入口。
 */
function withStreamState(
  bubbles: Bubble[],
  state: StreamState,
  nextId: number,
): { bubbles: Bubble[]; nextId: number } {
  if (state.blocks.length === 0) {
    return { bubbles, nextId }
  }

  const blocks: Block[] = state.blocks.map((block) => ({
    id: block.block_id,
    kind: block.kind,
    text: block.text,
    status: block.status,
  }))
  const last = bubbles[bubbles.length - 1]
  if (last !== undefined && last.role === 'assistant') {
    return {
      bubbles: bubbles.map((bubble) =>
        bubble.id === last.id
          ? { ...bubble, blocks, runId: state.run_id, runStatus: state.status }
          : bubble,
      ),
      nextId,
    }
  }

  return {
    bubbles: [
      ...bubbles,
      {
        id: nextId,
        role: 'assistant',
        content: '',
        agentSource: null,
        pending: false,
        interrupted: true,
        failed: false,
        blocks,
        runId: state.run_id,
        runStartedAt: null,
        thinkingMs: null,
        runStatus: state.status,
      },
    ],
    nextId: nextId + 1,
  }
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
  // 流式分片先攒在这里，每帧只 setState 一次（思考分片实测能到每秒几十片）
  const deltaBufferRef = useRef<DeltaItem[]>([])
  const frameRef = useRef<number | null>(null)

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
          let nextId = restored.length + 1
          // 过程块只是增强信息：拿不到（没记录 / Redis 不可用）就只看账本正文
          let state: StreamState | null = null
          try {
            state = await fetchStreamState(stored)
          } catch {
            state = null
          }
          if (cancelled) {
            return
          }
          let restoredBubbles = restored
          if (state !== null) {
            const merged = withStreamState(restored, state, nextId)
            restoredBubbles = merged.bubbles
            nextId = merged.nextId
          }
          setBubbles(restoredBubbles)
          nextIdRef.current = nextId
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

  /** 把缓冲里的分片立刻合并进气泡（块边界与结束时用，避免丢掉最后一帧）。 */
  function flushDeltas(): void {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }
    const buffered = deltaBufferRef.current
    if (buffered.length === 0) {
      return
    }
    deltaBufferRef.current = []
    setBubbles((prev) => applyDeltas(prev, buffered))
  }

  /** 攒到下一帧再合并，避免每个分片触发一次渲染。 */
  function scheduleFlush(): void {
    if (frameRef.current !== null) {
      return
    }
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = null
      const buffered = deltaBufferRef.current
      if (buffered.length === 0) {
        return
      }
      deltaBufferRef.current = []
      setBubbles((prev) => applyDeltas(prev, buffered))
    })
  }

  /** 新过程块开始：思考与正文的切换就发生在这里。 */
  function startBlock(id: number, blockId: string, kind: StreamBlockKind): void {
    setBubbles((prev) =>
      prev.map((item) => {
        if (item.id !== id) {
          return item
        }
        const blocks: Block[] = [
          ...item.blocks,
          { id: blockId, kind, text: '', status: 'running' },
        ]
        // 正文块一出现就说明思考结束：记下想了多久，收起后还能显示
        const thought = item.blocks.some((block) => block.kind === 'thinking')
        const thinkingMs =
          kind === 'text' && thought && item.thinkingMs === null
            ? elapsedSince(item.runStartedAt)
            : item.thinkingMs
        return { ...item, blocks, thinkingMs }
      }),
    )
  }

  /** 块收尾：写回终态；思考块顺便补上耗时（思考到一半就断了也能显示）。 */
  function stopBlock(id: number, blockId: string, status: StreamBlockStatus): void {
    setBubbles((prev) =>
      prev.map((item) => {
        if (item.id !== id) {
          return item
        }
        const stopped = item.blocks.find((block) => block.id === blockId)
        const blocks = item.blocks.map((block) =>
          block.id === blockId ? { ...block, status } : block,
        )
        const thinkingMs =
          item.thinkingMs === null && stopped?.kind === 'thinking'
            ? elapsedSince(item.runStartedAt)
            : item.thinkingMs
        return { ...item, blocks, thinkingMs }
      }),
    )
  }

  /**
   * 用户主动终止当前回答。
   *
   * 这里只中断前端请求：后端在断流分支里会把已经生成的那部分落库并标记为
   * 已终止，所以刷新页面后这段文字还在，重试也走正常的发消息流程。
   */
  function stopStreaming(): void {
    abortRef.current?.abort()
  }

  /**
   * 重试这一轮没答完的回答。
   *
   * 不新发消息：丢掉这一轮里没答完的回复气泡，让后端用账本里的原文重新回答一次，
   * 分片流进新建的那个气泡。
   */
  function retryTurn(text: string, index: number): void {
    if (sending) {
      return
    }
    setBubbles((prev) => prev.slice(0, index + 1))
    void send(text, { retry: true })
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

  // 一轮对话只能有一个重试入口：失败的回复优先拿走按钮，它前面的提问就不再显示。
  function retryTargetAt(index: number): number | null {
    const bubble = bubbles[index]
    if (sending || bubble === undefined) {
      return null
    }
    const next = bubbles[index + 1]
    // 失败的回复：按钮挂在这条回复上，重试它前面那条提问
    if (bubble.role === 'assistant' && bubble.failed) {
      const previous = index > 0 ? bubbles[index - 1] : undefined
      return previous !== undefined && previous.role === 'user' ? index - 1 : null
    }
    if (bubble.role === 'user') {
      // 后面那条失败的回复已经接管了这个入口
      if (next !== undefined && next.role === 'assistant' && next.failed) {
        return null
      }
      // 这一轮没答完：后面没消息、后面又是提问，或者回复被终止（用户点了「停止」）
      if (next === undefined || next.role === 'user' || next.interrupted) {
        return index
      }
    }
    return null
  }

  async function send(rawText: string, options?: { retry?: boolean }): Promise<void> {
    const text = rawText.trim()
    const isRetry = options?.retry ?? false
    if (text === '' || sending || (isRetry && sessionId === null)) {
      return
    }

    setError(null)
    setSending(true)
    // 自己发消息时无条件回到最新位置，哪怕刚才在往上翻
    stickRef.current = true

    const userId = nextIdRef.current
    const replyId = userId + 1
    nextIdRef.current = userId + 2

    const userBubble: Bubble = {
      id: userId,
      role: 'user',
      content: text,
      agentSource: null,
      pending: false,
      interrupted: false,
      failed: false,
      blocks: [],
      runId: null,
      runStartedAt: null,
      thinkingMs: null,
      runStatus: null,
    }
    const replyBubble: Bubble = {
      id: replyId,
      role: 'assistant',
      content: '',
      agentSource: null,
      pending: true,
      interrupted: false,
      failed: false,
      blocks: [],
      runId: null,
      runStartedAt: null,
      thinkingMs: null,
      runStatus: null,
    }
    // 重试沿用账本里那条提问，不再新增一条用户消息
    setBubbles((prev) => (isRetry ? [...prev, replyBubble] : [...prev, userBubble, replyBubble]))

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const body =
        sessionId === null
          ? { message: text }
          : { message: text, session_id: sessionId, retry: isRetry }
      for await (const event of streamChat(body, controller.signal)) {
        if (event.type === 'session') {
          setSessionId(event.session_id)
          saveLastSession(event.session_id)
        } else if (event.type === 'run-start') {
          patchBubble(replyId, {
            runId: event.run_id,
            agentSource: event.agent_source,
            runStartedAt: Date.now(),
          })
        } else if (event.type === 'reset') {
          // 重新生成：丢掉本轮的块，气泡本身留着
          flushDeltas()
          patchBubble(replyId, { blocks: [] })
        } else if (event.type === 'block-start') {
          flushDeltas()
          startBlock(replyId, event.block_id, event.kind)
        } else if (event.type === 'block-delta') {
          deltaBufferRef.current.push({
            id: replyId,
            blockId: event.block_id,
            delta: event.delta,
          })
          scheduleFlush()
        } else if (event.type === 'block-stop') {
          flushDeltas()
          stopBlock(replyId, event.block_id, event.status)
        } else if (event.type === 'compaction') {
          // 后端直接给了分隔线位置，不必再多拉一次账本
          setCompactedCount(event.covered)
        } else if (event.type === 'chunk') {
          // 旧协议的正文分片：后端已不再发，保留这段是为了回滚协议时前端不用改
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
          // 失败当一条回复显示：气泡里就是错误文案，重试入口在这条回复上
          flushDeltas()
          patchBubble(replyId, { content: event.content, failed: true })
        }
      }
    } catch (err) {
      if (isAbortError(err)) {
        // 用户点了「停止」：保留已经收到的内容，并标上「已终止」
        patchBubble(replyId, { interrupted: true })
      } else {
        // 网络/HTTP 层失败：同样当成一条失败回复，重试入口在那条回复上
        patchBubble(replyId, {
          content: err instanceof Error ? err.message : '请求失败，请检查后端服务是否已启动',
          failed: true,
        })
      }
    } finally {
      // 收尾前先把缓冲里的分片并进去，否则最后不足一帧的内容会被丢掉
      flushDeltas()
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
    deltaBufferRef.current = []
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }
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
            {bubbles.map((bubble, index) => {
              // 返回要重试的提问下标；非 null 才在这条气泡上显示重试按钮
              const retryTarget = retryTargetAt(index)
              return (
                <div key={bubble.id} className="flex flex-col gap-4">
                  <BubbleRow
                    bubble={bubble}
                    onRetry={
                      retryTarget === null
                        ? null
                        : () => retryTurn(bubbles[retryTarget].content, retryTarget)
                    }
                  />
                  {compactedCount > 0 && index + 1 === compactedCount ? (
                    <CompactionDivider count={compactedCount} />
                  ) : null}
                </div>
              )
            })}
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
                  type={sending ? 'button' : 'submit'}
                  size="lg"
                  className="h-14 shrink-0 px-5"
                  onClick={sending ? stopStreaming : undefined}
                  disabled={!sending && message.trim() === ''}
                >
                  {sending ? <Square /> : <SendHorizontal />}
                  {sending ? '停止' : '发送'}
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
