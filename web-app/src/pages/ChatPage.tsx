import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import {
  ArrowUp,
  Bot,
  Brain,
  CircleAlert,
  FileText,
  LoaderCircle,
  Paperclip,
  Plus,
  RotateCcw,
  Sparkles,
  Square,
  X,
} from 'lucide-react'
import { useForm, useWatch } from 'react-hook-form'
import { z } from 'zod'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Field, FieldError, FieldGroup } from '@/components/ui/field'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { MarkdownMessage } from '@/components/MarkdownMessage'
import { ThinkingBlock } from '@/components/ThinkingBlock'
import { useChatSession } from '@/contexts/chat-session-context'
import {
  fetchSessionMessages,
  fetchSessionSummary,
  fetchStreamState,
  streamChat,
  uploadFile,
  type ChatMessage,
  type StreamBlockKind,
  type StreamBlockStatus,
  type StreamState,
  type StreamStatus,
} from '@/lib/api'
import { fileExtension, formatFileSize } from '@/lib/file'
import { clearLastSession } from '@/lib/session'
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

/** 输入框上方附件卡片的状态。 */
interface Attachment {
  /** 本地自增 id：异步上传回来时用它判断选中的还是不是当前这个文件。 */
  id: number
  name: string
  size: number
  status: 'uploading' | 'done' | 'failed'
  /** 失败时的提示文案；上传中 / 成功时为 null。 */
  error: string | null
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
  // 首字等待期：`session`/`run-start` 已经回来、第一个过程块还没到。实测这段
  // 空等有 1–14 秒（上游排队波动），此前什么都不渲染，助手一侧只剩一个头像，
  // 看起来像卡死了
  const waiting = bubble.pending && bubble.blocks.length === 0 && bubble.content === ''
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
        {isUser ? null : waiting ? (
          <WaitingIndicator />
        ) : (
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

/** 三个小圆点的错峰延迟（对应 `--animate-dot-wave` 的 1.2s 周期）。 */
const WAITING_DOT_DELAYS = [0, 150, 300] as const

/**
 * 首字等待期的占位反馈。
 *
 * 刻意做成与 `ThinkingBlock` 头部同款（同等字号、同一个大脑图标、同一个位置）：
 * 首个思考分片到达时两个组件在原地交接，不会跳一下。三个点用错峰的上跳波，
 * 而不是 `animate-pulse`——后者落在 6px 的小点上几乎看不出变化。
 */
function WaitingIndicator() {
  return (
    <div className="flex items-center gap-1 text-[11px] text-muted-foreground" role="status">
      <Brain className="size-3.5 animate-pulse" aria-hidden="true" />
      <span>正在思考</span>
      <span className="flex items-center gap-0.5" aria-hidden="true">
        {WAITING_DOT_DELAYS.map((delay) => (
          <span
            key={delay}
            className="size-1.5 animate-dot-wave rounded-full bg-current"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
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

/** 输入框上方的附件卡片：图标 + 文件名 + 大小 / 状态，对应设计参考图。 */
function AttachmentCard({
  attachment,
  onRemove,
}: {
  attachment: Attachment
  onRemove: () => void
}) {
  const { status } = attachment
  const subtitle =
    status === 'uploading'
      ? '上传中...'
      : status === 'failed'
        ? (attachment.error ?? '上传失败')
        : `${fileExtension(attachment.name)} ${formatFileSize(attachment.size)}`.trim()

  return (
    <div className="flex w-fit max-w-full items-center gap-3 rounded-xl border bg-card px-3 py-2">
      <span className="grid size-8 shrink-0 place-items-center">
        {status === 'uploading' ? (
          <LoaderCircle className="size-5 animate-spin text-muted-foreground" aria-hidden="true" />
        ) : status === 'failed' ? (
          <CircleAlert className="size-5 text-destructive" aria-hidden="true" />
        ) : (
          <FileText className="size-5 text-primary" aria-hidden="true" />
        )}
      </span>
      <div className="min-w-0">
        <p className="truncate text-sm font-medium">{attachment.name}</p>
        <p
          className={cn(
            'truncate text-xs',
            status === 'failed' ? 'text-destructive' : 'text-muted-foreground',
          )}
          role={status === 'uploading' ? 'status' : undefined}
        >
          {subtitle}
        </p>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label="移除附件"
        onClick={onRemove}
      >
        <X />
      </Button>
    </div>
  )
}

export function ChatPage() {
  const { sessionId, newChatToken, openSession, startNewChat, refreshSessions } =
    useChatSession()
  const [bubbles, setBubbles] = useState<Bubble[]>([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [compactedCount, setCompactedCount] = useState(0)
  // 初始值直接看当前会话：有会话才需要在首屏拉历史，避免在 effect 里同步 setState
  const [loadingHistory, setLoadingHistory] = useState(() => sessionId !== null)
  // TODO(临时): 摘要查看块只用于开发期验收压缩效果，验收后连同状态一起删除
  const [summaryOpen, setSummaryOpen] = useState(false)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [summary, setSummary] = useState<string | null>(null)
  const [attachment, setAttachment] = useState<Attachment | null>(null)

  const nextIdRef = useRef(1)
  const abortRef = useRef<AbortController | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  // 是否停在底部：停在底部才跟随新内容，用户自己往上翻时不再把他拽回来
  const stickRef = useRef(true)
  // 流式分片先攒在这里，每帧只 setState 一次（思考分片实测能到每秒几十片）
  const deltaBufferRef = useRef<DeltaItem[]>([])
  const frameRef = useRef<number | null>(null)
  // 界面上这批气泡属于哪个会话。用 ref 而不是 state：它只用来判断「要不要重新拉
  // 账本」，不参与渲染
  const loadedSessionRef = useRef<string | null>(null)
  // 上一次的「新对话」计数，用来把「点新对话清空界面」和「会话变化」区分开
  const lastNewChatTokenRef = useRef(newChatToken)
  const fileInputRef = useRef<HTMLInputElement>(null)
  // 附件 id 自增：异步上传回来时用它判断「还是不是当前这个文件」
  const attachmentSeqRef = useRef(0)

  const composer = useForm<ComposerValues>({
    resolver: zodResolver(composerSchema),
    defaultValues: { message: '' },
  })
  // useWatch 走订阅，比 watch() 更贴合 React Compiler
  const message = useWatch({ control: composer.control, name: 'message' })
  const messageError = composer.formState.errors.message
  const hasMessages = !loadingHistory && bubbles.length > 0

  // 会话变化时把账本贴回来：刷新页面恢复上次会话、点侧边栏切换，以及新对话回到空白
  useEffect(() => {
    // 流式新建会话时后端会先发 `session` 事件（见 send），此时账本里还没有助手回复，
    // 重新拉一次反而会把正在生成的气泡抹掉；那种情况下这段已经由事件分支标记过了
    if (loadedSessionRef.current === sessionId) {
      return
    }
    // 先记账再接数据：下面这一整段都是「换到某个会话」的副作用
    loadedSessionRef.current = sessionId
    setBubbles([])
    setCompactedCount(0)
    setError(null)
    setLoadingHistory(sessionId !== null)

    if (sessionId === null) {
      return
    }

    let cancelled = false
    // 这次加载有没有跑完（成功或失败都算）。没跑完就被卸载时要把所有权还回去
    let finished = false
    void (async () => {
      try {
        const ledger = await fetchSessionMessages(sessionId)
        if (cancelled) {
          return
        }
        const restored = toBubbles(ledger.messages)
        let nextId = restored.length + 1
        // 过程块只是增强信息：拿不到（没记录 / Redis 不可用）就只看账本正文
        let state: StreamState | null = null
        try {
          state = await fetchStreamState(sessionId)
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
      } catch (err) {
        if (cancelled) {
          return
        }
        // 会话已被删除或不属于当前用户：忘掉它，退回空白对话
        clearLastSession()
        setError(err instanceof Error ? err.message : '读取历史对话失败')
      } finally {
        finished = true
        if (!cancelled) {
          setLoadingHistory(false)
        }
      }
    })()

    return () => {
      cancelled = true
      // 加载还没跑完就被卸载重挂（React StrictMode 在开发期会故意双调用 effect）：
      // 上面已经记过账，下一个 effect 会以为这批气泡已经贴上而直接跳过，界面就永远
      // 停在骨架屏上。所以把「这批气泡属于哪个会话」还回去，让它重新拉一次
      if (!finished) {
        loadedSessionRef.current = null
      }
    }
  }, [sessionId])

  // 点「新对话」时清空界面。不能只靠 sessionId 的变化来推：网络失败时（还没收到
  // `session` 事件）它本来就是 null，再点一次在 effect 里看不到任何变化
  useEffect(() => {
    if (lastNewChatTokenRef.current === newChatToken) {
      return
    }
    lastNewChatTokenRef.current = newChatToken
    abortRef.current?.abort()
    abortRef.current = null
    deltaBufferRef.current = []
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }
    stickRef.current = true
    loadedSessionRef.current = null
    setBubbles([])
    setError(null)
    setSending(false)
    setCompactedCount(0)
    setSummary(null)
    setSummaryOpen(false)
    // 丢掉附件卡片；同时让还在路上的上传结果失效（id 对不上就被忽略）
    attachmentSeqRef.current += 1
    setAttachment(null)
  }, [newChatToken])

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

  /** 点击回形针：转交给隐藏的 file input。 */
  function pickFile(): void {
    fileInputRef.current?.click()
  }

  /**
   * 选中文件后立刻上传（参考图就是选完即传，卡片先显示「上传中...」）。
   *
   * 一次只带一个附件：新选的文件会替换上一个。后端目前只校验、不保存，上传成功
   * 也只表示「这个文件能收」。
   */
  async function onFileChange(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0]
    // 清空 value，否则连续选同一个文件不会再触发 change
    event.target.value = ''
    if (file === undefined) {
      return
    }

    const id = attachmentSeqRef.current + 1
    attachmentSeqRef.current = id
    setAttachment({ id, name: file.name, size: file.size, status: 'uploading', error: null })
    try {
      const uploaded = await uploadFile(file)
      setAttachment((current) =>
        current?.id === id
          ? { ...current, name: uploaded.filename, size: uploaded.size, status: 'done' }
          : current,
      )
    } catch (err) {
      setAttachment((current) =>
        current?.id === id
          ? {
              ...current,
              status: 'failed',
              error: err instanceof Error ? err.message : '上传失败',
            }
          : current,
      )
    }
  }

  function removeAttachment(): void {
    setAttachment(null)
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
          // 先标记这批气泡已经属于新会话，否则下面 sessionId 一变，加载 effect
          // 就会把正在生成的气泡换成账本里的旧内容
          loadedSessionRef.current = event.session_id
          openSession(event.session_id)
          // 会话刚入库，顺手让侧边栏的「对话记录」多出一条
          void refreshSessions()
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
          // 一轮结束会更新 updated_at，列表顺序跟着变
          void refreshSessions()
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
    // 附件跟着这条消息走：发出去就把卡片收掉（后端当前只校验、不保存）
    setAttachment(null)
    void send(values.message)
  }

  // 经事件回调再交给 handleSubmit：避免在 render 期间把含 ref 的闭包传给未知函数
  function onFormSubmit(event: FormEvent<HTMLFormElement>): void {
    void composer.handleSubmit(streamMessage)(event)
  }

  return (
    /* 外层不再套 Card：右侧主内容区本身就是「聊天窗口」，再加一圈卡片边框只是多
       一层折线，也会让高度算不准 */
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-center gap-2 border-b px-5 py-3">
        <Sparkles className="size-4 text-muted-foreground" aria-hidden="true" />
        <h1 className="text-sm font-semibold">对话</h1>
        <p className="text-xs text-muted-foreground">
          回复会流式逐字显示
          {sessionId === null ? '' : `（会话 ${sessionId.slice(0, 8)}）`}
        </p>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          onClick={startNewChat}
          disabled={bubbles.length === 0}
        >
          <Plus />
          新对话
        </Button>
      </header>

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

      <footer className="shrink-0 px-5 pt-2 pb-4">
        <form noValidate className="w-full" onSubmit={onFormSubmit}>
          <FieldGroup className="gap-2">
            <Field data-invalid={messageError !== undefined}>
              {/* 仿 DeepSeek：外层圆角容器同时包住附件卡片、多行输入和底部工具栏 */}
              <div
                className={cn(
                  'rounded-3xl border bg-card p-2 transition-colors',
                  messageError !== undefined
                    ? 'border-destructive'
                    : 'focus-within:border-ring',
                )}
              >
                {attachment === null ? null : (
                  <div className="px-1 pt-1 pb-2">
                    <AttachmentCard attachment={attachment} onRemove={removeAttachment} />
                  </div>
                )}
                <Textarea
                  rows={2}
                  placeholder="输入你的问题…"
                  className="max-h-40 min-h-14 resize-none border-0! bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0! aria-invalid:ring-0! dark:bg-transparent"
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
                <div className="flex items-center justify-end gap-1 px-1 pt-1">
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    aria-hidden="true"
                    tabIndex={-1}
                    onChange={(event) => void onFileChange(event)}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-lg"
                    className="shrink-0 rounded-full"
                    aria-label="添加附件"
                    title="添加附件"
                    disabled={sending || attachment?.status === 'uploading'}
                    onClick={pickFile}
                  >
                    <Paperclip />
                  </Button>
                  <Button
                    type={sending ? 'button' : 'submit'}
                    size="icon-lg"
                    className="shrink-0 rounded-full"
                    aria-label={sending ? '停止生成' : '发送'}
                    title={sending ? '停止生成' : '发送'}
                    onClick={sending ? stopStreaming : undefined}
                    disabled={
                      !sending && (message.trim() === '' || attachment?.status === 'uploading')
                    }
                  >
                    {sending ? <Square /> : <ArrowUp />}
                  </Button>
                </div>
              </div>
              <FieldError errors={[messageError]} />
            </Field>
          </FieldGroup>
        </form>
      </footer>
    </div>
  )
}
