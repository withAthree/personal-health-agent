import { useEffect, useRef, useState } from 'react'
import { Bot, Plus, SendHorizontal, Sparkles } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { streamChat } from '@/lib/api'
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
            'rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap',
            isUser
              ? 'rounded-br-sm bg-primary text-primary-foreground'
              : 'rounded-bl-sm bg-muted text-foreground',
          )}
        >
          {bubble.pending && bubble.content === '' ? (
            <span className="text-muted-foreground">正在思考…</span>
          ) : (
            bubble.content
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

export function ChatPage() {
  const [bubbles, setBubbles] = useState<Bubble[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const nextIdRef = useRef(1)
  const abortRef = useRef<AbortController | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // 新内容到达后把列表滚到底部
  useEffect(() => {
    const list = listRef.current
    if (list !== null) {
      list.scrollTop = list.scrollHeight
    }
  }, [bubbles])

  function patchBubble(id: number, patch: Partial<Bubble>): void {
    setBubbles((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }

  async function send(): Promise<void> {
    const text = input.trim()
    if (text === '' || sending) {
      return
    }

    setInput('')
    setError(null)
    setSending(true)

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
        } else if (event.type === 'chunk') {
          setBubbles((prev) =>
            prev.map((item) =>
              item.id === replyId
                ? { ...item, content: item.content + event.content, agentSource: event.agent_source }
                : item,
            ),
          )
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

  function startNewChat(): void {
    abortRef.current?.abort()
    abortRef.current = null
    setBubbles([])
    setSessionId(null)
    setError(null)
    setSending(false)
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

      <div ref={listRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
        {bubbles.length === 0 ? (
          <div className="grid h-full place-items-center text-center">
            <div>
              <p className="text-sm font-medium">还没有对话</p>
              <p className="mt-1 text-xs text-muted-foreground">
                在下面输入你的健康问题，例如「最近总是睡不好怎么办？」
              </p>
            </div>
          </div>
        ) : (
          bubbles.map((bubble) => <BubbleRow key={bubble.id} bubble={bubble} />)
        )}
      </div>

      {error !== null ? (
        <div className="px-5 pb-3">
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      ) : null}

      <CardFooter className="border-t px-5 py-4">
        <form
          className="w-full space-y-2"
          onSubmit={(event) => {
            event.preventDefault()
            void send()
          }}
        >
          <div className="flex items-end gap-3">
            <Textarea
              value={input}
              rows={2}
              placeholder="输入你的问题…"
              onChange={(event) => {
                setInput(event.target.value)
              }}
              onKeyDown={(event) => {
                // 中文输入法组词期间的 Enter 不应触发发送
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  void send()
                }
              }}
              className="max-h-40 min-h-14 flex-1 resize-none"
            />
            <Button
              type="submit"
              size="lg"
              className="h-14 shrink-0 px-5"
              disabled={sending || input.trim() === ''}
            >
              <SendHorizontal />
              {sending ? '回复中…' : '发送'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">Enter 发送，Shift + Enter 换行</p>
        </form>
      </CardFooter>
    </Card>
  )
}
