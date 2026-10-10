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

const composerSchema = z.object({
  message: z.string().trim().min(1, '请输入内容'),
})

type ComposerValues = z.infer<typeof composerSchema>

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
            <>
              {/* 气泡底色是 bg-muted，Skeleton 默认的 bg-accent 在同色系上看不出来 */}
              <Skeleton className="h-4 w-24 bg-foreground/15" />
              <span className="sr-only">正在思考…</span>
            </>
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
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const nextIdRef = useRef(1)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const composer = useForm<ComposerValues>({
    resolver: zodResolver(composerSchema),
    defaultValues: { message: '' },
  })
  // useWatch 走订阅，比 watch() 更贴合 React Compiler
  const message = useWatch({ control: composer.control, name: 'message' })
  const messageError = composer.formState.errors.message

  // 新内容到达后滚到底部；ScrollArea 的 viewport 拿不到 ref，所以用列表末尾的哨兵
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [bubbles])

  function patchBubble(id: number, patch: Partial<Bubble>): void {
    setBubbles((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }

  async function send(rawText: string): Promise<void> {
    const text = rawText.trim()
    if (text === '' || sending) {
      return
    }

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

      {bubbles.length === 0 ? (
        <div className="grid flex-1 place-items-center px-5 py-5 text-center">
          <div>
            <p className="text-sm font-medium">还没有对话</p>
            <p className="mt-1 text-xs text-muted-foreground">
              在下面输入你的健康问题，例如「最近总是睡不好怎么办？」
            </p>
          </div>
        </div>
      ) : (
        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-4 px-5 py-5">
            {bubbles.map((bubble) => (
              <BubbleRow key={bubble.id} bubble={bubble} />
            ))}
            <div ref={endRef} aria-hidden="true" />
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
        <form
          noValidate
          className="w-full"
          onSubmit={onFormSubmit}
        >
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
