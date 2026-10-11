import { useEffect, useRef, useState } from 'react'
import { Brain, ChevronRight } from 'lucide-react'

import { cn } from '@/lib/utils'

/**
 * 思考过程面板。
 *
 * 只做纯文本渲染，**不走 `MarkdownMessage`**：思考分片可以到每秒几十片（实测
 * 1015 片 / 1510 字），逐片重新解析 Markdown 会把主线程吃满。折叠状态本身也是
 * 性能设计——长时间思考默认收起后，正文流式渲染不再为它重排。
 *
 * 展开状态由 `running` 推导而不是用 effect 同步：思考期间默认展开，正文一开始
 * （`running` 变 false）就自动收起；用户手动点过之后则一直尊重用户的选择。
 *
 * 宽度交给外层气泡列（`max-w-[85%]`）：这里是 `w-full`，长度长的思考会自动把
 * 整列撑到上限，而不是把外层的答案气泡一起拉宽。
 */
export function ThinkingBlock({
  text,
  running,
  thinkingMs,
}: {
  /** 已累积的思考原文。 */
  text: string
  /** 思考是否仍在进行（后端尚未发出该块的 `block-stop`）。 */
  running: boolean
  /** 从本轮开始到正文出现的耗时；未知为 null。 */
  thinkingMs: number | null
}) {
  // 流式中自动展开，进入正文后自动收起
  const [manualOpen, setManualOpen] = useState<boolean | null>(null)
  const open = manualOpen ?? running
  const bodyRef = useRef<HTMLDivElement>(null)

  // 思考期间让面板一直显示最新一行，用户不需要自己滚
  useEffect(() => {
    if (!open || !running) {
      return
    }
    const body = bodyRef.current
    if (body !== null) {
      body.scrollTop = body.scrollHeight
    }
  }, [open, running, text])

  if (text === '') {
    return null
  }

  const label = running
    ? '深度思考中…'
    : thinkingMs === null
      ? '深度思考'
      : `已深度思考 ${Math.max(1, Math.round(thinkingMs / 1000))} 秒`

  return (
    <div className="w-full min-w-0">
      <button
        type="button"
        className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        aria-expanded={open}
        onClick={() => {
          setManualOpen(!open)
        }}
      >
        <Brain className={cn('size-3.5', running && 'animate-pulse')} aria-hidden="true" />
        <span>{label}</span>
        <ChevronRight
          className={cn('size-3.5 transition-transform', open && 'rotate-90')}
          aria-hidden="true"
        />
      </button>
      {open ? (
        <div
          ref={bodyRef}
          className="mt-1.5 max-h-64 overflow-y-auto rounded-lg border-l-2 border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground"
        >
          {text}
        </div>
      ) : null}
    </div>
  )
}
