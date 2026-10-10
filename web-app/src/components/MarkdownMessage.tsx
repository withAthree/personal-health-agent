import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { cn } from '@/lib/utils'

/**
 * 助手回复的 Markdown 渲染。
 *
 * 只开 GFM（表格、删除线、任务列表等），**不开 rehype-raw**：模型输出里的
 * HTML 一律按纯文本处理，避免注入。
 */
const components: Components = {
  p: ({ children }) => <p className="my-2 whitespace-pre-wrap first:mt-0 last:mb-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ children, href }) => (
    <a
      className="underline underline-offset-2"
      href={href}
      target="_blank"
      rel="noreferrer noopener"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => <ul className="my-2 ml-4 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 ml-4 list-decimal space-y-1">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mt-3 mb-1.5 text-base font-semibold first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-3 mb-1.5 text-base font-semibold first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-3 mb-1.5 text-sm font-semibold first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-3 mb-1.5 text-sm font-semibold first:mt-0">{children}</h4>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-border" />,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-md bg-background/70 p-3 text-xs leading-relaxed">
      {children}
    </pre>
  ),
  code: ({ children, className }) => {
    // react-markdown 10 不再传 inline 标记，带 language- 前缀的才是围栏代码块
    const isBlock = typeof className === 'string' && className.startsWith('language-')
    return isBlock ? (
      <code className={cn('font-mono', className)}>{children}</code>
    ) : (
      <code className="rounded bg-background/70 px-1 py-0.5 font-mono text-[0.85em]">
        {children}
      </code>
    )
  },
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-background/50 px-2 py-1 text-left font-medium">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1 align-top">{children}</td>
  ),
}

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="text-sm leading-relaxed">
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </Markdown>
    </div>
  )
}
