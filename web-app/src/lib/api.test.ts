import { describe, expect, it, vi } from 'vitest'

import { streamChat, type ChatStreamEvent } from '@/lib/api'
import type { ECOption } from '@/lib/echarts'

/** 把若干行 SSE 拼成一个流式 Response；`chunks` 用来模拟分块到达。 */
function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

async function collect(chunks: string[]): Promise<ChatStreamEvent[]> {
  vi.stubGlobal(
    'fetch',
    vi.fn(() => Promise.resolve(sseResponse(chunks))),
  )
  const events: ChatStreamEvent[] = []
  for await (const event of streamChat({ message: '我最近的体重是多少？' })) {
    events.push(event)
  }
  return events
}

/** 后端 `agent-app/src/agent_app/ds/echarts.py` 的真实输出形状（原样抄录）。 */
const CHART_OPTION = {
  title: { text: '体重趋势', left: 8, top: 0 },
  tooltip: { trigger: 'axis' },
  grid: { left: 8, right: 8, top: 48, bottom: 8, containLabel: true },
  xAxis: {
    type: 'category',
    data: ['2026-09-10', '2026-10-14'],
    boundaryGap: false,
    name: '日期',
  },
  yAxis: { type: 'value', name: '千克' },
  series: [{ name: '体重', type: 'line', data: [77.5, 76.3], smooth: true, symbolSize: 7 }],
} as unknown as ECOption

describe('streamChat 的图表通路', () => {
  it('done 事件把图表带给调用方，内容一字不改', async () => {
    const events = await collect([
      'data: {"type":"session","session_id":"s1"}\n',
      `data: ${JSON.stringify({ type: 'done', session_id: 's1', charts: [CHART_OPTION] })}\n`,
    ])

    const done = events.at(-1)
    expect(done).toMatchObject({ type: 'done', session_id: 's1' })
    expect((done as { charts: ECOption[] }).charts).toEqual([CHART_OPTION])
  })

  it('这一轮没有图表时 charts 缺省，前端据此渲染成空列表', async () => {
    const events = await collect(['data: {"type":"done","session_id":"s1"}\n'])

    expect(events.at(-1)).not.toHaveProperty('charts')
  })

  it('一行的半个 JSON 分两次到达时不丢事件', async () => {
    const line = 'data: {"type":"done","session_id":"s1","charts":[]}\n'
    const cut = 25
    const events = await collect([line.slice(0, cut), line.slice(cut)])

    expect(events).toEqual([{ type: 'done', session_id: 's1', charts: [] }])
  })

  it('sse-starlette 的注释行（ping）被跳过', async () => {
    const events = await collect([
      ': ping\n',
      'data: {"type":"done","session_id":"s1"}\n',
    ])

    expect(events).toHaveLength(1)
  })

  it('未知事件类型被丢掉，但不打断整条流', async () => {
    const events = await collect([
      'data: {"type":"something_new","session_id":"s1"}\n',
      'data: {"type":"done","session_id":"s1"}\n',
    ])

    expect(events.map((event) => event.type)).toEqual(['done'])
  })
})
