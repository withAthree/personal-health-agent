import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { EChart } from '@/components/EChart'
import type { ECOption } from '@/lib/echarts'

/**
 * 用桩替换 echarts 本体：测试要验的是「option 有没有原样交下去」和「实例生命周期」，
 * 不是 echarts 自己怎么画图。真实绘制需要 canvas 或 SVG 量尺寸，jsdom 都不给。
 */
const chart = vi.hoisted(() => {
  const setOption = vi.fn()
  const resize = vi.fn()
  const dispose = vi.fn()
  const init = vi.fn(() => ({ setOption, resize, dispose }))
  return { init, setOption, resize, dispose }
})

vi.mock('@/lib/echarts', () => ({ default: { init: chart.init } }))

/**
 * 这段 option 是后端 `agent-app/src/agent_app/ds/echarts.py` 的 `to_echarts()` 对一张
 * 体重折线图的**真实输出**，原样抄录、未手工调整。
 *
 * 它的意义是「接口契约的线上形态」：DS 子 Agent 算完数据后，落到 `messages.charts`
 * 里、经 SSE 的 `done` 事件送到前端的，就是这样一个对象。前端不该改动它。
 */
const BACKEND_OPTION = {
  title: { text: '体重趋势', left: 8, top: 0 },
  tooltip: { trigger: 'axis' },
  grid: { left: 8, right: 8, top: 48, bottom: 8, containLabel: true },
  xAxis: {
    type: 'category',
    data: ['2026-09-10', '2026-09-18', '2026-10-01', '2026-10-14'],
    boundaryGap: false,
    name: '日期',
  },
  yAxis: { type: 'value', name: '千克' },
  series: [
    {
      name: '体重',
      type: 'line',
      // 中间那一天没有记录：后端保留成 null 让 ECharts 画断点，而不是补 0
      data: [77.5, 77.2, null, 76.3],
      smooth: true,
      symbolSize: 7,
    },
  ],
} as unknown as ECOption

let resizeCallback: (() => void) | null = null
const disconnect = vi.fn()

beforeEach(() => {
  chart.init.mockClear()
  chart.setOption.mockClear()
  chart.resize.mockClear()
  chart.dispose.mockClear()
  disconnect.mockClear()
  resizeCallback = null

  globalThis.ResizeObserver = class {
    constructor(callback: () => void) {
      resizeCallback = callback
    }
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {
      disconnect()
    }
  } as unknown as typeof ResizeObserver
})

describe('EChart', () => {
  it('把后端给的 option 原样交给 echarts', () => {
    render(<EChart option={BACKEND_OPTION} className="h-64" />)

    expect(chart.setOption).toHaveBeenCalledTimes(1)
    expect(chart.setOption).toHaveBeenCalledWith(BACKEND_OPTION, { notMerge: true })
  })

  it('option 换了一个就重新下发，且不是合并', () => {
    const { rerender } = render(<EChart option={BACKEND_OPTION} />)
    const other = { ...BACKEND_OPTION, title: { text: '步数趋势', left: 8, top: 0 } }
    rerender(<EChart option={other} />)

    expect(chart.setOption).toHaveBeenCalledTimes(2)
    expect(chart.setOption).toHaveBeenLastCalledWith(other, { notMerge: true })
  })

  it('notMerge 可以关掉', () => {
    render(<EChart option={BACKEND_OPTION} notMerge={false} />)

    expect(chart.setOption).toHaveBeenCalledWith(BACKEND_OPTION, { notMerge: false })
  })

  it('同一份 option 引用重复渲染不重发', () => {
    const { rerender } = render(<EChart option={BACKEND_OPTION} />)
    rerender(<EChart option={BACKEND_OPTION} />)

    expect(chart.setOption).toHaveBeenCalledTimes(1)
  })

  it('容器尺寸变化时重绘', () => {
    render(<EChart option={BACKEND_OPTION} />)

    expect(resizeCallback).not.toBeNull()
    resizeCallback?.()

    expect(chart.resize).toHaveBeenCalledTimes(1)
  })

  it('卸载时销毁实例并断开观察', () => {
    const { unmount } = render(<EChart option={BACKEND_OPTION} />)
    unmount()

    expect(disconnect).toHaveBeenCalledTimes(1)
    expect(chart.dispose).toHaveBeenCalledTimes(1)
  })

  it('容器是 img 角色，aria-label 可传', () => {
    render(<EChart option={BACKEND_OPTION} ariaLabel="体重趋势" />)

    expect(screen.getByRole('img', { name: '体重趋势' })).toBeInTheDocument()
  })

  it('不传 aria-label 时有兜底文案', () => {
    // 读屏软件读到一个没有名字的 img，等于没有图表
    render(<EChart option={BACKEND_OPTION} />)

    expect(screen.getByRole('img', { name: '数据图表' })).toBeInTheDocument()
  })
})
