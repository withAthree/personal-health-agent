import { useEffect, useRef } from 'react'

import echarts, { type ECOption, type EChartsType } from '@/lib/echarts'

export interface EChartProps {
  /** ECharts 配置项，类型由 `@/lib/echarts` 中已注册的模块推导 */
  option: ECOption
  /** 容器类名。必须显式给出高度（如 `h-72`），否则 canvas 高度为 0 */
  className?: string
  /** 为 true 时整体替换旧配置而非合并，切换主题或数据结构时更安全 */
  notMerge?: boolean
  /** 无障碍标签 */
  ariaLabel?: string
}

/**
 * ECharts 容器组件：负责实例生命周期与容器尺寸自适应，
 * 配置项本身由调用方通过 `option` 传入。
 */
export function EChart({ option, className, notMerge = true, ariaLabel }: EChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const chart = echarts.init(container)
    chartRef.current = chart

    // 容器尺寸变化（窗口缩放、侧栏折叠等）时重绘；
    // StrictMode 下的重复挂载由 cleanup 中的 dispose 保证不泄漏实例
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(container)

    return () => {
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge })
  }, [option, notMerge])

  return (
    <div ref={containerRef} className={className} role="img" aria-label={ariaLabel ?? '数据图表'} />
  )
}
