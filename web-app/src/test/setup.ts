import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// 本项目没开 vitest 的 `globals`，RTL 就不会自己注册 afterEach 清理，
// 于是上一个用例渲染的 DOM 会留到下一个用例里（多个同名节点直接报错）。
// 所以显式清理。
afterEach(cleanup)

/**
 * jsdom 不提供 `ResizeObserver`，而 `src/components/EChart.tsx` 用它在容器尺寸变化时
 * 重绘。这里给一个最小实现让组件能挂载；需要断言 resize 行为的测试自己在
 * `beforeEach` 里换成会记录回调的版本。
 */
class NoopResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

globalThis.ResizeObserver = NoopResizeObserver
