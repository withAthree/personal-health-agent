/**
 * ECharts 按需注册入口。
 *
 * 只注册实际用到的图表与组件，避免把整个 echarts（1MB+）打进产物。
 * 需要新图表类型时：在此处补 import、补 `echarts.use([...])`、补 `ECOption` 联合类型，
 * 三处保持一致，否则 option 会出现类型缺失或运行时组件不生效。
 */
import { BarChart, GaugeChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TitleComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'

import type { BarSeriesOption, GaugeSeriesOption, LineSeriesOption } from 'echarts/charts'
import type {
  GridComponentOption,
  LegendComponentOption,
  TitleComponentOption,
  TooltipComponentOption,
} from 'echarts/components'
import type { ComposeOption } from 'echarts/core'

echarts.use([
  LineChart,
  BarChart,
  GaugeChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  CanvasRenderer,
])

/** 由已注册模块推导出的配置类型，保证 option 里不会出现未注册的图表/组件 */
export type ECOption = ComposeOption<
  | BarSeriesOption
  | GaugeSeriesOption
  | LineSeriesOption
  | GridComponentOption
  | LegendComponentOption
  | TitleComponentOption
  | TooltipComponentOption
>

export type { EChartsType } from 'echarts/core'
export default echarts
