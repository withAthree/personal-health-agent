import type { ReactNode } from 'react'
import { Footprints, Gauge, HeartPulse, Moon } from 'lucide-react'

import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { EChart } from '@/components/EChart'
import { healthScore, weeklyMetrics } from '@/data/demo'
import type { ECOption } from '@/lib/echarts'

const average = (values: readonly number[]): number =>
  values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length

const summary = {
  avgSleepHours: average(weeklyMetrics.map((metric) => metric.sleepHours)),
  avgSteps: Math.round(average(weeklyMetrics.map((metric) => metric.steps))),
  avgRestingHeartRate: Math.round(
    average(weeklyMetrics.map((metric) => metric.restingHeartRate)),
  ),
}

const sleepAndActivityOption: ECOption = {
  tooltip: { trigger: 'axis' },
  legend: { data: ['睡眠时长', '步数'], top: 0, icon: 'roundRect' },
  grid: { left: 8, right: 8, top: 52, bottom: 8, containLabel: true },
  xAxis: {
    type: 'category',
    data: weeklyMetrics.map((metric) => metric.date),
    boundaryGap: false,
  },
  yAxis: [
    {
      type: 'value',
      name: '睡眠',
      axisLabel: { formatter: '{value} h' },
      splitLine: { lineStyle: { type: 'dashed' } },
    },
    {
      type: 'value',
      name: '步数',
      axisLabel: { formatter: '{value}' },
      splitLine: { show: false },
    },
  ],
  series: [
    {
      name: '睡眠时长',
      type: 'line',
      smooth: true,
      symbolSize: 7,
      areaStyle: { opacity: 0.12 },
      data: weeklyMetrics.map((metric) => metric.sleepHours),
    },
    {
      name: '步数',
      type: 'bar',
      yAxisIndex: 1,
      barWidth: '38%',
      itemStyle: { borderRadius: [4, 4, 0, 0] },
      data: weeklyMetrics.map((metric) => metric.steps),
    },
  ],
}

const heartRateOption: ECOption = {
  tooltip: { trigger: 'axis' },
  grid: { left: 8, right: 8, top: 32, bottom: 8, containLabel: true },
  xAxis: {
    type: 'category',
    data: weeklyMetrics.map((metric) => metric.date),
    boundaryGap: false,
  },
  yAxis: {
    type: 'value',
    name: '次/分',
    min: 50,
    max: 75,
    splitLine: { lineStyle: { type: 'dashed' } },
  },
  series: [
    {
      name: '静息心率',
      type: 'line',
      smooth: true,
      symbolSize: 7,
      lineStyle: { width: 3 },
      data: weeklyMetrics.map((metric) => metric.restingHeartRate),
    },
  ],
}

const healthScoreOption: ECOption = {
  series: [
    {
      type: 'gauge',
      startAngle: 210,
      endAngle: -30,
      min: 0,
      max: 100,
      radius: '92%',
      progress: { show: true, width: 14, roundCap: true },
      axisLine: { lineStyle: { width: 14 } },
      axisTick: { show: false },
      splitLine: { length: 8, distance: 12 },
      axisLabel: { distance: 20, fontSize: 10 },
      pointer: { show: false },
      anchor: { show: false },
      title: { show: false },
      detail: {
        valueAnimation: true,
        offsetCenter: [0, '-8%'],
        fontSize: 32,
        fontWeight: 'bolder',
        formatter: '{value}',
      },
      data: [{ value: healthScore.value, name: '健康评分' }],
    },
  ],
}

interface StatTileProps {
  label: string
  value: string
  hint: string
  icon: ReactNode
}

function StatTile({ label, value, hint, icon }: StatTileProps) {
  return (
    <Card>
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
        <CardAction>
          <span className="grid size-8 place-items-center rounded-lg bg-muted text-muted-foreground">
            {icon}
          </span>
        </CardAction>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  )
}

function ChartCard({
  title,
  description,
  className,
  children,
}: {
  title: string
  description: string
  className?: string
  children: ReactNode
}) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">健康总览</h2>
        <p className="mt-1 text-sm text-muted-foreground">最近 7 天的睡眠、活动与心血管指标</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="平均睡眠"
          value={`${summary.avgSleepHours.toFixed(1)} h`}
          hint="近 7 天平均"
          icon={<Moon className="size-4" aria-hidden="true" />}
        />
        <StatTile
          label="日均步数"
          value={summary.avgSteps.toLocaleString('zh-CN')}
          hint="近 7 天平均"
          icon={<Footprints className="size-4" aria-hidden="true" />}
        />
        <StatTile
          label="静息心率"
          value={`${String(summary.avgRestingHeartRate)} 次/分`}
          hint="近 7 天平均"
          icon={<HeartPulse className="size-4" aria-hidden="true" />}
        />
        <StatTile
          label="健康评分"
          value={`${String(healthScore.value)} / 100`}
          hint={healthScore.label}
          icon={<Gauge className="size-4" aria-hidden="true" />}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartCard
          className="lg:col-span-2"
          title="睡眠与活动"
          description="折线为睡眠时长，柱状为每日步数"
        >
          <EChart
            option={sleepAndActivityOption}
            className="h-72 w-full"
            ariaLabel="近 7 天睡眠时长与步数组合图"
          />
        </ChartCard>
        <ChartCard title="健康评分" description="综合睡眠、活动与心率计算">
          <EChart
            option={healthScoreOption}
            className="h-72 w-full"
            ariaLabel={`健康评分 ${String(healthScore.value)} 分`}
          />
        </ChartCard>
      </div>

      <ChartCard title="静息心率趋势" description="静息心率下降通常代表心肺适应改善">
        <EChart
          option={heartRateOption}
          className="h-64 w-full"
          ariaLabel="近 7 天静息心率趋势图"
        />
      </ChartCard>
    </div>
  )
}
