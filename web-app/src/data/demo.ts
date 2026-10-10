/**
 * 演示用静态数据。
 * 接入后端后由 `@/services` 的真实 API 调用替换。
 */

export interface DailyMetric {
  /** 展示用日期标签，如 `10-04` */
  date: string
  /** 睡眠时长（小时） */
  sleepHours: number
  /** 步数 */
  steps: number
  /** 静息心率（次/分） */
  restingHeartRate: number
}

export const weeklyMetrics: readonly DailyMetric[] = [
  { date: '10-04', sleepHours: 7.2, steps: 8420, restingHeartRate: 61 },
  { date: '10-05', sleepHours: 6.4, steps: 11250, restingHeartRate: 64 },
  { date: '10-06', sleepHours: 8.1, steps: 6380, restingHeartRate: 58 },
  { date: '10-07', sleepHours: 5.6, steps: 4270, restingHeartRate: 68 },
  { date: '10-08', sleepHours: 7.8, steps: 9860, restingHeartRate: 60 },
  { date: '10-09', sleepHours: 7.4, steps: 12480, restingHeartRate: 59 },
  { date: '10-10', sleepHours: 8.3, steps: 7710, restingHeartRate: 57 },
]

export interface HealthScore {
  /** 综合健康评分（0-100） */
  value: number
  /** 评分说明 */
  label: string
}

export const healthScore: HealthScore = { value: 78, label: '良好' }
