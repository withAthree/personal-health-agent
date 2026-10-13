import { LoaderCircle, Plus, Trash2 } from 'lucide-react'

import type { ParsedContent, ParsedMetric } from '@/lib/api'
import { cn } from '@/lib/utils'

/** 表头与每行共用的列宽，保证上下对齐。 */
const COLUMNS =
  'grid grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,0.8fr)_minmax(0,1fr)_auto] items-center gap-2'

interface ParsedEditorProps {
  content: ParsedContent
  /** 确认请求进行中，按钮转圈并禁用。 */
  confirming: boolean
  /** 已经写进正式表：面板转只读，只留一句「已入库」。 */
  confirmed: boolean
  onChange: (next: ParsedContent) => void
  onConfirm: () => void
}

/** 受控输入框。`label` 变成 `aria-label`，表格里没有可见表头与之对应。 */
function Field({
  label,
  value,
  onChange,
  className,
  readOnly = false,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  className?: string
  readOnly?: boolean
}) {
  return (
    <input
      className={cn(
        'w-full rounded-md border bg-background px-2 py-1 text-xs outline-none focus:border-ring',
        readOnly && 'text-muted-foreground',
        className,
      )}
      value={value}
      aria-label={label}
      readOnly={readOnly}
      onChange={(event) => onChange(event.target.value)}
    />
  )
}

/**
 * 解析草稿的编辑面板。
 *
 * 组件自己不存状态：整份 `ParsedContent` 由父组件持有，任何一次修改都回调整份新对象。
 * 这样「确定入库」可以直接把当前 content 发出去，不需要在提交时再拼一次。
 *
 * `confirmed` 不只是换按钮文案：确认后的记录按需求不允许再改，所以整个面板转成只读
 * （输入框用 `readOnly` 而不是 `disabled`，这样文本仍然能选中复制）。要修正只能重新
 * 上传这份文件。
 */
export function ParsedEditor({
  content,
  confirming,
  confirmed,
  onChange,
  onConfirm,
}: ParsedEditorProps) {
  const updateMetric = (index: number, patch: Partial<ParsedMetric>) => {
    onChange({
      ...content,
      dataList: content.dataList.map((item, i) => (i === index ? { ...item, ...patch } : item)),
    })
  }

  const removeMetric = (index: number) => {
    onChange({ ...content, dataList: content.dataList.filter((_, i) => i !== index) })
  }

  const addMetric = () => {
    onChange({
      ...content,
      dataList: [
        ...content.dataList,
        { en_name: '', zh_name: '', value: '', unit: '', reference_range: '' },
      ],
    })
  }

  const updateFinding = (index: number, value: string) => {
    onChange({
      ...content,
      otherInfoList: content.otherInfoList.map((item, i) => (i === index ? value : item)),
    })
  }

  const removeFinding = (index: number) => {
    onChange({
      ...content,
      otherInfoList: content.otherInfoList.filter((_, i) => i !== index),
    })
  }

  const addFinding = () => {
    onChange({ ...content, otherInfoList: [...content.otherInfoList, ''] })
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-dashed bg-muted/30 p-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          日期
          <Field
            label="日期"
            className="w-32"
            readOnly={confirmed}
            value={content.date ?? ''}
            onChange={(next) => onChange({ ...content, date: next.trim() === '' ? null : next })}
          />
        </label>
        {content.date_raw !== '' && (
          <span className="min-w-0 truncate text-xs text-muted-foreground" title={content.date_raw}>
            原文：{content.date_raw}
          </span>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <div className={cn(COLUMNS, 'px-2 text-xs text-muted-foreground')}>
          <span>指标</span>
          <span>数值</span>
          <span>单位</span>
          <span>参考范围</span>
          <span className="w-6" />
        </div>

        {content.dataList.map((item, index) => (
          <div key={index} className={COLUMNS}>
            <Field
              label={`第 ${String(index + 1)} 行指标名`}
              readOnly={confirmed}
              value={item.zh_name}
              onChange={(next) => updateMetric(index, { zh_name: next })}
            />
            <Field
              label={`第 ${String(index + 1)} 行数值`}
              readOnly={confirmed}
              value={item.value}
              onChange={(next) => updateMetric(index, { value: next })}
            />
            <Field
              label={`第 ${String(index + 1)} 行单位`}
              readOnly={confirmed}
              value={item.unit}
              onChange={(next) => updateMetric(index, { unit: next })}
            />
            <Field
              label={`第 ${String(index + 1)} 行参考范围`}
              readOnly={confirmed}
              value={item.reference_range}
              onChange={(next) => updateMetric(index, { reference_range: next })}
            />
            {!confirmed && (
              <button
                type="button"
                className="grid size-6 place-items-center rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                aria-label={`删除第 ${String(index + 1)} 行`}
                onClick={() => removeMetric(index)}
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
              </button>
            )}
          </div>
        ))}

        {!confirmed && (
          <button
            type="button"
            className="flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={addMetric}
          >
            <Plus className="size-3.5" aria-hidden="true" />
            添加指标
          </button>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <p className="px-2 text-xs text-muted-foreground">其他发现</p>

        {content.otherInfoList.map((item, index) => (
          <div key={index} className="flex items-center gap-2">
            <Field
              label={`第 ${String(index + 1)} 条其他发现`}
              readOnly={confirmed}
              value={item}
              onChange={(next) => updateFinding(index, next)}
            />
            {!confirmed && (
              <button
                type="button"
                className="grid size-6 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                aria-label={`删除第 ${String(index + 1)} 条其他发现`}
                onClick={() => removeFinding(index)}
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
              </button>
            )}
          </div>
        ))}

        {!confirmed && (
          <button
            type="button"
            className="flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={addFinding}
          >
            <Plus className="size-3.5" aria-hidden="true" />
            添加发现
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3">
        <span className="text-xs text-muted-foreground">
          {content.dataList.length} 项指标，{content.otherInfoList.length} 条发现
        </span>
        {confirmed ? (
          <span className="text-xs text-muted-foreground">
            已入库，需要修正请重新上传这份文件
          </span>
        ) : (
          <button
            type="button"
            className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-60"
            disabled={confirming}
            onClick={onConfirm}
          >
            {confirming && <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />}
            确定入库
          </button>
        )}
      </div>
    </div>
  )
}
