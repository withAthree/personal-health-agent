import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChevronDown,
  CircleAlert,
  CircleCheck,
  FileText,
  Image as ImageIcon,
  LoaderCircle,
} from 'lucide-react'

import { FileUploader } from '@/components/FileUploader'
import { ParsedEditor } from '@/components/ParsedEditor'
import {
  confirmFile,
  fetchFile,
  listFiles,
  uploadFile,
  type FileStatus,
  type ParsedContent,
} from '@/lib/api'
import { fileExtension, formatDateTime, formatFileSize } from '@/lib/file'
import { toApiError } from '@/lib/http'
import { isImageFile, validateFile } from '@/lib/upload-rules'
import { cn } from '@/lib/utils'

/**
 * 页面上的行状态，比后端多一个 `uploading`。
 *
 * `pending` / `parsing` 都转圈，区别只是「还没轮到」和「正在解析」；`parsed` 是
 * 「解析出来了、等用户确认」；`confirmed` 是已经写进正式表。
 */
type RowStatus = 'uploading' | FileStatus

interface UploadItem {
  /** React key，整个生命周期不变；`fileId` 要等上传成功才知道。 */
  key: string
  fileId: string | null
  name: string
  size: number
  status: RowStatus
  /** 上传时间（ISO），列表里直接展示，解析不出来日期时它就是正式表里的时间。 */
  createdAt?: string
  error?: string
  content?: ParsedContent
}

/** 列表右侧的状态文案。 */
const STATUS_LABEL: Record<RowStatus, string> = {
  uploading: '上传中',
  pending: '排队中',
  parsing: '解析中',
  parsed: '待确认',
  confirmed: '已入库',
  failed: '解析失败',
}

/**
 * 轮询间隔。
 *
 * 解析一张手机长截图要十几秒，2 秒查一次够及时，也不会把后端问烦。
 */
const POLL_INTERVAL_MS = 2000

/** 列表里的一行：图标 + 文件名 + 大小/失败原因 + 上传时间 + 状态；能展开的行整行可点。 */
function UploadRow({
  item,
  expanded,
  expandable,
  onToggle,
}: {
  item: UploadItem
  expanded: boolean
  expandable: boolean
  onToggle: () => void
}) {
  const failed = item.status === 'failed'
  const done = item.status === 'confirmed'
  const busy =
    item.status === 'uploading' || item.status === 'pending' || item.status === 'parsing'
  const uploadedAt = item.createdAt === undefined ? '' : formatDateTime(item.createdAt)

  return (
    <button
      type="button"
      className={cn(
        'flex w-full items-center gap-3 text-left',
        expandable ? 'cursor-pointer' : 'cursor-default',
      )}
      aria-expanded={expandable ? expanded : undefined}
      onClick={() => {
        if (expandable) {
          onToggle()
        }
      }}
    >
      <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
        {isImageFile(item.name) ? (
          <ImageIcon className="size-4" aria-hidden="true" />
        ) : (
          <FileText className="size-4" aria-hidden="true" />
        )}
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium" title={item.name}>
          {item.name}
        </p>
        <p
          className={cn(
            'truncate text-xs',
            failed ? 'text-destructive' : 'text-muted-foreground',
          )}
        >
          {failed
            ? (item.error ?? '解析失败')
            : `${fileExtension(item.name)} ${formatFileSize(item.size)}`.trim()}
        </p>
      </div>

      {uploadedAt !== '' && (
        <span className="shrink-0 text-xs text-muted-foreground">{uploadedAt}</span>
      )}

      <span
        className={cn(
          'flex shrink-0 items-center gap-1.5 text-xs',
          failed
            ? 'text-destructive'
            : busy || done
              ? 'text-muted-foreground'
              : 'text-primary',
        )}
        role={busy ? 'status' : undefined}
      >
        {busy ? (
          <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
        ) : failed ? (
          <CircleAlert className="size-4" aria-hidden="true" />
        ) : (
          <CircleCheck className="size-4" aria-hidden="true" />
        )}
        {STATUS_LABEL[item.status]}
      </span>

      {expandable && (
        <ChevronDown
          className={cn(
            'size-4 shrink-0 text-muted-foreground transition-transform',
            expanded && 'rotate-180',
          )}
          aria-hidden="true"
        />
      )}
    </button>
  )
}

/** 有草稿、且已经解析完的行才能展开（失败和还在转圈的行没东西可看）。 */
function canExpand(item: UploadItem): boolean {
  return (
    item.fileId !== null &&
    item.content !== undefined &&
    (item.status === 'parsed' || item.status === 'confirmed')
  )
}

/** 个人档案页：上传区 + 每个文件独立的状态 + 解析草稿确认。 */
export function ProfilePage() {
  const [items, setItems] = useState<UploadItem[]>([])
  const [confirmingKey, setConfirmingKey] = useState<string | null>(null)
  // 展开的确认表。默认收起，只有「刚解析完」那一次会自动展开
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set())
  // 自动展开过的 key，避免用户手动收起后又被下一轮轮询重新展开
  const autoExpandedRef = useRef(new Set<string>())
  // 轮询回调里要知道某个 id 对应哪一行，所以留一份最新列表的镜像
  const itemsRef = useRef<UploadItem[]>([])
  itemsRef.current = items

  const toggleExpanded = useCallback((key: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }, [])

  // 进页面时把历史文件捞回来，刷新后列表不丢
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const files = await listFiles()
        if (cancelled) {
          return
        }
        setItems(
          files.map((file) => ({
            key: file.id,
            fileId: file.id,
            name: file.original_name,
            size: file.size,
            status: file.status,
            error: file.error_message ?? undefined,
            content: file.parsed_content ?? undefined,
            createdAt: file.created_at,
          })),
        )
      } catch {
        // 历史列表拉不到不影响上传本身，上传失败会单独报错
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // 把「需要轮询的 id」拼成一个字符串当依赖，避免每次渲染都重建定时器
  const pendingKey = items
    .filter(
      (item) =>
        item.fileId !== null && (item.status === 'pending' || item.status === 'parsing'),
    )
    .map((item) => item.fileId)
    .join(',')

  useEffect(() => {
    if (pendingKey === '') {
      return
    }
    const ids = pendingKey.split(',')
    let cancelled = false

    const tick = async () => {
      for (const id of ids) {
        try {
          const detail = await fetchFile(id)
          if (cancelled) {
            return
          }
          // 这一轮只会查 pending/parsing 的行，所以拿到 parsed 就是「刚解析完」
          if (detail.status === 'parsed') {
            const row = itemsRef.current.find((item) => item.fileId === id)
            if (row !== undefined && !autoExpandedRef.current.has(row.key)) {
              autoExpandedRef.current.add(row.key)
              const key = row.key
              setExpandedKeys((prev) => new Set(prev).add(key))
            }
          }
          setItems((prev) =>
            prev.map((item) =>
              item.fileId === id
                ? {
                    ...item,
                    status: detail.status,
                    error: detail.error_message ?? undefined,
                    content: detail.parsed_content ?? undefined,
                    createdAt: detail.created_at,
                  }
                : item,
            ),
          )
        } catch {
          // 单次轮询失败不改状态，下一次 tick 再试
        }
      }
    }

    void tick()
    const timer = window.setInterval(() => void tick(), POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [pendingKey])

  const startUpload = useCallback(async (key: string, file: File) => {
    try {
      const uploaded = await uploadFile(file)
      setItems((prev) =>
        prev.map((item) =>
          item.key === key
            ? {
                ...item,
                fileId: uploaded.id,
                status: uploaded.status as FileStatus,
                // 先用本地时间顶上，轮询拿到服务端 created_at 后会覆盖
                createdAt: new Date().toISOString(),
              }
            : item,
        ),
      )
    } catch (error) {
      setItems((prev) =>
        prev.map((item) =>
          item.key === key
            ? { ...item, status: 'failed', error: toApiError(error).message }
            : item,
        ),
      )
    }
  }, [])

  const handleFiles = useCallback(
    (files: File[]) => {
      // 每个文件一个请求并发上传，互不阻塞；校验不过的直接标失败，不进上传流程
      const created = files.map<UploadItem>((file) => {
        const reason = validateFile(file)
        return {
          key: crypto.randomUUID(),
          fileId: null,
          name: file.name,
          size: file.size,
          status: reason === null ? 'uploading' : 'failed',
          error: reason ?? undefined,
        }
      })

      setItems((prev) => [...created, ...prev])
      for (const [index, item] of created.entries()) {
        const file = files[index]
        if (item.status === 'uploading' && file !== undefined) {
          void startUpload(item.key, file)
        }
      }
    },
    [startUpload],
  )

  const updateContent = useCallback((key: string, content: ParsedContent) => {
    setItems((prev) => prev.map((item) => (item.key === key ? { ...item, content } : item)))
  }, [])

  const handleConfirm = useCallback(
    async (key: string, fileId: string, content: ParsedContent) => {
      setConfirmingKey(key)
      try {
        await confirmFile(fileId, content)
        setItems((prev) =>
          prev.map((item) =>
            item.key === key ? { ...item, status: 'confirmed', error: undefined } : item,
          ),
        )
      } catch (error) {
        setItems((prev) =>
          prev.map((item) =>
            item.key === key ? { ...item, error: toApiError(error).message } : item,
          ),
        )
      } finally {
        setConfirmingKey(null)
      }
    },
    [],
  )

  /** 只有解析完、且已展开的行才渲染编辑面板。 */
  const renderEditor = (item: UploadItem) => {
    const { content, fileId } = item
    if (
      fileId === null ||
      content === undefined ||
      !canExpand(item) ||
      !expandedKeys.has(item.key)
    ) {
      return null
    }
    return (
      <ParsedEditor
        content={content}
        confirming={confirmingKey === item.key}
        confirmed={item.status === 'confirmed'}
        onChange={(next) => updateContent(item.key, next)}
        onConfirm={() => void handleConfirm(item.key, fileId, content)}
      />
    )
  }

  return (
    <div className="flex h-full flex-col gap-6 p-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold">个人档案</h1>
        <p className="text-sm text-muted-foreground">
          上传体检报告、化验单或健康截图，系统会自动解析其中的指标。
        </p>
      </div>

      <FileUploader onFiles={handleFiles} />

      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium">文件列表</h2>
          <span className="text-xs text-muted-foreground">{items.length} 个文件</span>
        </div>

        {items.length === 0 ? (
          <p className="rounded-xl border border-dashed px-4 py-6 text-center text-xs text-muted-foreground">
            还没有上传文件
          </p>
        ) : (
          <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-2 [scrollbar-gutter:stable]">
            {items.map((item) => (
              <li
                key={item.key}
                className="flex flex-col gap-3 rounded-xl border bg-card px-3 py-2.5"
              >
                <UploadRow
                  item={item}
                  expanded={expandedKeys.has(item.key)}
                  expandable={canExpand(item)}
                  onToggle={() => {
                    toggleExpanded(item.key)
                  }}
                />
                {renderEditor(item)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
