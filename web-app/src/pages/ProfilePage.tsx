import { useCallback, useEffect, useRef, useState } from 'react'
import {
  CircleAlert,
  CircleCheck,
  FileText,
  Image as ImageIcon,
  LoaderCircle,
} from 'lucide-react'

import { FileUploader } from '@/components/FileUploader'
import { uploadFile } from '@/lib/api'
import { fileExtension, formatFileSize } from '@/lib/file'
import { toApiError } from '@/lib/http'
import { isImageFile, validateFile } from '@/lib/upload-rules'
import { cn } from '@/lib/utils'

/** 单个文件在页面上的处理状态。 */
type UploadStatus = 'uploading' | 'parsing' | 'parsed' | 'failed'

interface UploadItem {
  id: string
  file: File
  status: UploadStatus
  error?: string
}

/** 列表右侧的状态文案。 */
const STATUS_LABEL: Record<UploadStatus, string> = {
  uploading: '上传中',
  parsing: '解析中',
  parsed: '解析完成',
  failed: '解析失败',
}

/**
 * 解析接口还没做：后端 `POST /api/v1/upload` 校验完就把文件丢掉了。
 *
 * 所以「上传成功 → 解析完成」这一段先用固定延时占位，好让四种状态都能在
 * 页面上看到。解析接口就绪后把定时器换成轮询文件状态即可，状态机不用改。
 */
const MOCK_PARSE_DELAY_MS = 2500

/** 列表里的一行：图标 + 文件名 + 大小/失败原因 + 状态。 */
function UploadRow({ item }: { item: UploadItem }) {
  const { file, status } = item
  const failed = status === 'failed'
  const busy = status === 'uploading' || status === 'parsing'

  return (
    <li className="flex items-center gap-3 rounded-xl border bg-card px-3 py-2.5">
      <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
        {isImageFile(file.name) ? (
          <ImageIcon className="size-4" aria-hidden="true" />
        ) : (
          <FileText className="size-4" aria-hidden="true" />
        )}
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium" title={file.name}>
          {file.name}
        </p>
        <p
          className={cn(
            'truncate text-xs',
            failed ? 'text-destructive' : 'text-muted-foreground',
          )}
        >
          {failed
            ? (item.error ?? '解析失败')
            : `${fileExtension(file.name)} ${formatFileSize(file.size)}`.trim()}
        </p>
      </div>

      <span
        className={cn(
          'flex shrink-0 items-center gap-1.5 text-xs',
          failed ? 'text-destructive' : busy ? 'text-muted-foreground' : 'text-primary',
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
        {STATUS_LABEL[status]}
      </span>
    </li>
  )
}

/** 个人档案页：上传区 + 每个文件独立的上传/解析状态。 */
export function ProfilePage() {
  const [items, setItems] = useState<UploadItem[]>([])
  // 占位用的定时器；解析接口接上后连同 MOCK_PARSE_DELAY_MS 一起删掉
  const timersRef = useRef(new Map<string, number>())

  useEffect(() => {
    const timers = timersRef.current
    return () => {
      for (const timer of timers.values()) {
        window.clearTimeout(timer)
      }
      timers.clear()
    }
  }, [])

  const patchItem = useCallback((id: string, patch: Partial<UploadItem>) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }, [])

  const startUpload = useCallback(
    async (item: UploadItem) => {
      try {
        await uploadFile(item.file)
      } catch (error) {
        patchItem(item.id, { status: 'failed', error: toApiError(error).message })
        return
      }

      patchItem(item.id, { status: 'parsing' })
      const timer = window.setTimeout(() => {
        timersRef.current.delete(item.id)
        patchItem(item.id, { status: 'parsed' })
      }, MOCK_PARSE_DELAY_MS)
      timersRef.current.set(item.id, timer)
    },
    [patchItem],
  )

  const handleFiles = useCallback(
    (files: File[]) => {
      // 每个文件一个请求并发上传，互不阻塞；校验不过的直接标失败，不进上传流程
      const created = files.map<UploadItem>((file) => {
        const reason = validateFile(file)
        return {
          id: crypto.randomUUID(),
          file,
          status: reason === null ? 'uploading' : 'failed',
          error: reason ?? undefined,
        }
      })

      setItems((prev) => [...created, ...prev])
      for (const item of created) {
        if (item.status === 'uploading') {
          void startUpload(item)
        }
      }
    },
    [startUpload],
  )

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold">个人档案</h1>
        <p className="text-sm text-muted-foreground">
          上传体检报告、化验单或健康截图，系统会自动解析其中的指标。
        </p>
      </div>

      <FileUploader onFiles={handleFiles} />

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium">文件列表</h2>
          <span className="text-xs text-muted-foreground">{items.length} 个文件</span>
        </div>

        {items.length === 0 ? (
          <p className="rounded-xl border border-dashed px-4 py-6 text-center text-xs text-muted-foreground">
            还没有上传文件
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {items.map((item) => (
              <UploadRow key={item.id} item={item} />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
