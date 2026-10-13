/**
 * 文件展示相关的纯函数。
 *
 * 对话页的附件卡片和个人档案页的文件列表用的是同一套文案格式，
 * 所以抽到这里共用，避免两处各写一份、改一处漏一处。
 */

/** 取小写扩展名（`Report.PDF` → `pdf`）；没有扩展名时返回空串。 */
export function fileExtensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot + 1).toLowerCase()
}

/** 取扩展名并大写（`report.md` → `MD`）；没有扩展名时返回空串。 */
export function fileExtension(name: string): string {
  return fileExtensionOf(name).toUpperCase()
}

/** 把字节数格式化成参考图那样的大小文案（512B / 1.24KB / 2.00MB）。 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${String(bytes)}B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)}KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)}MB`
}

/**
 * 把后端返回的 ISO 时间格式化成 `2026-10-12 21:43`。
 *
 * 后端存的是 UTC，`Date` 会按浏览器时区换算，显示出来和用户上传时的墙上时钟一致。
 * 时间串解析不出来时返回空串，调用方据此不渲染。
 */
export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return ''
  }
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${String(date.getFullYear())}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}
