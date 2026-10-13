/**
 * 上传校验规则。
 *
 * 这是一份与后端同步的前端副本：权威来源是 `agent-app/src/agent_app/config.py`
 * 里的 `upload_allowed_extensions` 和 `upload_max_bytes`。没有这份副本时，
 * 用户要等后端拒绝才知道文件不合格；后端改规则时这里必须一起改。
 */
import { fileExtensionOf, formatFileSize } from '@/lib/file'

/** 允许的扩展名（小写、不带点），与后端 `upload_allowed_extensions` 一致。 */
export const ACCEPTED_EXTENSIONS: readonly string[] = [
  'jpg',
  'jpeg',
  'png',
  'gif',
  'webp',
  'heic',
  'pdf',
  'txt',
  'md',
  'csv',
  'json',
  'docx',
  'xlsx',
]

/** 单个文件大小上限，与后端 `upload_max_bytes` 一致。 */
export const MAX_FILE_BYTES = 20 * 1024 * 1024

/** 图片类扩展名，只用来决定列表里显示哪个图标。 */
const IMAGE_EXTENSIONS: readonly string[] = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'heic']

/** `<input accept>` 的属性值；只影响文件对话框里的过滤提示，真正的校验看 `validateFile`。 */
export const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(',')

/** 判断文件名是不是图片类型。 */
export function isImageFile(name: string): boolean {
  return IMAGE_EXTENSIONS.includes(fileExtensionOf(name))
}

/** 校验通过返回 `null`，否则返回给用户看的原因。 */
export function validateFile(file: File): string | null {
  if (!ACCEPTED_EXTENSIONS.includes(fileExtensionOf(file.name))) {
    return '不支持的文件类型'
  }
  if (file.size > MAX_FILE_BYTES) {
    return `文件超过 ${formatFileSize(MAX_FILE_BYTES)}`
  }
  if (file.size === 0) {
    return '文件内容为空'
  }
  return null
}
