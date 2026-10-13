import { CloudUpload } from 'lucide-react'
import { useDropzone } from 'react-dropzone'

import { formatFileSize } from '@/lib/file'
import { ACCEPT_ATTRIBUTE, MAX_FILE_BYTES } from '@/lib/upload-rules'
import { cn } from '@/lib/utils'

/**
 * 拖拽 / 点击选择的上传区。
 *
 * 这里只负责「把用户选中的文件交出去」：类型、大小、空内容的校验由 `onFiles`
 * 的调用方做，这样校验失败的文件能和解析状态显示在同一份列表里。
 */
export function FileUploader({ onFiles }: { onFiles: (files: File[]) => void }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    multiple: true,
    onDropAccepted: (files) => {
      if (files.length > 0) {
        onFiles(files)
      }
    },
    // 这里不配 accept：dropzone 会拿文件 MIME 做判断，而 .md / .heic 这类
    // 文件的 MIME 在浏览器里并不可靠。校验统一走 validateFile（按扩展名），
    // 文件对话框的过滤提示由下面 getInputProps 的 accept 属性提供。
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center transition-colors',
        isDragActive
          ? 'border-primary bg-primary/5'
          : 'border-input hover:border-primary/60 hover:bg-muted/40',
      )}
    >
      <input {...getInputProps({ accept: ACCEPT_ATTRIBUTE })} />
      <span className="grid size-10 place-items-center rounded-full bg-muted text-muted-foreground">
        <CloudUpload className="size-5" aria-hidden="true" />
      </span>
      <p className="text-sm font-medium">
        {isDragActive ? '松开即可上传' : '拖拽文件到此处，或点击选择'}
      </p>
      <p className="text-xs text-muted-foreground">
        支持 PDF、图片等常见格式，可一次选择多份；单个文件不超过
        {formatFileSize(MAX_FILE_BYTES)}
      </p>
    </div>
  )
}
