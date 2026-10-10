/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API 基地址；开发环境留空走 vite 代理 */
  readonly VITE_API_BASE_URL?: string
  /** 应用标题 */
  readonly VITE_APP_TITLE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
