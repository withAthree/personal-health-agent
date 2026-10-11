import { clearAuth, getToken, type AuthUser } from '@/lib/auth'
import { apiErrorFromResponse, http } from '@/lib/http'

/**
 * 后端聊天接口的 SSE 事件负载。
 *
 * 字段与 `POST /api/v1/chat` 保持一致，见 agent-app/src/agent_app/api/v1/chat.py。
 */
export type ChatStreamEvent =
  | { type: 'session'; session_id: string }
  | { type: 'chunk'; content: string; agent_source: string }
  | { type: 'error'; content: string }
  | { type: 'done'; session_id: string }

export interface ChatRequestBody {
  message: string
  session_id?: string
  /** 重试本会话最后一条用户消息（后端会忽略 message，改用账本里的原文）。 */
  retry?: boolean
}

/** 已登录时带上 Bearer 令牌；SSE 走 fetch，需要自己拼请求头。 */
function toAuthHeaders(): Record<string, string> {
  const token = getToken()
  return token === null ? {} : { Authorization: `Bearer ${token}` }
}

/** 解析单行 SSE；非 `data:` 行（例如 sse-starlette 的 ping 注释）返回 null。 */
function parseDataLine(line: string): ChatStreamEvent | null {
  const trimmed = line.trim()
  if (!trimmed.startsWith('data:')) {
    return null
  }

  const payload = trimmed.slice('data:'.length).trim()
  if (payload === '') {
    return null
  }

  const parsed = JSON.parse(payload) as unknown
  if (typeof parsed !== 'object' || parsed === null) {
    return null
  }

  const { type } = parsed as { type?: unknown }
  if (type !== 'session' && type !== 'chunk' && type !== 'error' && type !== 'done') {
    return null
  }

  return parsed as ChatStreamEvent
}

/**
 * 调用后端聊天接口，按到达顺序逐条产出事件。
 *
 * 这里用 fetch + ReadableStream 而不是 axios 或 EventSource：
 * EventSource 只支持 GET，而本接口需要用 POST 携带 JSON 请求体；
 * axios 的浏览器适配器也拿不到流式的分块数据。
 */
export async function* streamChat(
  body: ChatRequestBody,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent, void, void> {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...toAuthHeaders() },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    throw await apiErrorFromResponse(response)
  }
  if (response.body === null) {
    throw new Error('浏览器未提供响应流，无法接收流式回复')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      // SSE 按行分隔，最后一段可能是半行，留在缓冲区等下一个数据块
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const event = parseDataLine(line)
        if (event !== null) {
          yield event
        }
      }
    }

    // 流结束时缓冲区里可能还剩最后一行（没有结尾换行）
    const tail = parseDataLine(buffer)
    if (tail !== null) {
      yield tail
    }
  } finally {
    try {
      await reader.cancel()
    } catch {
      // 流已正常结束或已被中止，取消失败无需处理
    }
  }
}

export interface Credentials {
  phone: string
  password: string
}
/** `POST /api/v1/auth/register` 与 `/login` 的响应体。 */
export interface AuthTokenResponse {
  access_token: string
  token_type: string
  expires_at: string
  user: AuthUser
}

/** 登录；失败时抛出 `ApiError`。 */
export async function login(credentials: Credentials): Promise<AuthTokenResponse> {
  const response = await http.post<AuthTokenResponse>('/auth/login', credentials)
  return response.data
}

/** 注册；成功后后端直接签发令牌，等于同时完成登录。 */
export async function register(credentials: Credentials): Promise<AuthTokenResponse> {
  const response = await http.post<AuthTokenResponse>('/auth/register', credentials)
  return response.data
}

/** 校验当前令牌，返回用户信息；令牌失效时抛出 401 的 `ApiError`。 */
export async function fetchMe(): Promise<AuthUser> {
  const response = await http.get<AuthUser>('/auth/me')
  return response.data
}

/**
 * 退出登录。
 *
 * 后端会把当前令牌加入吊销名单；不论服务端是否成功，本地登录态都会被清掉。
 */
export async function logout(): Promise<void> {
  try {
    await http.post('/auth/logout')
  } finally {
    clearAuth()
  }
}

/** 账本里的一条消息，字段对应后端 `MessageOut`。 */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  agent_source: string | null
  /** 该助手回复是否被用户主动终止（前端据此显示「已终止」并给出重试入口）。 */
  interrupted: boolean
  created_at: string
}

/** 后端 `SessionMessagesOut`：会话元信息 + 全量消息。 */
export interface SessionMessages {
  session_id: string
  title: string | null
  /** 前多少条消息已被摘要压缩（只影响模型上下文，账本里原文仍在）。 */
  compacted_message_count: number
  messages: ChatMessage[]
}

/**
 * 拉取会话的全量账本。
 *
 * 这是前端渲染历史的唯一来源：模型侧无论怎么压缩，这里拿到的都是原文，
 * 因此用户看到的对话永不缺失。
 */
export async function fetchSessionMessages(sessionId: string): Promise<SessionMessages> {
  const response = await http.get<SessionMessages>(`/chat/${sessionId}/messages`)
  return response.data
}

/**
 * 【临时】读取模型当前上下文里的摘要，只服务开发期肉眼验收压缩效果。
 *
 * TODO(临时): 压缩功能验收完成后，连同 ChatPage 里的摘要查看块一起删除。
 */
export async function fetchSessionSummary(sessionId: string): Promise<string | null> {
  const response = await http.get<{ session_id: string; summary: string | null }>(
    `/chat/${sessionId}/summary`,
  )
  return response.data.summary
}
