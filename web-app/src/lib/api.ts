import { clearAuth, getToken, type AuthUser } from '@/lib/auth'
import { apiErrorFromResponse, http } from '@/lib/http'

/**
 * 过程块的类型。
 *
 * v1 后端只会发 `thinking` 与 `text`，其余是协议预留：多 Agent 上线后由同一套
 * 事件承载工具调用、结果、分析计划等。前端应当把未知 kind 也当普通文本渲染，
 * 而不是丢弃整条事件。
 */
export type StreamBlockKind =
  | 'thinking'
  | 'text'
  | 'tool_call'
  | 'tool_result'
  | 'plan'
  | 'image'
  | 'error'

/** 块的终态；`running` 表示还在追加内容。 */
export type StreamBlockStatus = 'running' | 'complete' | 'interrupted' | 'error'

/** 一轮（run）的整体状态。 */
export type StreamStatus = 'running' | 'done' | 'error' | 'interrupted'

/**
 * 后端聊天接口的 SSE 事件负载。
 *
 * 字段与 `POST /api/v1/chat` 保持一致，见 agent-app/src/agent_app/api/v1/chat.py。
 * 分片引用 `block_id` 而不是 `index`：重新生成（`reset`）会让 index 从头复用。
 */
export type ChatStreamEvent =
  | { type: 'session'; session_id: string }
  | { type: 'run-start'; run_id: string; agent_source: string }
  | {
      type: 'block-start'
      block_id: string
      index: number
      step: number
      kind: StreamBlockKind
      meta: Record<string, unknown> | null
    }
  | { type: 'block-delta'; block_id: string; delta: string }
  | { type: 'block-stop'; block_id: string; status: StreamBlockStatus }
  | { type: 'reset'; from_index: number }
  | { type: 'compaction'; covered: number }
  | { type: 'usage'; input_tokens: number; output_tokens: number }
  /** 旧协议的正文分片；后端已不再发送，保留解析分支以便回滚。 */
  | { type: 'chunk'; content: string; agent_source: string }
  | { type: 'error'; content: string }
  | { type: 'done'; session_id: string }

/** 已知事件类型；未知类型只记日志，不影响这一轮其余事件。 */
const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set([
  'session',
  'run-start',
  'block-start',
  'block-delta',
  'block-stop',
  'reset',
  'compaction',
  'usage',
  'chunk',
  'error',
  'done',
])

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
  if (typeof type !== 'string' || !KNOWN_EVENT_TYPES.has(type)) {
    // 后端协议比前端新时不要把整条流打断：记一条调试日志然后跳过这条事件
    console.debug('[chat-stream] 忽略未知事件', type, parsed)
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

/** 服务端聚合态里的一个过程块，对应后端 `StreamBlockOut`。 */
export interface StreamStateBlock {
  block_id: string
  index: number
  step: number
  kind: StreamBlockKind
  text: string
  status: StreamBlockStatus
  meta: Record<string, unknown> | null
}

/** 服务端保留的本轮聚合态，对应后端 `StreamStateOut`。 */
export interface StreamState {
  session_id: string
  run_id: string
  status: StreamStatus
  started_at: number
  finished_at: number | null
  blocks: StreamStateBlock[]
  compaction: { covered: number } | null
  usage: Record<string, unknown> | null
}

/**
 * 读取服务端保留的本轮聚合态。
 *
 * 它只活在 Redis 里（TTL 30 分钟），从来不是历史来源：正文一律以账本为准，
 * 这里拿到的过程块只用来还原「刚才那一轮模型想了什么」。没有记录时返回 null。
 */
export async function fetchStreamState(sessionId: string): Promise<StreamState | null> {
  const response = await http.get<StreamState | null>(`/chat/${sessionId}/stream-state`)
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
