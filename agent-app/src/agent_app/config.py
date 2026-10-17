"""应用配置：从环境变量 / ``agent-app/.env`` 读取。"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``agent-app/`` 目录，用于定位 ``.env``。
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行期配置。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debug: bool = True
    secret_key: str = "change-me"

    database_url: str = "postgresql+psycopg://pha:123456@localhost:5432/pha"
    redis_url: str = "redis://:123456@localhost:6379/0"

    backend_port: int = 8000

    #: 访问令牌有效期（小时）。
    access_token_expire_hours: int = 48

    llm_api_key: str = ""
    #: DeepSeek 官方接口（OpenAI 兼容格式）。base_url **不要带 `/v1`**：官方文档给的是
    #: `https://api.deepseek.com`，带 `/v1` 也能用，但两处写法不一致会让人以为是两个
    #: 不同的端点。切换 provider 时这里必须与 ``agent-app/.env`` 的 ``LLM_BASE_URL``
    #: 保持一致或直接删掉那一行——环境变量优先级高于代码默认值。
    llm_base_url: str = "https://api.deepseek.com"

    #: 对话模型名。**开发期直接改这里**：后端跑 ``uvicorn --reload``，改完进程自动
    #: 重启、``lru_cache`` 的客户端跟着重建，不用碰容器。上线或临时切换时再用 .env
    #: 里的 ``LLM_MODEL`` 覆盖（pydantic-settings 优先级：环境变量 > 代码默认值），
    #: 那种情况下才需要重建容器——``docker compose restart`` 不会重读 ``env_file``，
    #: 要用 ``up -d``。
    llm_model: str = "deepseek-flash"

    #: 上下文摘要（压缩）用的模型。摘要是纯改写任务，输入长、输出短，且**在用户
    #: 等待路径上**，所以要便宜且够快。**留空表示回落到 ``llm_model``**。
    #: 开发期直接改这里即可（理由见 ``llm_model``）；各调用点的要求与验收标准见
    #: ``document/模型使用清单.md``。
    llm_summary_model: str = "deepseek-flash"

    #: 长期记忆抽取用的模型。它必须在后台可靠地吐出符合 schema 的结构化输出
    #: （``memory/extract.py`` 的 ``MemoryOps``），判错会污染之后每一轮对话，
    #: 所以选型标准与摘要不同，两者分开配置。**留空表示回落到 ``llm_model``**。
    #: 开发期也直接改这里。
    llm_memory_model: str = "deepseek-flash"

    #: 文档解析抽取用的模型：把 OCR 出来的纯文本转成 ``dataList`` /
    #: ``otherInfoList`` 结构。它**与对话主流程解耦**——不复用 ``get_llm`` 的
    #: 客户端，也不跟着 ``llm_model`` 走，因为这个模型的选型将来可能单独更换。
    #: **留空表示回落到 ``llm_model``**。开发期直接改这里即可（理由见
    #: ``llm_model``）。
    llm_parse_model: str = "deepseek-flash"

    #: 数据分析子 Agent（DS）用的模型：把自然语言问题变成分析计划（Plan），再把计划
    #: 变成一段 pandas 代码（Code）。两个调用点在 ``ds/graph.py`` 的 ``LlmDsModel``，
    #: 共用一个客户端。
    #: **它不跟着 ``llm_model`` 走，必须写死一个模型名**：DS 的失败是静默的——代码
    #: 跑得通、数字算错，用户拿到的是一份看起来正常的错误结论，不会报错。所以不能
    #: 因为「把闲聊模型换成便宜的了」让它跟着降级。**留空表示回落到 ``llm_model``**
    #: （不推荐，仅用于临时排查）。开发期直接改这里即可（理由见 ``llm_model``）。
    llm_ds_model: str = "deepseek-flash"

    #: 是否让对话模型输出思考过程（reasoning）。
    #: **DeepSeek 官方接口的思考模式默认是开的**（默认档 ``high``），关闭必须显式传
    #: ``extra_body={"thinking": {"type": "disabled"}}``（见 ``llm/client.py`` 的
    #: ``get_llm()``），不传等于开着。硅基流动相反——那边不传就是关。
    #: 开启后工具调用也能走通：官方要求带 ``tools`` 时把每轮的 ``reasoning_content``
    #: 原样回传，否则 400；实测当前链路（``ChatOpenAIWithReasoning`` + langchain +
    #: DeepSeek 官方接口）能正常回环，没有 400，且 ``reasoning_content`` 能被拾回。
    #: 代价是真实存在的：在硅基流动的 ``deepseek-ai/DeepSeek-V3.2`` 上实测开启后输出
    #: token 约为关闭时的 30 倍、首个正文分片从约 3 秒推迟到约 14 秒（见
    #: ``document/中间过程可见-设计方案.md``）。
    #: 经 ``llm/client.py`` 的 ``_build_utility_llm()`` 构造的客户端永远不开启。
    llm_enable_thinking: bool = False

    #: 思考**强度**，只在 ``llm_enable_thinking`` 为真时生效，作为请求体根上的
    #: ``reasoning_effort`` 透传给 DeepSeek。官方默认档是 ``high``。
    #: 可取的只有这三个：``low`` / ``high`` / ``max``——官方另外还接受
    #: minimal / medium / xhigh / ultra，但它们分别映射到 low/high/high/max，
    #: 写 ``medium`` 与写 ``high`` 完全等价，写了只会让人误以为省了钱。
    llm_thinking_effort: Literal["low", "high", "max"] = "low"

    #: 模型的输入窗口（token）。框架的百分比口径（``("fraction", x)``）必须能从
    #: 模型的 ``profile`` 里读到这个值，否则 ``SummarizationMiddleware`` 在构造期
    #: 就抛 ``ValueError: Model profile information is required...``——所以它不只
    #: 决定下面两个百分比算成多少 token，还是它们能否生效的前提。
    #: 值取 ``deepseek-flash``（模型版本 ``DeepSeek-V4.1-Flash``）的官方窗口 1M，
    #: 最大输出 384K（https://api-docs.deepseek.com/quick_start/pricing）。
    #: 副作用要知道：摘要触发线是窗口的 0.6，窗口变成 1M 之后触发线就是 600K token，
    #: **正常对话里摘要几乎不会触发**。这是预期行为——窗口够大，而且
    #: 缓存命中的输入价格极低，保留完整历史比压缩更好。真触发时配置照旧生效。
    llm_max_input_tokens: int = 1_000_000

    #: 上下文压缩：触发摘要的阈值。**当前生效的是百分比口径**
    #: （``llm/context.py`` 的 ``SUMMARY_TRIGGER_FRACTION``），按
    #: ``llm_max_input_tokens`` 的 0.8 触发。估算口径见 ``llm/context.py`` 的
    #: ``chinese_token_counter``（按中文校准，偏保守、会高估）。
    #: **别调小**：摘要是一次非流式调用，实测要几十秒，且完全阻塞在回答之前。
    #: 另外，框架还会拿「上一条回复上报的 token 数」独立判断一次、与本阈值取或，
    #: 而 provider 上报的数字曾经被打大几百倍（见 ``llm/client.py`` 的
    #: ``_is_terminal_stream_chunk``），导致摘要无视本阈值每轮都触发；那段修复
    #: 之前，把这里从 2000 提到 40000 并不起作用。
    #: ``summary_trigger_tokens`` 是等价的绝对口径，当前未被使用。
    summary_trigger_tokens: int = 40000
    summary_trigger_fraction: float = 0.6

    #: 上下文压缩：摘要后保留多少最近内容。**当前生效的是百分比口径**
    #: （``llm/context.py`` 的 ``SUMMARY_KEEP_FRACTION``），保留窗口的 0.1。
    #: ``summary_keep_messages`` 是等价的条数口径，当前未被使用；用它时
    #: **必须保持奇数**：压缩总是发生在「新用户消息已追加、助手回复还没生成」
    #: 的时刻，此时消息列表长度是偶数（摘要 + 若干完整问答对 + 1 条新提问），
    #: 保留奇数条才能让切点落在完整问答对的边界上；保留偶数条会把一次问答从
    #: 中间劈开（提问被摘要、回答留在窗口里）。
    summary_keep_messages: int = 21
    summary_keep_fraction: float = 0.1

    #: 文件上传：落盘根目录。上传的文件按 ``{user_id}/{file_id}.{ext}`` 存放，
    #: 原始文件名不参与路径拼接（防路径穿越），只作为展示用的元信息返回。
    #: 容器里 ``BASE_DIR`` 是 ``/app``，所以是 ``/app/uploads``，由
    #: ``docker/docker-compose.yml`` 挂到宿主机，否则容器重建即丢。
    upload_dir: Path = BASE_DIR / "uploads"

    #: 文件上传：单文件大小上限（字节）。超过直接拒绝，不落盘。
    upload_max_bytes: int = 20 * 1024 * 1024

    #: 文件上传：允许的扩展名（小写、不含点），逗号分隔。
    #: 刻意用字符串而不是集合：pydantic-settings 对复杂类型默认按 JSON 解析，
    #: 写成"jpg,png"会在启动时直接解析失败。取用时走下面的
    #: ``upload_allowed_extension_set``。
    upload_allowed_extensions: str = (
        "jpg,jpeg,png,gif,webp,heic,pdf,txt,md,csv,json,docx,xlsx"
    )

    @property
    def upload_allowed_extension_set(self) -> frozenset[str]:
        """把逗号分隔的扩展名配置解析成集合（忽略空白与空项）。"""
        return frozenset(
            item.strip().lower()
            for item in self.upload_allowed_extensions.split(",")
            if item.strip()
        )

    #: 数据分析子 Agent（DS）的代码执行沙箱镜像。由 ``docker/sandbox/Dockerfile``
    #: 构建（``docker build -t pha-ds-sandbox:0.1.0 docker/sandbox``），**不是**
    #: ``docker-compose.yml`` 里的一个服务：沙箱不常驻，每次分析起一个一次性容器。
    #: 镜像里只有 pandas / numpy / scipy，没有本项目代码——运行器由命令行传进去
    #: （``agent_app/ds/docker_executor.py``）。
    #: 容器的隔离参数**不可配置**：它们是安全边界，不该有「用环境变量关掉」的开关。
    ds_sandbox_image: str = "pha-ds-sandbox:0.1.0"

    #: docker 二进制路径。做成配置项是为了可测——测试直接指到一个假路径，不用改 PATH。
    #: backend 容器里的 ``docker`` 由 ``agent-app/Dockerfile`` 从 ``docker:cli`` 镜像
    #: 拷进来，并靠 ``docker/docker-compose.yml`` 挂进来的 ``docker.sock``
    #: 连宿主守护进程。
    ds_docker_bin: str = "docker"

    #: 长期记忆：总开关。关闭后既不注入也不抽取（表与代码仍在）。
    memory_enabled: bool = True

    #: 长期记忆：注入时的总条数上限，超出按「越新越优先」截断。
    #: 记忆是每次调用都注入的，所以这个值是**每轮对话的固定成本**，
    #: 调大要按 ``memory_max_injected`` × 单条长度的 token 量重新估算窗口。
    memory_max_injected: int = 50

    #: 长期记忆：``preference`` / ``insight`` 两类各注入最近多少条。
    #: ``health_fact`` / ``obstacle`` / ``goal`` 不受此限制——这三类天然少，
    #: 且漏掉一次就是一次错答，必须「总是在场」。
    memory_max_per_loose_category: int = 20

    #: 长期记忆：写入时的置信度门槛，低于它的条目直接丢弃；``0`` 表示不过滤
    #: 只记录。**默认 0 是刻意的**：模型自报的 confidence 没有校准（普遍过度
    #: 自信、跨模型不可比），先用它攒样本看真实准确率，不要拿它当硬阈值。
    memory_min_confidence: float = 0.0


settings = Settings()
