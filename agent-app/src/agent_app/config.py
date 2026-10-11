"""应用配置：从环境变量 / ``agent-app/.env`` 读取。"""

from pathlib import Path

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
    llm_base_url: str = "https://api.siliconflow.cn/v1"

    #: 模型名。**这个默认值只是兜底**：只要 .env / 环境变量里配了 LLM_MODEL，
    #: 就以后者为准（pydantic-settings 优先级：环境变量 > 代码默认值），所以换模型
    #: 要改 .env 并重启后端容器，光改这里不生效。
    llm_model: str = "deepseek-ai/DeepSeek-V3.2"

    #: 旁路任务（上下文摘要、长期记忆抽取）用的模型。
    #: 这两类任务与对话的性质完全不同——非流式、要一次性拿到完整结果、要便宜快，
    #: 记忆抽取还要模型可靠地吐出结构化输出——所以它们能独立选型，换掉时不影响
    #: 对话质量。**留空表示回落到 ``llm_model``**。
    #: 各调用点的要求与验收标准见 ``document/模型使用清单.md``。
    llm_utility_model: str = ""

    #: 是否让对话模型输出思考过程（reasoning）。
    #: **默认关闭**：实测 ``deepseek-ai/DeepSeek-V3.2`` 开启后输出 token 约为关闭时的
    #: 30 倍，首个正文分片从约 3 秒推迟到约 14 秒（见
    #: ``document/中间过程可见-设计方案.md``）。
    #: 通过 ``extra_body`` 透传给 provider：硅基流动的参数名是 ``enable_thinking``。
    #: 旁路任务客户端（``llm/client.py`` 的 ``get_utility_llm``）永远不开启。
    llm_enable_thinking: bool = False

    #: 模型的输入窗口（token）。框架的百分比口径（``("fraction", x)``）必须能从
    #: 模型的 ``profile`` 里读到这个值，否则 ``SummarizationMiddleware`` 在构造期
    #: 就抛 ``ValueError: Model profile information is required...``——所以它不只
    #: 决定下面两个百分比算成多少 token，还是它们能否生效的前提。
    #: 值取 ``deepseek-ai/DeepSeek-V3.2`` 的官方窗口（上游实测接受 48K token 输入）。
    llm_max_input_tokens: int = 128000

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

    #: 文件上传：单文件大小上限（字节）。超过直接拒绝，不落盘。
    #: 当前阶段上传只是占位（校验完即丢弃），这个上限仍要生效，否则等于开放
    #: 一个无限制的接收端。
    upload_max_bytes: int = 5 * 1024 * 1024

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
