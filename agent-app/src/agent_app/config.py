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

    #: 上下文压缩：触发摘要的 token 阈值（按中文口径估算，见
    #: ``llm/context.py`` 的 ``chinese_token_counter``）。
    #: TODO(窗口确认): 2000 只是测试期的小阈值，方便几轮对话内看到压缩；
    #: 确认模型真实上下文窗口后改成窗口的 35%~40%。
    summary_trigger_tokens: int = 2000

    #: 上下文压缩：摘要后保留的最近消息条数。**必须保持奇数**：
    #: 压缩总是发生在「新用户消息已追加、助手回复还没生成」的时刻，此时消息
    #: 列表长度是偶数（摘要 + 若干完整问答对 + 1 条新提问），保留奇数条才能让
    #: 切点落在完整问答对的边界上；保留偶数条会把一次问答从中间劈开
    #: （提问被摘要、回答留在窗口里）。
    summary_keep_messages: int = 5


settings = Settings()
