"""LLM 客户端（硅基流动，OpenAI 兼容接口）。"""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent_app.config import settings

#: 功能单元 1 使用的系统提示词，后续由多 Agent 路由替换。
SYSTEM_PROMPT = (
    "你是一名专业的个人健康助手，负责回答一般健康知识问题，"
    "提供准确、有依据的健康信息。请使用简体中文，回答简洁清晰。"
    "你不是医生，涉及诊断、用药或急症时，请明确提示用户咨询专业医疗人员。"
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回对话用的 LLM 客户端单例。"""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.7,
        streaming=True,
    )


@lru_cache(maxsize=1)
def get_summary_llm() -> ChatOpenAI:
    """返回摘要专用的 LLM 客户端单例。

    与对话客户端分开：摘要需要一次性拿到完整文本，不需要流式，
    温度也调低，避免摘要本身被重新演绎。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.2,
        streaming=False,
    )
