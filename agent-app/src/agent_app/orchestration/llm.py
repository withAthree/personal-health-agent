"""图编排各角色的模型选型（**唯一**把「角色」翻译成「模型名」的地方）。

**为什么单独一个模块**：让节点代码不认识模型名——节点写 ``get_agent_llm("ds")``，
换模型只改 ``config.py``，这既是用户要的「三个 Agent 先共用现有模型，但要解耦，
留给后续单独选型」，也是 ``document/模型使用清单.md`` 里「换模型不动业务代码」
那条原则的落点。

**缓存不在这里**：客户端按**模型名**缓存在 ``llm/client.py``（``get_dialog_llm``
/ ``get_utility_llm``），所以这里刻意不加缓存——同一模型的两个角色会拿到同一个
客户端，既省连接也避免出现两份互不同步的实例。``lru_cache`` 是**进程级**单例，
换模型要重启进程（操作见 ``document/模型使用清单.md``）。

**控制节点为什么另走一条路**：编排 / 整合 / 反思 / 收尾四个节点不与用户直接
对话、要一次拿到完整结果、要能 ``with_structured_output``，所以走非流式低温的
``get_utility_llm``，而不是 ``get_dialog_llm``（后者是给对话输出调的：流式、
temperature 0.7）。用户可见的正文只由其中一个节点流式产出，见 ``graph.py`` 顶部。
"""

from langchain_openai import ChatOpenAI

from agent_app.config import settings
from agent_app.llm.client import get_dialog_llm, get_utility_llm
from agent_app.orchestration.state import AgentCode


def _dialog_model(agent: AgentCode) -> str:
    """取某个业务 Agent 配置的模型名；留空回落 ``settings.llm_model``。

    每次调用都现读 ``settings``（不在导入期固化成模块常量），这样配置改动
    （开发期改 ``config.py``、上线改 ``.env``）和测试里的 ``monkeypatch`` 都能
    立刻生效。
    """
    configured: dict[AgentCode, str] = {
        "ds": settings.llm_ds_model,
        "de": settings.llm_de_model,
        "hc": settings.llm_hc_model,
    }
    return configured[agent] or settings.llm_model


def get_agent_llm(agent: AgentCode) -> ChatOpenAI:
    """返回某个业务 Agent 的对话客户端（流式、可带思考）。

    Args:
        agent: Agent 代号，取值见 ``orchestration/state.py`` 的 ``AgentCode``。

    Returns:
        该 Agent 配置的模型对应的客户端；未单独配置时与 ``get_llm()`` 同一个实例。
    """
    return get_dialog_llm(_dialog_model(agent))


def get_general_llm() -> ChatOpenAI:
    """返回通用对话（「不需要分配任务」那条路径）的流式客户端。

    它不属于三个业务 Agent：没有分工时由模型直接作答，所以用对话主模型
    ``settings.llm_model``——与单 Agent 时期同一条路径（``llm/client.py`` 的
    ``get_llm()``）。用户的等待时间全在这条流上，所以必须是流式的，不能用
    ``get_orchestrator_llm()``（非流式）。

    Returns:
        对话主模型对应的流式客户端。
    """
    return get_dialog_llm(settings.llm_model)


def get_orchestrator_llm() -> ChatOpenAI:
    """返回控制节点共用的客户端（非流式、低温）。

    控制节点 = 编排决策 / 结果整合 / 反思 / 收尾。它们都要求稳定地吐出符合
    schema 的结构化输出，所以共用 ``settings.llm_orchestrator_model``，留空时回落
    ``settings.llm_model``。

    温度 0.0：编排是**分类**任务（选角色、派查询），不是创作。0.2 会让「你好」
    「好的」这种极短输入在每次调用间来回摇摆（实测同一输入能分别落到 ds 和
    general），而路由一错整条链路都错。与 ``get_parse_llm`` 同一个理由。

    Returns:
        该模型对应的旁路客户端。
    """
    return get_utility_llm(
        settings.llm_orchestrator_model or settings.llm_model, temperature=0.0
    )
