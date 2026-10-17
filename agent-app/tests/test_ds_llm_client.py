"""DS 客户端（``get_ds_llm``）的选型约束。

这个用例原来挂在 ``agent-app/tests/test_context.py`` 里，跟着那批上下文客户端一起测。
后来测试代码整体不入库，就把它单独挪到 DS 自己的测试文件下，让 ``test_context.py``
回到版本库里的原样。
"""

from __future__ import annotations

import pytest

from agent_app.config import settings
from agent_app.llm.client import get_ds_llm


def test_ds_client_is_low_temperature_but_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DS 客户端独立选型，且温度不能是 0.0。

    温度这条是一个真实约束而非风格偏好：``ds/graph.py`` 的 Code 节点会重试，
    重试的前提是这一次能写出与上一次不同的代码；0.0 下模型容易把同一份代码
    逐字再交一次，把重试预算白烧掉。
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "base-model")
    monkeypatch.setattr(settings, "llm_ds_model", "ds-model")
    get_ds_llm.cache_clear()
    try:
        client = get_ds_llm()

        assert client.model_name == "ds-model"
        assert client.streaming is False
        assert 0 < client.temperature < 0.7, "必须低温，但也不能是 0.0"

        monkeypatch.setattr(settings, "llm_ds_model", "")
        get_ds_llm.cache_clear()
        assert get_ds_llm().model_name == "base-model"
    finally:
        get_ds_llm.cache_clear()
