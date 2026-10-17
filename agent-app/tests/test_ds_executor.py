"""执行层：子进程协议与沙箱返回值的解析。

跑真实子进程，但注入一个不依赖 pandas 的运行器——应用进程里没有 pandas，而
``runner.py`` 是给沙箱镜像用的。协议（结果行、退出码、超时）与 pandas 无关，所以这里
测的是能测的那部分。
"""

from __future__ import annotations

import io
import sys

import pytest

from agent_app.ds.executor import (
    ExecutionOutcome,
    PayloadError,
    SubprocessExecutor,
    extract_payload,
    result_from_payload,
)
from agent_app.ds.runner import _read_request
from agent_app.ds.schemas import AnalysisPlan

#: 把 stdin 里的 ``csv`` 原文交给 ``analysis``，用来验证数据真的进了子进程。
#: 只做协议验证，不 import pandas。
FAKE_RUNNER = """\
import json
import sys

request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
namespace: dict = {}
exec(compile(request["code"], "<solution>", "exec"), namespace)
print("__DS_RESULT__:" + json.dumps(namespace["analysis"](request["csv"])))
"""

#: 什么都不输出的运行器，用来测「退出码 0 但没有结果行」。
SILENT_RUNNER = """\
import sys

sys.stdin.buffer.read()
"""


def _executor(runner_source: str = FAKE_RUNNER) -> SubprocessExecutor:
    return SubprocessExecutor(runner_source=runner_source)


# ---------------------------------------------------------------- extract_payload


def test_extract_payload_takes_the_last_result_line() -> None:
    stdout = (
        "模型自己打印的东西\n"
        '__DS_RESULT__:{"a": 1}\n'
        "又一个中间输出\n"
        '__DS_RESULT__:{"a": 2}\n'
    )

    assert extract_payload(stdout) == {"a": 2}


def test_extract_payload_returns_none_without_marker() -> None:
    assert extract_payload("普通输出\n") is None


def test_extract_payload_returns_none_on_broken_json() -> None:
    assert extract_payload("__DS_RESULT__:{不是 json}\n") is None


def test_extract_payload_returns_none_when_not_an_object() -> None:
    assert extract_payload("__DS_RESULT__:[1, 2]\n") is None


# ------------------------------------------------------------ result_from_payload


def test_result_from_payload_ignores_unknown_keys() -> None:
    """模型多写一个键不该烧掉一次重试预算。"""
    payload = {
        "findings": ["结论。"],
        "notes": "我额外加的解释",
        "explanation": {"nested": True},
    }

    result = result_from_payload(payload, plan=None, attempts=1)

    assert result.kind == "ok"
    assert result.findings == ["结论。"]


def test_result_from_payload_defaults_kind_to_ok() -> None:
    result = result_from_payload({"findings": ["结论。"]}, plan=None, attempts=1)

    assert result.kind == "ok"


def test_result_from_payload_overwrites_plan_and_attempts() -> None:
    plan = AnalysisPlan(zh_names=["体重"], sufficient=True)
    payload = {"findings": ["结论。"], "plan": None, "attempts": 99}

    result = result_from_payload(payload, plan=plan, attempts=2)

    assert result.plan is plan
    assert result.attempts == 2


def test_result_from_payload_rejects_nan() -> None:
    """算崩的结果不能变成结果。"""
    payload = {
        "metrics": [
            {
                "zh_name": "体重",
                "unit": "公斤",
                "statistic": "平均值",
                "value": float("nan"),
                "n": 1,
                "method": "mean",
            }
        ]
    }

    with pytest.raises(PayloadError, match="metrics"):
        result_from_payload(payload, plan=None, attempts=1)


def test_result_from_payload_reports_which_field_is_wrong() -> None:
    with pytest.raises(PayloadError, match=r"metrics\.0\.n"):
        result_from_payload(
            {
                "metrics": [
                    {
                        "zh_name": "体重",
                        "unit": "公斤",
                        "statistic": "平均值",
                        "value": 70.0,
                        "n": 0,
                        "method": "mean",
                    }
                ]
            },
            plan=None,
            attempts=1,
        )


# ------------------------------------------------------------- SubprocessExecutor


async def test_successful_run_returns_payload() -> None:
    outcome = await _executor().run(
        code=(
            "def analysis(csv_text):\n"
            "    return {'findings': ['读到 %d 字节' % len(csv_text)]}\n"
        ),
        csv_text="measured_date,zh_name\n",
        timeout_s=30.0,
    )

    assert outcome.ok
    assert outcome.payload == {"findings": ["读到 22 字节"]}


async def test_csv_text_reaches_the_subprocess() -> None:
    csv_text = "measured_date,zh_name,value_num\n2026-05-01,体重,70.0\n"

    outcome = await _executor().run(
        code="def analysis(csv_text):\n    return {'findings': [csv_text]}\n",
        csv_text=csv_text,
        timeout_s=30.0,
    )

    assert outcome.payload == {"findings": [csv_text]}


async def test_exception_in_solution_is_an_error_not_a_crash() -> None:
    outcome = await _executor().run(
        code="def analysis(csv_text):\n    raise ValueError('boom')\n",
        csv_text="x\n",
        timeout_s=30.0,
    )

    assert not outcome.ok
    assert outcome.status == "error"
    assert "boom" in outcome.stderr
    assert "ValueError" in outcome.stderr


async def test_missing_analysis_callable_is_an_error() -> None:
    outcome = await _executor().run(
        code="answer = 42\n",
        csv_text="x\n",
        timeout_s=30.0,
    )

    assert not outcome.ok
    assert "analysis" in outcome.stderr


async def test_exit_code_zero_without_result_line_is_an_error() -> None:
    outcome = await _executor(runner_source=SILENT_RUNNER).run(
        code="def analysis(csv_text):\n    return {'findings': ['只打印不返回']}\n",
        csv_text="x\n",
        timeout_s=30.0,
    )

    assert not outcome.ok
    assert outcome.error is not None
    assert "必须 **return** 一个字典" in outcome.error


async def test_real_runner_reports_a_syntax_error() -> None:
    """这里跑的是真正的 ``runner.py``（不是替身），所以它钉住的是 stdin 契约本身。

    代码在编译期就挂了，走不到 ``_load_frame``，因此不需要 pandas也能测。
    """
    outcome = await SubprocessExecutor().run(
        code="def analysis(df) ->\n",
        csv_text="measured_date\n",
        timeout_s=30.0,
    )

    assert outcome.status == "error"
    assert "SyntaxError" in outcome.stderr
    assert "<solution>" in outcome.stderr  # 回溯指向模型写的文件，不是运行器


async def test_real_runner_reports_a_missing_analysis_callable() -> None:
    outcome = await SubprocessExecutor().run(
        code="answer = 42\n",
        csv_text="measured_date\n",
        timeout_s=30.0,
    )

    assert outcome.status == "error"
    assert "没有可调用的 analysis" in outcome.stderr


async def test_timeout_is_reported_as_timeout() -> None:
    outcome = await _executor().run(
        code=(
            "import time\n"
            "def analysis(csv_text):\n"
            "    time.sleep(30)\n"
            "    return {'findings': []}\n"
        ),
        csv_text="x\n",
        timeout_s=0.5,
    )

    assert outcome.status == "timeout"
    assert not outcome.ok
    assert outcome.error is not None
    assert "0.5 秒" in outcome.error


def test_failure_note_carries_the_reason() -> None:
    """失败说明是给模型看的下一轮输入，不能是空的。"""
    outcome = ExecutionOutcome(
        status="error",
        stdout="",
        stderr="Traceback ...\nValueError: boom",
        duration_s=0.1,
        error="执行以退出码 1 结束",
    )

    note = outcome.failure_note()

    assert "执行以退出码 1 结束" in note
    assert "ValueError: boom" in note


# ------------------------------------------------------------- stdin 输入通道
#
# 只测 ``_read_request``：它下游的 ``_load_frame`` 要 pandas，而应用进程里没有 pandas。
# 这一段钉的是 2a 换过的契约——代码与数据走 stdin 的 JSON，不再走文件。


def _stdin(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    """把 ``sys.stdin`` 换成一个有 ``.buffer`` 的替身。

    ``_read_request`` 读的是 ``sys.stdin.buffer``，所以替身必须有这个属性——传
    ``io.StringIO`` 反而不行。
    """

    class _Stub:
        buffer = io.BytesIO(raw)

    monkeypatch.setattr(sys, "stdin", _Stub())


def test_read_request_returns_code_and_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    _stdin(monkeypatch, b'{"code": "a = 1", "csv": "zh_name\\n"}')

    assert _read_request() == ("a = 1", "zh_name\n")


def test_read_request_decodes_utf8_regardless_of_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中文指标名必须原样出来。

    文本层按 locale 解码，Windows 上是 cp936——直接读 ``sys.stdin`` 会把 UTF-8 的
    「体重」解成乱码，而沙箱里的 locale 是 C.UTF-8，同一个输入在两处结果不同。
    """
    _stdin(monkeypatch, '{"code": "体重", "csv": ""}'.encode())

    assert _read_request() == ("体重", "")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"not json at all", "不是合法 JSON"),
        (b"[1, 2, 3]", "必须是一个对象"),
        (b'{"code": "a = 1"}', "字符串类型的 code 与 csv"),
        (b'{"code": 1, "csv": "x"}', "字符串类型的 code 与 csv"),
    ],
)
def test_read_request_rejects_bad_payloads(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, match: str
) -> None:
    _stdin(monkeypatch, raw)

    with pytest.raises(ValueError, match=match):
        _read_request()
