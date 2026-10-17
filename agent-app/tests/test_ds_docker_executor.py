"""``DockerExecutor`` 的测试。

绝大多数用例不碰 Docker：``argv`` 是纯函数，``run`` 只要换掉真正跑进程的那一层就能测。
安全参数漏一条不会报错，只会静默地少一层隔离，所以每一条都有断言。

最后两条是端到端的：真镜像、真容器、真 pandas。本机没有 docker 或没构建镜像时跳过
（``docker build -t pha-ds-sandbox:0.1.0 docker/sandbox``）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import pytest

from agent_app.config import settings
from agent_app.ds import docker_executor
from agent_app.ds.docker_executor import (
    SANDBOX_FLAGS,
    DockerExecutor,
    SandboxUnavailableError,
)
from agent_app.ds.executor import RawRun

FAKE_RUNNER = "print('runner')"


def _executor() -> DockerExecutor:
    return DockerExecutor(
        runner_source=FAKE_RUNNER, docker="docker", image="sandbox:test"
    )


def _value_after(argv: Sequence[str], flag: str) -> str | None:
    """取 ``--flag value`` 的值；``flag`` 不在 argv 里时返回 ``None``。"""
    if flag not in argv:
        return None
    index = argv.index(flag)
    return argv[index + 1] if index + 1 < len(argv) else None


def test_every_sandbox_flag_reaches_the_command_line() -> None:
    """``SANDBOX_FLAGS`` 里每一条都必须出现在 argv 里。漏一条就是少一层隔离。"""
    argv = _executor().argv("ds-x")
    assert set(SANDBOX_FLAGS) <= set(argv)


def test_sandbox_is_offline() -> None:
    """沙箱里连不上网，模型代码里的 requests / urllib 一律失败。"""
    assert _value_after(_executor().argv("ds-x"), "--network") == "none"


def test_sandbox_does_not_run_as_root() -> None:
    assert _value_after(_executor().argv("ds-x"), "--user") == "65534:65534"


def test_sandbox_root_filesystem_is_read_only() -> None:
    assert "--read-only" in _executor().argv("ds-x")


def test_sandbox_has_a_writable_tmp_that_cannot_execute() -> None:
    """只读根上唯一的可写位置；``noexec`` 让它也不能用来落一个二进制再执行。"""
    tmpfs = _value_after(_executor().argv("ds-x"), "--tmpfs")
    assert tmpfs is not None
    assert tmpfs.startswith("/tmp:")
    assert "noexec" in tmpfs


def test_sandbox_drops_every_capability() -> None:
    argv = _executor().argv("ds-x")
    assert _value_after(argv, "--cap-drop") == "ALL"
    assert _value_after(argv, "--security-opt") == "no-new-privileges"


def test_sandbox_resources_are_bounded() -> None:
    """死循环或内存膨胀只拖垮这一个容器。"""
    argv = _executor().argv("ds-x")
    for flag in ("--memory", "--memory-swap", "--cpus", "--pids-limit"):
        assert _value_after(argv, flag) is not None, flag


def test_sandbox_does_not_swap() -> None:
    """不写 ``--memory-swap`` 时上限是内存的两倍。"""
    argv = _executor().argv("ds-x")
    assert _value_after(argv, "--memory-swap") == _value_after(argv, "--memory")


def test_container_is_named_and_removed_after_exit() -> None:
    argv = _executor().argv("ds-abc")
    assert _value_after(argv, "--name") == "ds-abc"
    assert "--rm" in argv


def test_runner_is_the_container_command_not_part_of_the_image() -> None:
    """镜像是镜像，运行器是命令行——换运行器不需要重建镜像。"""
    argv = _executor().argv("ds-x")
    image = argv.index("sandbox:test")
    assert argv[image + 1 :] == ["python", "-c", FAKE_RUNNER]


@dataclass
class _Recorder:
    """记录被执行器传下去的东西。"""

    argv: list[str] = field(default_factory=list)
    request: str = ""
    killed: list[str] = field(default_factory=list)


def _install(
    monkeypatch: pytest.MonkeyPatch, raw: RawRun, *, kill_succeeds: bool = True
) -> _Recorder:
    """把真正跑进程的那一层换成直接返回 ``raw``。"""
    recorder = _Recorder()

    def fake_run_blocking(
        argv: Sequence[str], request: str, **kwargs: object
    ) -> RawRun:
        recorder.argv = list(argv)
        recorder.request = request
        return raw

    def fake_kill(self: DockerExecutor, name: str) -> bool:
        recorder.killed.append(name)
        return kill_succeeds

    monkeypatch.setattr(docker_executor, "run_blocking", fake_run_blocking)
    monkeypatch.setattr(DockerExecutor, "_kill_blocking", fake_kill)
    return recorder


def _ok(stdout: str) -> RawRun:
    return RawRun(returncode=0, stdout=stdout, stderr="", timed_out=False)


async def test_code_and_data_go_through_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """沙箱里没有可读的本地文件，代码与数据只能从 stdin 的 JSON 进去。"""
    recorder = _install(monkeypatch, _ok('__DS_RESULT__:{"metrics": []}\n'))

    await _executor().run(code="print(1)", csv_text="a,b\n1,2\n", timeout_s=1.0)

    assert json.loads(recorder.request) == {"code": "print(1)", "csv": "a,b\n1,2\n"}


async def test_success_does_not_touch_the_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _install(monkeypatch, _ok('__DS_RESULT__:{"metrics": []}\n'))

    outcome = await _executor().run(code="x", csv_text="y", timeout_s=1.0)

    assert outcome.status == "ok"
    assert outcome.payload == {"metrics": []}
    assert recorder.killed == []


async def test_missing_result_line_is_an_error_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """退出码 0 但没写结果行——最常见的原因是 ``analysis`` 只打印没 return。"""
    _install(monkeypatch, _ok("正在分析……\n"))

    outcome = await _executor().run(code="x", csv_text="y", timeout_s=1.0)

    assert outcome.status == "error"
    assert outcome.error is not None
    assert "return" in outcome.error


async def test_timeout_kills_the_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """``subprocess.run`` 杀的是 docker CLI，容器还在跑——必须按名字补一刀。"""
    recorder = _install(
        monkeypatch, RawRun(returncode=None, stdout="", stderr="", timed_out=True)
    )

    outcome = await _executor().run(code="x", csv_text="y", timeout_s=1.0)

    assert outcome.status == "timeout"
    assert recorder.killed == [_value_after(recorder.argv, "--name")]
    assert recorder.killed != [None]


async def test_leaked_container_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理失败要写进错误里：``--rm`` 在没跑完的容器上不生效，它会一直留着。"""
    _install(
        monkeypatch,
        RawRun(returncode=None, stdout="", stderr="", timed_out=True),
        kill_succeeds=False,
    )

    outcome = await _executor().run(code="x", csv_text="y", timeout_s=1.0)

    assert outcome.error is not None
    assert "没能清理掉" in outcome.error


async def test_missing_image_raises_instead_of_burning_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """镜像没构建时 ``docker run`` 返回 125。

    这类失败不是模型的错，所以抛异常：不进重试循环，模型不会因为「镜像不存在」把本来
    正确的代码反复重写四遍。
    """
    _install(
        monkeypatch,
        RawRun(
            returncode=125,
            stdout="",
            stderr="Unable to find image 'pha-ds-sandbox:0.1.0' locally",
            timed_out=False,
        ),
    )

    with pytest.raises(SandboxUnavailableError, match="125"):
        await _executor().run(code="x", csv_text="y", timeout_s=1.0)


async def test_code_failure_is_a_retryable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型代码自己报错（退出码 1）是正常结果，要能被重试。"""
    _install(
        monkeypatch,
        RawRun(
            returncode=1,
            stdout="",
            stderr='File "<solution>", line 3, in <module>',
            timed_out=False,
        ),
    )

    outcome = await _executor().run(code="x", csv_text="y", timeout_s=1.0)

    assert outcome.status == "error"
    assert "<solution>" in outcome.failure_note()


REAL_CSV = (
    "measured_date,zh_name,en_name,kind,value_num,low,high,minute_of_day,unit,"
    "ref_low,ref_high,source_value\n"
    "2026-01-01,体重,,scalar,70.0,,,,千克,,,70\n"
    "2026-01-08,体重,,scalar,72.0,,,,千克,,,72\n"
)

REAL_CODE = """
import pandas as pd


def analysis(df: pd.DataFrame) -> dict:
    mean = float(df["value_num"].mean())
    n = int(df["value_num"].count())
    return {
        "metrics": [
            {
                "zh_name": "体重",
                "unit": "千克",
                "statistic": "mean",
                "value": mean,
                "n": n,
                "method": "mean",
            }
        ],
        "sample_size": {"体重": n},
        "findings": [f"平均体重 {mean:.1f} 千克。"],
    }
"""

#: 算同样的东西，外加探一遍隔离是否真的生效。发现的破口写进 warnings，
#: 断言 ``warnings == []`` 就知道是哪一条漏了。
REAL_PROBE_CODE = """
import socket

import pandas as pd


def analysis(df: pd.DataFrame) -> dict:
    mean = float(df["value_num"].mean())
    n = int(df["value_num"].count())
    holes = []

    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        holes.append("网络可达")
    except OSError:
        pass

    try:
        with open("/etc/probe", "w", encoding="utf-8") as handle:
            handle.write("x")
        holes.append("根文件系统可写")
    except OSError:
        pass

    try:
        with open("/tmp/probe", "w", encoding="utf-8") as handle:
            handle.write("x")
    except OSError:
        holes.append("/tmp 不可写")

    return {
        "metrics": [
            {
                "zh_name": "体重",
                "unit": "千克",
                "statistic": "mean",
                "value": mean,
                "n": n,
                "method": "mean",
            }
        ],
        "sample_size": {"体重": n},
        "findings": [f"平均体重 {mean:.1f} 千克。"],
        "warnings": holes,
    }
"""


@lru_cache(maxsize=1)
def _sandbox_ready() -> bool:
    if shutil.which(settings.ds_docker_bin) is None:
        return False
    probe = subprocess.run(
        [settings.ds_docker_bin, "image", "inspect", settings.ds_sandbox_image],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


_sandbox_missing = pytest.mark.skipif(
    not _sandbox_ready(), reason="本机没有 docker，或没构建沙箱镜像"
)


@_sandbox_missing
async def test_real_sandbox_computes_with_pandas() -> None:
    """端到端：真镜像、真容器、真 pandas。

    这是本文件里唯一能证明「命令行真的拼对了」的用例——其余都在假进程上跑。
    """
    outcome = await DockerExecutor().run(
        code=REAL_CODE, csv_text=REAL_CSV, timeout_s=30.0
    )

    assert outcome.status == "ok", outcome.failure_note()
    assert outcome.payload is not None
    assert outcome.payload["metrics"][0]["value"] == 71.0


@_sandbox_missing
async def test_real_sandbox_is_actually_isolated() -> None:
    """隔离参数真的生效，而不只是出现在命令行上。"""
    outcome = await DockerExecutor().run(
        code=REAL_PROBE_CODE, csv_text=REAL_CSV, timeout_s=30.0
    )

    assert outcome.status == "ok", outcome.failure_note()
    assert outcome.payload is not None
    assert outcome.payload["warnings"] == []
