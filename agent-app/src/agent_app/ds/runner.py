"""沙箱内运行器：从 stdin 读代码与数据，调 ``analysis(df)``，把返回值打成一行 JSON。

这个文件**不在应用进程里执行**。执行器把它作为 ``python -c <源码>`` 的实参起一个子进程，
所以它：

- 不 import ``agent_app`` 的任何东西（沙箱里没有这个包）；
- 只依赖标准库与 pandas；
- 用 ``sys.stdout.write`` 而不是 ``print``——输出的是一行协议，不是给人看的日志。

**输入通道**：stdin 是一整段 JSON，两个键——``code`` 是模型写的 Python 源码，``csv`` 是
``ds.dataset.build_dataset`` 产出的长表原文。**没有文件，也没有工作目录约定**：Docker 下
容器内的路径与 backend 容器的路径不是同一个文件系统，用 ``-v`` 共享一个 CSV 行不通，所以
数据一律走标准流。数据加载由这里负责，不由模型负责——模型拿到的 DataFrame 直接就是
``csv`` 的内容，它没有机会读别的文件或自己造一份数据（调研报告 §4.1）。

**结果通道**：模型自己的打印不作为结果来源，唯一的结果是 stdout 最后一行
``__DS_RESULT__:<json>``（调研报告 §4.2：用 Python 对象传递结果，不解析 stdout）。

**退出码**：``0`` 成功；``1`` 代码抛异常（详情在 stderr）；``2`` 输入不是合法 JSON，或
``code`` 里没有可调用的 ``analysis``。
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

#: 结果行的前缀。与 ``agent_app.ds.executor`` 共用同一个字面量，两边都不能改单边。
RESULT_MARKER = "__DS_RESULT__:"

#: 长表里的日期列。解析成 ``datetime64`` 之后再交给模型，省得每个方案都自己转。
DATE_COLUMN = "measured_date"


def _use_utf8() -> None:
    """把 stdout / stderr 钉成 UTF-8。

    结果通道里要过中文指标名。沙箱镜像的 locale 是 C.UTF-8，本来就是 UTF-8；本地测试却
    跑在 Windows 上，默认是 cp936——同一份结果在两处会产生不同的字节。钉死它，两边的输出
    才可比，也不会因为某个生僻字编码不过去而把一次执行弄崩。

    pytest 会替换 ``sys.stdout``，那种对象没有 ``reconfigure``，所以先判断类型。
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _read_request() -> tuple[str, str]:
    """读 stdin 的 JSON，返回 ``(code, csv_text)``。

    读 ``buffer`` 而不是 ``sys.stdin``：文本层按 locale 解码，本地跑会按 cp936 把 UTF-8
    的中文指标名解成乱码。字节流自己解 UTF-8 与 locale 无关。
    """
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stdin 不是合法 JSON：{exc}") from exc
    if not isinstance(request, dict):
        raise ValueError("stdin 的 JSON 必须是一个对象")
    code = request.get("code")
    csv_text = request.get("csv")
    if not isinstance(code, str) or not isinstance(csv_text, str):
        raise ValueError("stdin 的 JSON 必须有字符串类型的 code 与 csv")
    return code, csv_text


def _load_solution(source: str) -> ModuleType:
    """把源码编译进一个空模块。

    编译时挂上文件名 ``<solution>``，回溯里就会显示这个位置，而不是指到运行器自己身上。
    不用 importlib：没有文件可导入。
    """
    module = ModuleType("ds_solution")
    sys.modules["ds_solution"] = module
    exec(compile(source, "<solution>", "exec"), module.__dict__)
    return module


def _load_frame(csv_text: str) -> pd.DataFrame:
    """解析长表。pandas 由沙箱镜像提供，应用进程里没有它。"""
    import pandas as pd

    return pd.read_csv(io.StringIO(csv_text), parse_dates=[DATE_COLUMN])


def main() -> int:
    _use_utf8()

    try:
        code, csv_text = _read_request()
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        module = _load_solution(code)
        analysis = getattr(module, "analysis", None)
        if not callable(analysis):
            sys.stderr.write("提交的代码里没有可调用的 analysis 函数\n")
            return 2
        payload = analysis(_load_frame(csv_text))
    except BaseException:
        # 连 SystemExit 与 KeyboardInterrupt 一起报出来：这是沙箱内的一次执行，任何
        # 中断都要变成可读的回溯交给上层，而不是静默消失。
        traceback.print_exc()
        return 1

    sys.stdout.write(
        RESULT_MARKER + json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
