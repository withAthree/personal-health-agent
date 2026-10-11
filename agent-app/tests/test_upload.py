"""功能单元 4：文件上传接口集成测试。

当前上传是**占位实现**：只校验扩展名与大小，校验通过后丢弃文件。所以这里的断言
集中在「校验行为」与「返回的元信息」，不涉及任何落盘或入库——测试也因此不需要
额外的清理逻辑。

需要 ``docker compose up -d`` 中的 Postgres 与 Redis 可用：登录态要走令牌校验
（吊销名单在 Redis 里）。
"""

import httpx
import pytest

import agent_app.config as config_module


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_upload_returns_file_metadata(
    client: httpx.AsyncClient, account_token: str
) -> None:
    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("report.md", b"# health report\n", "text/markdown")},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "filename": "report.md",
        "size": len(b"# health report\n"),
        "content_type": "text/markdown",
        "extension": "md",
    }


async def test_upload_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("report.md", b"# health report\n", "text/markdown")},
    )

    assert response.status_code == 401


async def test_upload_rejects_disallowed_extension(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_file_type"


async def test_upload_rejects_missing_extension(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("README", b"no extension", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_file_type"


async def test_upload_rejects_empty_file(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_file"


async def test_upload_rejects_oversized_file(
    client: httpx.AsyncClient,
    account_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 直接造一个 5MB+ 的文件太重：把上限调小，验证的是「超限即拒」这条分支。
    monkeypatch.setattr(config_module.settings, "upload_max_bytes", 16)

    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("big.txt", b"x" * 64, "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "file_too_large"
