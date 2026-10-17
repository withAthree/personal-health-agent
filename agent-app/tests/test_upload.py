"""功能单元 4：文件上传接口集成测试。

上传已经不是占位实现：文件会真的落盘（路径是
``settings.upload_dir/{user_id}/{文件 id}.{扩展名}``）、元信息入库、解析任务入队。
所以这里除了校验行为与返回的元信息，还要在用例结束时把落盘的文件删掉——数据库行由
``created_users`` 级联删除，磁盘文件不在其中。

需要 ``docker compose up -d`` 中的 Postgres 与 Redis 可用：登录态要走令牌校验
（吊销名单在 Redis 里）。
"""

import uuid

import httpx
import pytest

import agent_app.config as config_module


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_upload_returns_file_metadata(
    client: httpx.AsyncClient,
    account_token: str,
    created_users: list[uuid.UUID],
) -> None:
    """响应里有落库的元信息、服务端生成的 ``id`` 与解析状态。

    解析在后台队列里跑，所以刚上传时 ``status`` 一定是 ``pending``；``id`` 是后续
    查解析状态的唯一凭据，这里顺手用它取一次详情。
    """
    content = b"# health report\n"
    response = await client.post(
        "/api/v1/upload",
        headers=_auth(account_token),
        files={"file": ("report.md", content, "text/markdown")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {
        "id",
        "filename",
        "size",
        "content_type",
        "extension",
        "status",
    }
    assert payload["filename"] == "report.md"
    assert payload["size"] == len(content)
    assert payload["content_type"] == "text/markdown"
    assert payload["extension"] == "md"
    assert payload["status"] == "pending"

    file_id = uuid.UUID(payload["id"])
    detail = await client.get(f"/api/v1/files/{file_id}", headers=_auth(account_token))
    assert detail.status_code == 200, detail.text
    assert detail.json()["original_name"] == "report.md"

    # 文件真的落盘了，位置由上传接口决定（文件名是 id 的 hex 形式，不带连字符）；
    # 磁盘文件不受数据库级联删除约束，自己收干净。
    path = (
        config_module.settings.upload_dir / str(created_users[-1]) / f"{file_id.hex}.md"
    )
    assert path.read_bytes() == content
    path.unlink()
    path.parent.rmdir()


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
