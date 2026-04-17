from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


class TestUrlSafety:
    def test_blocks_localhost(self) -> None:
        from tools._net_safety import validate_public_http_url

        s = validate_public_http_url("http://localhost:8000/")
        assert s.ok is False

    def test_blocks_private_ip(self) -> None:
        from tools._net_safety import validate_public_http_url

        s = validate_public_http_url("http://192.168.1.10/")
        assert s.ok is False

    def test_allows_public_ip(self) -> None:
        from tools._net_safety import validate_public_http_url

        s = validate_public_http_url("https://1.1.1.1/")
        assert s.ok is True

    def test_blocks_non_http_scheme(self) -> None:
        from tools._net_safety import validate_public_http_url

        s = validate_public_http_url("file:///etc/passwd")
        assert s.ok is False


class TestApprovalPolicy:
    def test_scoped_auto_allows_low_risk_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.protocols import mcp_client

        monkeypatch.setenv("SLM_AGENT_APPROVAL_MODE", "scoped_auto")
        ok, reason = mcp_client._approve("write_file", {"path": "notes.txt", "content": "hi"})
        assert ok is True
        assert reason == ""

    def test_scoped_auto_prompts_on_high_risk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.protocols import mcp_client

        monkeypatch.setenv("SLM_AGENT_APPROVAL_MODE", "scoped_auto")
        # High blast-radius write: pyproject
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
        ok, reason = mcp_client._approve("write_file", {"path": "pyproject.toml", "content": "x"})
        assert ok is False
        assert "DENIED" in reason


class TestProviderDetection:
    def test_unknown_when_not_repo(self, tmp_path: Path) -> None:
        from tools.pr_tools import pr_fetch

        out = pr_fetch(1, cwd=str(tmp_path))
        assert out.get("provider") in {"unknown", "github", "gitlab"}
        assert out.get("provider") == "unknown"

    def test_detects_github_from_remote_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tools.pr_tools as pr

        def fake_run(cmd: list[str], cwd: str | None, timeout_s: float) -> pr._CmdResult:
            return pr._CmdResult(
                returncode=0,
                stdout="origin\thttps://github.com/acme/repo.git (fetch)\n",
                stderr="",
            )

        monkeypatch.setattr(pr, "_run", fake_run)
        assert pr._detect_provider(".") == "github"


class TestVerifyRepoTool:
    def test_verify_repo_shape(self) -> None:
        from tools.verify_repo import verify_repo

        out = verify_repo(path=".", run_tests=False)
        assert isinstance(out.get("ok"), bool)
        assert isinstance(out.get("results"), list)
        if out["results"]:
            item = out["results"][0]
            assert "validator" in item
            assert "outcome" in item

