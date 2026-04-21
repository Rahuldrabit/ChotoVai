from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import reset_config
from src.orchestrator.task_router import TaskRouter


@pytest.mark.asyncio
async def test_bare_filename_resolves_deterministically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config()
    repo_root = tmp_path

    (repo_root / "src" / "aa").mkdir(parents=True)
    (repo_root / "src" / "bb").mkdir(parents=True)

    (repo_root / "src" / "aa" / "utils.py").write_text(
        "def a() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "bb" / "utils.py").write_text(
        "def b() -> int:\n    return 2\n",
        encoding="utf-8",
    )

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please update utils.py to handle edge-cases.")

    assert res.mentioned_files == ["src/aa/utils.py"]
    assert "## src/aa/utils.py" in res.prefetched_stub_map

    # Ensure we don't leak temp root absolute paths into the injected context
    repo_root_str = str(repo_root).replace("\\", "/")
    assert repo_root_str not in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_explicit_relative_path_mention_is_resolved(tmp_path: Path) -> None:
    reset_config()
    repo_root = tmp_path
    (repo_root / "src" / "bb").mkdir(parents=True)
    (repo_root / "src" / "bb" / "mod.py").write_text(
        "def target(x: int) -> int:\n    return x\n",
        encoding="utf-8",
    )

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Look at src/bb/mod.py and update target.")

    assert "src/bb/mod.py" in res.mentioned_files
    assert "target" in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_non_python_file_prefetches_stubs(tmp_path: Path) -> None:
    reset_config()
    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "util.ts").write_text(
        "function ping(x: number): number { return x; }\n",
        encoding="utf-8",
    )

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please inspect src/util.ts before planning")

    assert "src/util.ts" in res.mentioned_files
    assert "ping" in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_python_docstring_is_prefetched(tmp_path: Path) -> None:
    reset_config()
    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "docmod.py").write_text(
        'def f(x: int) -> int:\n    """Does the thing."""\n    return x\n',
        encoding="utf-8",
    )

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please refactor src/docmod.py")

    assert "doc: Does the thing." in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_absolute_path_outside_repo_is_ignored(tmp_path: Path) -> None:
    reset_config()
    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    outside = tmp_path.parent / "outside.py"
    outside.write_text("def outside():\n    return 0\n", encoding="utf-8")

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify(f"Check {outside} and src/ok.py")

    assert "src/ok.py" in res.mentioned_files
    assert all("outside.py" not in p for p in res.mentioned_files)
    assert "outside" not in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_symbol_cap_truncates_deterministically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force a very small symbol cap to test truncation.
    reset_config()
    monkeypatch.setenv("SLM_AGENT_REPO_INTEL__PREFETCH_MAX_SYMBOLS_PER_FILE", "3")
    reset_config()

    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)

    funcs = "\n".join([f"def f{i}():\n    return {i}\n" for i in range(10)])
    (repo_root / "src" / "caps.py").write_text(funcs, encoding="utf-8")

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please check src/caps.py")

    assert "f0" in res.prefetched_stub_map
    assert "f1" in res.prefetched_stub_map
    assert "f2" in res.prefetched_stub_map
    assert "f3" not in res.prefetched_stub_map
