from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestrator.task_router import TaskRouter


@pytest.mark.asyncio
async def test_prefetcher_extracts_stub_map_for_mentioned_file(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)
    target = repo_root / "src" / "mod.py"
    target.write_text(
        """# module doc\n\n# foo does bar\n\nclass Foo:\n    def m(self, x: int) -> int:\n        return x\n\n\ndef my_func(a: int, b: int) -> int:\n    return a + b\n""",
        encoding="utf-8",
    )

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please check src/mod.py and update my_func.")

    assert "src/mod.py" in res.mentioned_files
    assert res.prefetched_stub_map
    assert "##" in res.prefetched_stub_map
    assert "my_func" in res.prefetched_stub_map
    assert "class Foo" in res.prefetched_stub_map or "Foo" in res.prefetched_stub_map


@pytest.mark.asyncio
async def test_prefetcher_blocks_path_traversal(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Look at ../secrets.py and src/ok.py")

    assert "src/ok.py" in res.mentioned_files
    assert all(".." not in p for p in res.mentioned_files)


@pytest.mark.asyncio
async def test_prefetcher_expands_directory_mentions(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "src" / "pkg").mkdir(parents=True)
    (repo_root / "src" / "pkg" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (repo_root / "src" / "pkg" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    router = TaskRouter(client=None, repo_root=repo_root)
    res = await router.classify("Please scan src/pkg/ and update a and b")

    assert "src/pkg/a.py" in res.mentioned_files
    assert "src/pkg/b.py" in res.mentioned_files
    assert "Prefetched File Stubs" in res.prefetched_stub_map
