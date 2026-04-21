from __future__ import annotations

from pathlib import Path

from src.repo_intel.symbol_slicer import SliceConfig, slice_symbols


def test_slice_symbols_python_function_and_class(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(
        """
class Foo:
    def m(self, x: int) -> int:
        y = x + 1
        return y


def my_func(a: int, b: int) -> int:
    c = a + b
    return c
""".lstrip(),
        encoding="utf-8",
    )

    snippets = slice_symbols(p, ["Foo", "my_func"], cfg=SliceConfig(max_snippets_total=8))
    assert len(snippets) >= 2
    joined = "\n".join(s.content for s in snippets)
    assert "class Foo" in joined
    assert "def my_func" in joined
    # Includes bodies (but only for sliced defs)
    assert "return y" in joined
    assert "return c" in joined
