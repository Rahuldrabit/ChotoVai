"""
Deterministic validators — lint, type check, and test runner.
These return ground-truth pass/fail and are always run before agentic critics.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import structlog

from src.core.schemas import ValidationOutcome, ValidationResult

logger = structlog.get_logger(__name__)


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 60.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return result.returncode, result.stdout[-4000:], result.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT: exceeded {timeout}s"
    except FileNotFoundError:
        return -1, "", f"ERROR: command not found: {cmd[0]}"


class DeterministicValidator:
    """
    Runs a suite of deterministic checks against a code path.
    All validators are independent — failures in one don't stop others.
    """

    def validate(self, path: str, run_tests: bool = True) -> list[ValidationResult]:
        """Validate existing files on disk."""
        results = []
        results.append(self._run_ruff(path))
        results.append(self._run_pyright(path))
        if run_tests:
            results.append(self._run_pytest(path))
        return results

    def validate_content(self, original_path: str, content: str, run_tests: bool = True) -> list[ValidationResult]:
        """
        Validate an in-memory string by writing it to a temporary file alongside the original.
        Useful for sandboxing parallel GoT coder outputs without corrupting the main codebase.
        """
        import os
        from pathlib import Path

        orig = Path(original_path)

        # Directory inputs are valid: write a temporary file in that directory
        # and run validators on the file instead of the whole directory.
        if orig.is_dir():
            target_dir = orig
            suffix = ".py"
            stem = "_validate_content"
        else:
            target_dir = orig.parent if orig.parent != Path("") else Path(".")
            suffix = orig.suffix or ".py"
            stem = orig.stem or "_validate_content"

        temp_path = target_dir / f"{stem}_tmp_{os.urandom(4).hex()}{suffix}"
        
        try:
            temp_path.write_text(content, encoding="utf-8")
            return self.validate(str(temp_path), run_tests)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _run_ruff(self, path: str) -> ValidationResult:
        t0 = time.perf_counter()
        code, out, err = _run([sys.executable, "-m", "ruff", "check", path, "--output-format=concise"])
        duration_ms = (time.perf_counter() - t0) * 1000
        if "No module named ruff" in (out + err) or "command not found" in err.lower():
            return ValidationResult(
                validator_name="ruff",
                outcome=ValidationOutcome.UNCERTAIN,
                message="ruff not installed — skipping lint",
                duration_ms=duration_ms,
            )
        passed = code == 0
        message = out.strip() or err.strip() or ("All checks passed" if passed else "Lint errors found")
        hint = f"Fix these lint errors:\n{message}" if not passed else None
        return ValidationResult(
            validator_name="ruff",
            outcome=ValidationOutcome.PASS if passed else ValidationOutcome.FAIL,
            message=message[:500],
            correction_hint=hint,
            duration_ms=duration_ms,
        )

    def _run_pyright(self, path: str) -> ValidationResult:
        t0 = time.perf_counter()
        code, out, err = _run([sys.executable, "-m", "pyright", path, "--outputjson"], timeout=120.0)
        duration_ms = (time.perf_counter() - t0) * 1000
        if "No module named pyright" in (out + err) or (code == -1 and "command not found" in err.lower()):
            return ValidationResult(
                validator_name="pyright",
                outcome=ValidationOutcome.UNCERTAIN,
                message="pyright not installed — skipping type check",
                duration_ms=duration_ms,
            )
        passed = code == 0
        message = out[:500] if out else err[:500]
        hint = f"Fix these type errors:\n{message}" if not passed else None
        return ValidationResult(
            validator_name="pyright",
            outcome=ValidationOutcome.PASS if passed else ValidationOutcome.FAIL,
            message=message,
            correction_hint=hint,
            duration_ms=duration_ms,
        )

    def _run_pytest(self, path: str) -> ValidationResult:
        t0 = time.perf_counter()
        code, out, err = _run(
            [sys.executable, "-m", "pytest", path, "-v", "--tb=short", "--no-header"],
            timeout=120.0,
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        if "No module named pytest" in (out + err) or (code == -1 and "command not found" in err.lower()):
            return ValidationResult(
                validator_name="pytest",
                outcome=ValidationOutcome.UNCERTAIN,
                message="pytest not installed — skipping tests",
                duration_ms=duration_ms,
            )
        passed = code == 0
        combined = (out + err).strip()[-1500:]
        hint = f"Fix failing tests:\n{combined}" if not passed else None
        return ValidationResult(
            validator_name="pytest",
            outcome=ValidationOutcome.PASS if passed else ValidationOutcome.FAIL,
            message=combined,
            correction_hint=hint,
            duration_ms=duration_ms,
        )
