#!/usr/bin/env bash
# ── ChotoVai install.sh ──────────────────────────────────────────────────────
# Installs slm-agent globally using uv.
#
# Usage:
#   bash install.sh          # standard install
#   bash install.sh --dev    # editable install with eval extras
#
# Requirements: Python >= 3.12, internet access for initial uv download.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_MODE=0
[[ "${1:-}" == "--dev" ]] && DEV_MODE=1

echo ""
echo "  ChotoVai SLM Agent Installer"
echo "  ============================="
echo ""

# ── 1. Check / install uv ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "  ERROR: uv install failed. Install manually: https://astral.sh/uv"
        exit 1
    fi
fi

echo "  uv: $(uv --version)"

# ── 2. Verify Python >= 3.12 ─────────────────────────────────────────────────
python_cmd=""
for cmd in python3.12 python3.13 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -gt 3 ]] || [[ "$major" -eq 3 && "$minor" -ge 12 ]]; then
            python_cmd="$cmd"
            break
        fi
    fi
done

if [[ -z "$python_cmd" ]]; then
    echo "  ERROR: Python >= 3.12 not found."
    echo "  Install from https://python.org or via your system package manager."
    exit 1
fi

echo "  Python: $($python_cmd --version)"

# ── 3. Install package ───────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

if [[ $DEV_MODE -eq 1 ]]; then
    echo "  Installing in editable/dev mode..."
    uv pip install -e ".[eval]"
else
    echo "  Installing..."
    uv pip install -e .
fi

# ── 4. Verify entry point ────────────────────────────────────────────────────
echo ""
if command -v slm-agent &>/dev/null; then
    echo "  slm-agent: $(slm-agent --version 2>/dev/null || echo 'installed')"
else
    echo "  WARNING: slm-agent not found on PATH."
    echo "  Add uv's bin directory to your PATH, for example:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "  Then re-open your terminal or run: source ~/.bashrc"
fi

# ── 5. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "  Installation complete!"
echo ""
echo "  First time? Run the setup wizard:"
echo "    slm-agent init"
echo ""
echo "  Or configure manually:"
echo "    slm-agent setup --list        # see available providers"
echo "    slm-agent setup lmstudio --copy-env"
echo "    slm-agent doctor              # verify connectivity"
echo ""
