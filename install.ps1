# ── ChotoVai install.ps1 ──────────────────────────────────────────────────────
# Installs slm-agent using uv on Windows (PowerShell 5.1+).
#
# Usage:
#   .\install.ps1              # standard install
#   .\install.ps1 -Dev         # editable install with eval extras
#   .\install.ps1 -SkipUvInstall   # skip automatic uv installation
#
# If you get an execution policy error, run first:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Requirements: Python >= 3.12 on PATH, internet access for uv download.
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$Dev,
    [switch]$SkipUvInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  ChotoVai SLM Agent Installer" -ForegroundColor Cyan
Write-Host "  =============================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check / install uv ────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if ($SkipUvInstall) {
        Write-Host "  ERROR: uv not found. Install from https://astral.sh/uv" -ForegroundColor Red
        exit 1
    }
    Write-Host "  uv not found — installing..." -ForegroundColor Yellow
    try {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    } catch {
        Write-Host "  ERROR: Failed to install uv: $_" -ForegroundColor Red
        Write-Host "  Install manually from https://astral.sh/uv" -ForegroundColor Yellow
        exit 1
    }
    # Refresh PATH for this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "  WARNING: uv installed but not yet on PATH." -ForegroundColor Yellow
        Write-Host "  Re-open PowerShell and run this script again." -ForegroundColor Yellow
        exit 1
    }
}

$uvVersion = uv --version
Write-Host "  uv: $uvVersion" -ForegroundColor Green

# ── 2. Verify Python >= 3.12 ─────────────────────────────────────────────────
$pythonFound = $false
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver -split "\."
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 12)) {
                $pythonVersion = & $cmd --version 2>&1
                Write-Host "  Python: $pythonVersion" -ForegroundColor Green
                $pythonFound = $true
                break
            }
        }
    } catch { }
}

if (-not $pythonFound) {
    Write-Host "  ERROR: Python >= 3.12 not found on PATH." -ForegroundColor Red
    Write-Host "  Download from https://python.org/downloads" -ForegroundColor Yellow
    exit 1
}

# ── 3. Install package ───────────────────────────────────────────────────────
Set-Location $ScriptDir

if ($Dev) {
    Write-Host "  Installing in editable/dev mode..." -ForegroundColor Yellow
    uv pip install -e ".[eval]"
} else {
    Write-Host "  Installing..." -ForegroundColor Yellow
    uv pip install -e .
}

Write-Host "  Installed successfully." -ForegroundColor Green

# ── 4. Verify entry point ────────────────────────────────────────────────────
Write-Host ""
if (Get-Command slm-agent -ErrorAction SilentlyContinue) {
    Write-Host "  slm-agent: ready" -ForegroundColor Green
} else {
    Write-Host "  WARNING: slm-agent not found on PATH." -ForegroundColor Yellow
    Write-Host "  You may need to add uv's Scripts directory to PATH." -ForegroundColor Yellow
    Write-Host "  Try running: uv tool update-shell" -ForegroundColor Gray
    Write-Host "  Then re-open PowerShell." -ForegroundColor Gray
}

# ── 5. Done ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  First time? Run the setup wizard:" -ForegroundColor Cyan
Write-Host "    slm-agent init" -ForegroundColor Gray
Write-Host ""
Write-Host "  Or configure manually:" -ForegroundColor Cyan
Write-Host "    slm-agent setup --list          # see available providers" -ForegroundColor Gray
Write-Host "    slm-agent setup lmstudio --copy-env" -ForegroundColor Gray
Write-Host "    slm-agent doctor                # verify connectivity" -ForegroundColor Gray
Write-Host ""
