# ChotoVai Startup Helper
# Ensures model service is running and connectivity is healthy before task execution.

param(
    [switch]$SkipOllama,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

function Test-PortOpen {
    param([string]$ComputerName = "127.0.0.1", [int]$Port = 11434, [int]$TimeoutMs = 500)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($ComputerName, $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            $client.EndConnect($async)
            $client.Close()
            return $true
        }
        return $false
    } catch {
        return $false
    }
}

Write-Host "╭─── ChotoVai Startup ───╮" -ForegroundColor Cyan
Write-Host "│                         │" -ForegroundColor Cyan
Write-Host "╰─────────────────────────╯" -ForegroundColor Cyan
Write-Host ""

# ──────────────────────────────────────────────────────────────────────────
# Step 1: Start Ollama service if not already running
# ──────────────────────────────────────────────────────────────────────────

if (-not $SkipOllama) {
    Write-Host "[1] Checking Ollama service..." -ForegroundColor Yellow
    
    $ollamaPath = "C:\Users\HP\AppData\Local\Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaPath)) {
        Write-Host "  ✗ Ollama not found at $ollamaPath" -ForegroundColor Red
        Write-Host "  Install from: https://ollama.ai" -ForegroundColor Gray
        exit 1
    }
    
    # Check if port 11434 is already open (fast socket check)
    if (Test-PortOpen) {
        Write-Host "  ✓ Ollama already running on port 11434" -ForegroundColor Green
    } else {
        Write-Host "  → Starting Ollama service..." -ForegroundColor Cyan
        Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Minimized | Out-Null
        
        # Wait for port to open
        $waited = 0
        $interval = 1
        while (-not (Test-PortOpen) -and $waited -lt $TimeoutSeconds) {
            Write-Host "  ⋯ Waiting for Ollama... ($waited/$TimeoutSeconds s)" -ForegroundColor Gray
            Start-Sleep -Seconds $interval
            $waited += $interval
        }
        
        if (Test-PortOpen) {
            Write-Host "  ✓ Ollama started and ready" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Ollama failed to start within $TimeoutSeconds seconds" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "[1] Skipping Ollama startup (--SkipOllama set)" -ForegroundColor Gray
}

Write-Host ""

# ──────────────────────────────────────────────────────────────────────────
# Step 2: Activate venv and run doctor
# ──────────────────────────────────────────────────────────────────────────

Write-Host "[2] Checking agent connectivity..." -ForegroundColor Yellow

& d:\ChotoVai\.venv\Scripts\Activate.ps1
$doctorResult = & slm-agent doctor 2>&1 | Out-String

Write-Host $doctorResult

# Check if all connections succeeded
if ($doctorResult -match "ConnectError|connection could be made because") {
    Write-Host ""
    Write-Host "  ✗ Connectivity check failed — not all model roles are reachable" -ForegroundColor Red
    Write-Host "  Make sure Ollama is running: ollama serve" -ForegroundColor Gray
    exit 1
} else {
    Write-Host "  ✓ All roles connected successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "[✓] Startup successful! Ready to run tasks." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  • Run a simple task: slm-agent run \"your task here\"" -ForegroundColor Gray
Write-Host "  • Or enter interactive mode: slm-agent repl" -ForegroundColor Gray
Write-Host ""
