# Ollama Setup Guide

Local, privacy-first model inference using Ollama.

## Installation

1. Download from https://ollama.ai
2. Run installer and complete setup

## Start Service

```bash
ollama serve
```

The API will be available at `http://localhost:11434/v1`

## Supported Models

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5:14b
ollama pull gemma2:9b
ollama pull llama2:7b
```

## Configuration

Copy to `.env`:
```
ORCHESTRATOR__BASE_URL=http://localhost:11434/v1
ORCHESTRATOR__MODEL_NAME=qwen2.5-coder:14b

CODER__BASE_URL=http://localhost:11434/v1
CODER__MODEL_NAME=qwen2.5-coder:7b

CRITIC__BASE_URL=http://localhost:11434/v1
CRITIC__MODEL_NAME=qwen2.5-coder:7b

TESTER__BASE_URL=http://localhost:11434/v1
TESTER__MODEL_NAME=qwen2.5-coder:7b

EXPLORER__BASE_URL=http://localhost:11434/v1
EXPLORER__MODEL_NAME=qwen2.5-coder:7b

SUMMARIZER__BASE_URL=http://localhost:11434/v1
SUMMARIZER__MODEL_NAME=qwen2.5-coder:7b

# Optional: Escalation (leave empty to skip frontier model escalation)
# ESCALATION__BASE_URL=...
# ESCALATION__API_KEY=...
# ESCALATION__MODEL_NAME=...
```

## Verification

```bash
ollama list
slm-agent doctor
```

Expected output: all roles connected with HTTP 200.

## Performance Tips

- **First model:** Start with smaller models (7B) for development
- **GPU support:** Ollama auto-detects NVIDIA/AMD/Metal GPUs
- **Memory:** Allocate 16GB+ RAM for 14B models running in parallel
- **Context window:** Adjust `max_tokens` per role in config based on GPU memory
