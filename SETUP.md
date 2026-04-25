# Model Provider Setup Guide

ChotoVai supports multiple model providers through a modular setup system. Each provider is organized in its own folder with provider-specific documentation and configuration templates.

## Quick Start

### 1. List Available Providers
```bash
slm-agent setup --list
```

### 2. View Provider Setup Guide
```bash
slm-agent setup ollama          # Show Ollama setup instructions
slm-agent setup vllm            # Show vLLM setup instructions
slm-agent setup openrouter      # Show OpenRouter setup instructions
slm-agent setup azure_openai    # Show Azure OpenAI setup instructions
```

### 3. Copy Configuration Template
```bash
# Copy to default .env
slm-agent setup ollama --copy-env

# Copy to custom location
slm-agent setup vllm --copy-env --output .env.vllm
```

### 4. Edit Configuration
Edit the `.env` file with your:
- API endpoints (for cloud services)
- API keys (for cloud services)
- Model names (match your deployment)
- Temperature and max_tokens per role

### 5. Verify Setup
```bash
slm-agent doctor
```

Expected output: all roles show HTTP 200 status ✓

## Provider Options

### 1. **Ollama** — Local, Privacy-First
- **Best for:** Development, privacy-conscious deployments, GPU testing
- **Setup time:** 5 minutes
- **Cost:** Free
- **Key features:** 
  - Runs entirely locally
  - No API keys needed
  - GPU auto-detection (NVIDIA, AMD, Metal)
  - Multiple models available

**Quick setup:**
```bash
slm-agent setup ollama --copy-env
# Then: ollama serve (in separate terminal)
```

**Folder:** [presets/ollama/](presets/ollama/)

---

### 2. **vLLM** — High-Performance
- **Best for:** Production inference, multi-GPU setups, performance optimization
- **Setup time:** 15 minutes
- **Cost:** Compute costs only (self-hosted)
- **Key features:**
  - High throughput
  - Tensor parallelism for large models
  - Multi-GPU support
  - OpenAI-compatible API

**Quick setup:**
```bash
pip install vllm
slm-agent setup vllm --copy-env
# Then: vllm serve meta-llama/Llama-3-70b-chat-hf --host 0.0.0.0 --port 8000
```

**Folder:** [presets/vllm/](presets/vllm/)

---

### 3. **OpenRouter** — Cloud API
- **Best for:** Easy cloud access, model variety, zero infrastructure
- **Setup time:** 5 minutes (after getting API key)
- **Cost:** Pay-per-token (varies by model)
- **Key features:**
  - Access to 50+ models
  - No infrastructure needed
  - Automatic failover
  - Per-request pricing

**Quick setup:**
```bash
# 1. Get API key from https://openrouter.ai
slm-agent setup openrouter --copy-env
# 2. Edit .env.openrouter with your API key
```

**Folder:** [presets/openrouter/](presets/openrouter/)

---

### 4. **Azure OpenAI** — Enterprise
- **Best for:** Enterprise deployments, compliance requirements, Azure integration
- **Setup time:** 20 minutes (including resource setup)
- **Cost:** Azure pricing
- **Key features:**
  - Enterprise security
  - Azure Monitor integration
  - Managed scaling
  - Compliance certifications

**Quick setup:**
```bash
# 1. Create Azure OpenAI resource
# 2. Deploy models (gpt-4, gpt-35-turbo, etc.)
slm-agent setup azure_openai --copy-env
# 3. Edit .env with endpoint and API key
```

**Folder:** [presets/azure_openai/](presets/azure_openai/)

---

## Provider Comparison

| Feature | Ollama | vLLM | OpenRouter | Azure OpenAI |
|---------|--------|------|-----------|-------------|
| Cost | Free | Compute | Pay-per-token | Azure pricing |
| Setup Time | 5 min | 15 min | 5 min | 20 min |
| GPU Support | Auto | Multi-GPU | N/A | N/A |
| Models | 10+ | Any HF | 50+ | GPT-4, GPT-3.5 |
| Infrastructure | Local | Self-hosted | Cloud | Azure |
| Best For | Dev | Prod | Cloud | Enterprise |

---

## Configuration Files

Each provider folder contains:

1. **config.md** — Detailed setup guide with:
   - Installation instructions
   - Service startup commands
   - Supported models
   - Performance tuning tips

2. **provider.env** — Configuration template with:
   - Base URL for each role
   - Model names per role
   - Temperature and max_tokens settings
   - Optional API keys

Example structure:
```
presets/
├── ollama/
│   ├── config.md          # Setup guide
│   └── ollama.env         # Config template
├── vllm/
│   ├── config.md
│   └── vllm.env
├── openrouter/
│   ├── config.md
│   └── openrouter.env
└── azure_openai/
    ├── config.md
    └── azure_openai.env
```

---

## Mixing Providers (Advanced)

You can mix providers for different roles by editing `.env`:

```bash
# Use Ollama for fast roles, OpenRouter for complex reasoning
ORCHESTRATOR__BASE_URL=https://openrouter.ai/api/v1
ORCHESTRATOR__API_KEY=sk-or-...
ORCHESTRATOR__MODEL_NAME=openai/gpt-4

CODER__BASE_URL=http://localhost:11434/v1
CODER__MODEL_NAME=qwen2.5-coder:7b

CRITIC__BASE_URL=https://openrouter.ai/api/v1
CRITIC__API_KEY=sk-or-...
CRITIC__MODEL_NAME=openai/gpt-3.5-turbo
```

---

## Troubleshooting

### Port Already in Use
```bash
# Ollama default: 11434
# vLLM default: 8000
netstat -ano | findstr :11434  # Windows
lsof -i :11434                 # macOS/Linux
```

### Model Not Found
- **Ollama:** Run `ollama pull model-name`
- **vLLM:** Model must exist on HuggingFace
- **OpenRouter/Azure:** Check deployment status

### Connectivity Issues
```bash
slm-agent doctor  # Full diagnostics
```

Check:
- Service is running
- Port is correct
- Firewall allows connection
- API key is valid (for cloud)

---

## Next Steps

1. **Choose a provider** based on your needs (dev vs prod, cloud vs local)
2. **Run setup:** `slm-agent setup <provider> --copy-env`
3. **Edit configuration:** Add credentials/URLs as needed
4. **Verify:** `slm-agent doctor`
5. **Start agent:** `slm-agent run "your task"`

For detailed provider-specific instructions, see each provider's `config.md` file in the [presets/](presets/) folder.
