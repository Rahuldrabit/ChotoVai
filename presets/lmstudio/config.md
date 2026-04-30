# LM Studio Setup Guide

LM Studio is a desktop application that lets you run open-weight LLMs locally with a built-in OpenAI-compatible API server. No Docker, no command line — just download, load a model, and click Start.

---

## 1. Download & Install

Go to **https://lmstudio.ai** and download the installer for your OS (Windows, macOS, Linux).

Run the installer and launch LM Studio.

---

## 2. Download a Model

1. Open LM Studio and click the **Discover** tab (magnifying glass icon in the left sidebar).
2. Search for `qwen2.5-coder`.
3. Download **Qwen2.5-Coder-7B-Instruct** (recommended for most machines).
   - If you have 16+ GB VRAM, try the **14B** variant instead.
   - Use **Q4_K_M** or **Q5_K_M** GGUF quantization for best speed/quality balance.
4. Wait for the download to finish (models are 4–8 GB).

**Alternative models that work well:**
- `llama-3-8b-instruct` — general purpose, slightly less code-focused
- `deepseek-coder-6.7b-instruct` — strong on Python/JS

---

## 3. Load the Model

1. Click the **My Models** tab (house icon).
2. Find your downloaded model and click **Load**.
3. Wait for the green "Model loaded" indicator.

---

## 4. Start the Local Server

1. Click the **Local Server** tab (arrow icon `<->` in the left sidebar).
2. In the model dropdown at the top, select your loaded model.
3. Click **Start Server**.
4. The server starts on `http://localhost:1234/v1` by default.

You'll see the server log on the right. Keep LM Studio open while running ChotoVai.

---

## 5. Configure ChotoVai

Copy the preset to `.env`:

```bash
# Linux / macOS
cp presets/lmstudio/lmstudio.env .env

# Windows (Command Prompt)
copy presets\lmstudio\lmstudio.env .env

# Windows (PowerShell)
Copy-Item presets\lmstudio\lmstudio.env .env
```

Or use the interactive wizard:

```bash
slm-agent init
# Select: lmstudio
```

---

## 6. Key Configuration Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `SLM_AGENT_MODELS__CODER__BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `SLM_AGENT_MODELS__CODER__API_KEY` | `lm-studio` | Placeholder — LM Studio ignores the key |
| `SLM_AGENT_MODELS__CODER__MODEL_NAME` | `qwen2.5-coder-7b-instruct` | Must match the slug shown in LM Studio's server tab |

> **Important:** LM Studio runs **one model at a time**. All agent roles (Coder, Critic, Tester, etc.) will use whichever model is currently loaded, regardless of what `MODEL_NAME` says in the env file. Set all roles to the same model slug.

To find the exact model slug: in LM Studio's Local Server tab, look at the model identifier shown next to the loaded model name (e.g., `qwen2.5-coder-7b-instruct`). Use that string as `MODEL_NAME`.

---

## 7. Verify

```bash
slm-agent doctor
```

Expected output: all three endpoints (`coder`, `orchestrator`, `critic`) show HTTP `200`.

If you see connection refused, make sure LM Studio's Local Server is running and a model is loaded.

---

## Performance Tips

| Setup | VRAM | Speed |
|-------|------|-------|
| Qwen2.5-Coder 7B Q4_K_M | ~5 GB | ~15 tok/s on RTX 3070 |
| Qwen2.5-Coder 7B Q5_K_M | ~6 GB | ~12 tok/s |
| Qwen2.5-Coder 14B Q4_K_M | ~9 GB | ~8 tok/s on RTX 3090 |

- Enable **GPU acceleration** in LM Studio → Settings → GPU → Metal (macOS) or CUDA (Windows/Linux).
- Set **Context Length** to 8192 in LM Studio's model settings for longer coding tasks.
- If responses are slow, lower `SLM_AGENT_AGENT__MAX_ITERATIONS` to `5` in `.env`.

---

## Next Steps

```bash
slm-agent doctor          # verify connectivity
slm-agent repl            # start interactive coding session
slm-agent run "write a fibonacci function in utils.py"
```
