# vLLM Setup Guide

High-performance LLM serving via vLLM with support for multiple backends.

## Installation

```bash
pip install vllm
```

## Start Server

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

The API will be available at `http://localhost:8000/v1`

## Supported Models

- Qwen2.5-Coder (7B, 14B)
- Llama-2, Llama-3 (7B, 70B)
- Mistral, Mixtral
- Deepseek Coder

## Configuration

Copy to `.env`:
```
ORCHESTRATOR__BASE_URL=http://localhost:8000/v1
ORCHESTRATOR__MODEL_NAME=meta-llama/Llama-3-70b-chat-hf

CODER__BASE_URL=http://localhost:8000/v1
CODER__MODEL_NAME=meta-llama/Llama-3-8b-chat-hf

# ... other roles
```

## Multi-GPU Setup

```bash
vllm serve meta-llama/Llama-2-70b \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2
```

## Performance Tips

- **GPU allocation:** Use `--gpu-memory-utilization 0.85` for stability
- **Batch size:** Larger models benefit from increased `--max-num-seqs`
- **Quantization:** Use `--quantization awq` for 4-bit models
