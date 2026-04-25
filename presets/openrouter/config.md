# OpenRouter Setup Guide

Cloud-based API aggregating multiple model providers.

## Get API Key

1. Sign up at https://openrouter.ai
2. Generate API key from dashboard
3. Add credits/billing

## Available Models

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Mistral, Cohere
- Open-weight models hosted

## Configuration

Copy to `.env`:
```
OPENROUTER__API_KEY=sk-or-...

ORCHESTRATOR__BASE_URL=https://openrouter.ai/api/v1
ORCHESTRATOR__API_KEY=${OPENROUTER__API_KEY}
ORCHESTRATOR__MODEL_NAME=openai/gpt-4-turbo-preview

CODER__BASE_URL=https://openrouter.ai/api/v1
CODER__API_KEY=${OPENROUTER__API_KEY}
CODER__MODEL_NAME=openai/gpt-3.5-turbo

# ... other roles
```

## Monitoring Costs

- Track usage at https://openrouter.ai/activity
- Set spending limits in account settings
- Monitor model pricing: https://openrouter.ai/models

## Best Practices

- Use cheaper models for simple tasks
- Reserve GPT-4 for complex reasoning
- Cache results when possible
