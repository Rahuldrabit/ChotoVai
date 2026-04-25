# Azure OpenAI Setup Guide

Enterprise deployment using Microsoft Azure OpenAI Service.

## Prerequisites

- Azure subscription
- Azure OpenAI resource created
- Models deployed (gpt-4, gpt-35-turbo, etc.)

## Get Credentials

1. Go to Azure Portal → OpenAI resource
2. Click "Keys and Endpoint" 
3. Copy API key and endpoint URL
4. Note deployed model names from "Model deployments"

## Configuration

Copy to `.env`:
```
AZURE_OPENAI__API_KEY=your-key-here
AZURE_OPENAI__ENDPOINT=https://your-resource.openai.azure.com/

ORCHESTRATOR__BASE_URL=${AZURE_OPENAI__ENDPOINT}v1
ORCHESTRATOR__API_KEY=${AZURE_OPENAI__API_KEY}
ORCHESTRATOR__MODEL_NAME=your-gpt4-deployment-name
ORCHESTRATOR__API_VERSION=2024-02-15-preview

CODER__BASE_URL=${AZURE_OPENAI__ENDPOINT}v1
CODER__API_KEY=${AZURE_OPENAI__API_KEY}
CODER__MODEL_NAME=your-gpt35-deployment-name
CODER__API_VERSION=2024-02-15-preview

# ... other roles
```

## Model Deployments

Recommended setup:
- **Orchestrator:** GPT-4-32K (planning, complex reasoning)
- **Coder:** GPT-3.5-Turbo (fast generation)
- **Critic:** GPT-4 (scoring, review)
- **Others:** GPT-3.5-Turbo (cost optimization)

## Quotas & Limits

- Request rate: Check TPM (tokens per minute) limits
- Monitor usage in Azure Portal
- Configure autoscaling for production

## Security

- Use managed identity when possible (in Azure compute)
- Rotate API keys regularly
- Use Azure Key Vault to store secrets
