"""
Setup module — configure API and model providers from presets.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

# Supported providers
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local, Privacy-First)",
        "service_cmd": "ollama serve",
        "status_check": "http://localhost:11434/v1/models",
        "docs": "https://ollama.ai",
    },
    "vllm": {
        "name": "vLLM (High-Performance)",
        "service_cmd": "vllm serve meta-llama/Llama-3-70b-chat-hf --host 0.0.0.0 --port 8000",
        "status_check": "http://localhost:8000/v1/models",
        "docs": "https://github.com/vllm-project/vllm",
    },
    "openrouter": {
        "name": "OpenRouter (Cloud API)",
        "service_cmd": "N/A (managed service)",
        "status_check": "https://openrouter.ai/api/v1/models",
        "docs": "https://openrouter.ai",
    },
    "azure_openai": {
        "name": "Azure OpenAI (Enterprise)",
        "service_cmd": "N/A (managed service)",
        "status_check": "Azure Portal",
        "docs": "https://azure.microsoft.com/en-us/products/ai-services/openai-service/",
    },
}

def get_preset_path(provider: str) -> Path:
    """Get the preset directory for a provider."""
    # Find project root by traversing up from this file
    current = Path(__file__)
    while current.parent != current:
        if (current / "presets").exists():
            preset_dir = current / "presets" / provider
            break
        current = current.parent
    else:
        raise ValueError(f"Could not find project root with 'presets' directory")
    
    if not preset_dir.exists():
        raise ValueError(f"Provider '{provider}' not found. Available: {', '.join(PROVIDERS.keys())}")
    return preset_dir

def list_providers() -> dict:
    """List all available providers."""
    return PROVIDERS

def get_provider_config_path(provider: str) -> Path:
    """Get the config.md path for a provider."""
    return get_preset_path(provider) / "config.md"

def get_provider_env_path(provider: str) -> Path:
    """Get the .env template path for a provider."""
    preset_path = get_preset_path(provider)
    # Look for provider-specific .env file first (e.g., ollama.env)
    specific_env = preset_path / f"{provider}.env"
    if specific_env.exists():
        return specific_env
    # Fallback: look for any .env file
    env_files = sorted(preset_path.glob("*.env"))  # Sort for deterministic selection
    if not env_files:
        raise ValueError(f"No .env template found for provider '{provider}'")
    # Prefer the first alphabetically (should be provider.env)
    return env_files[0]

def copy_env_template(provider: str, target_path: str | Path = ".env") -> Path:
    """Copy provider .env template to target location."""
    target = Path(target_path)
    source = get_provider_env_path(provider)
    
    if target.exists():
        logger.warning("setup.env_exists", path=str(target))
        # Backup existing
        backup = target.with_suffix(f".{provider}.backup")
        shutil.copy(target, backup)
        logger.info("setup.env_backed_up", backup=str(backup))
    
    shutil.copy(source, target)
    logger.info("setup.env_copied", provider=provider, target=str(target))
    return target

def show_provider_docs(provider: str) -> str:
    """Show documentation for a provider."""
    config_path = get_provider_config_path(provider)
    return config_path.read_text(encoding="utf-8")

def verify_provider_connectivity(provider: str) -> bool:
    """
    Verify connectivity to provider backend.
    Returns True if reachable, False otherwise.
    """
    import requests
    
    provider_info = PROVIDERS.get(provider)
    if not provider_info:
        return False
    
    status_url = provider_info.get("status_check")
    if not status_url or status_url.startswith("N/A"):
        logger.info("setup.provider_no_healthcheck", provider=provider)
        return True  # Managed services — skip health check
    
    try:
        if status_url.startswith("http"):
            response = requests.get(status_url, timeout=5)
            is_healthy = response.status_code in (200, 401)  # 401 = auth required, 200 = ok
            logger.info("setup.connectivity_check", provider=provider, healthy=is_healthy, status=response.status_code)
            return is_healthy
        else:
            logger.info("setup.provider_manual_check", provider=provider, url=status_url)
            return True  # Manual check required
    except Exception as e:
        logger.warning("setup.connectivity_failed", provider=provider, error=str(e))
        return False

def print_setup_instructions(provider: str) -> None:
    """Print setup instructions for a provider."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    
    console = Console()
    
    info = PROVIDERS[provider]
    docs = show_provider_docs(provider)
    
    console.print(Panel(
        f"[bold cyan]{info['name']}[/bold cyan]",
        title="[info]Setup Guide[/info]",
        border_style="cyan"
    ))
    
    console.print(Markdown(docs))
    
    console.print(Panel(
        f"[bold]Next steps:[/bold]\n"
        f"1. Copy env template: [cyan]slm-agent setup {provider} --copy-env[/cyan]\n"
        f"2. Edit [cyan].env[/cyan] with your credentials\n"
        f"3. Verify: [cyan]slm-agent doctor[/cyan]",
        border_style="green"
    ))
