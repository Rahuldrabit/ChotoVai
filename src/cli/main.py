"""
CLI entry point - Rich-based TUI for the SLM coding agent.
"""
from __future__ import annotations

import asyncio
import os
from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.status import Status
from rich.theme import Theme

from src.core.config import get_config
from src.core.tracing import setup_tracing

app = typer.Typer(name="slm-agent", help="SLM-First Coding Agent System", add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        try:
            v = _pkg_version("slm-agent")
        except _PkgNotFound:
            v = "dev"
        typer.echo(f"slm-agent {v}")
        raise typer.Exit()


@app.callback()
def _app_callback(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit"
    ),
) -> None:
    pass
console = Console(theme=Theme({
    "success": "bold green",
    "error": "bold red",
    "info": "bold cyan",
    "warn": "bold yellow",
}))

def _print_tool_trace(tools: list[dict] | None) -> None:
    if not tools:
        return
    console.print("[dim]Tools:[/dim]")
    for t in tools[:20]:
        name = t.get("tool", "?")
        is_error = bool(t.get("is_error"))
        args = t.get("arguments", {})
        style = "error" if is_error else "dim"
        console.print(f"  [{style}]- {name} {args}[/]")

def _print_verify_trace(items: list[dict] | None) -> None:
    if not items:
        return
    console.print("[dim]Verify:[/dim]")
    for it in items[:10]:
        v = it.get("validator", "?")
        outcome = (it.get("outcome") or "").lower()
        msg = (it.get("message") or "")[:120]
        style = "success" if outcome == "pass" else ("warn" if outcome == "uncertain" else "error")
        console.print(f"  [{style}]- {v}: {outcome}[/] [dim]{msg}[/dim]")


def _setup() -> None:
    cfg = get_config()
    obs = cfg.observability
    if obs.enable_tracing:
        setup_tracing(
            service_name=cfg.project_name,
            enable_langfuse=bool(obs.langfuse_public_key),
            langfuse_public_key=obs.langfuse_public_key,
            langfuse_secret_key=obs.langfuse_secret_key,
            langfuse_host=obs.langfuse_host,
            enable_console=(cfg.environment == "development"),
        )


@app.command()
def run(
    goal: str = typer.Argument(..., help="The coding task to accomplish"),
    cwd: str = typer.Option(None, help="Working directory (defaults to current shell directory)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed event stream"),
) -> None:
    """Run the SLM coding agent on a goal."""
    _setup()
    resolved_cwd = Path(cwd).resolve() if cwd else Path.cwd()
    asyncio.run(_async_run(goal, resolved_cwd, verbose))


async def _async_run(goal: str, cwd: Path, verbose: bool) -> None:
    from src.orchestrator.fsm import AgentFSM

    cfg = get_config()
    fsm = AgentFSM(data_dir=cfg.data_dir)

    console.print(Panel(f"[bold]{goal}[/bold]", title="[info]SLM Agent[/info]", border_style="cyan"))

    with Status("[info]Starting...[/info]", console=console, spinner="dots") as status:
        async for event in fsm.run(goal):
            etype = event.get("event", "")

            if etype == "routing":
                tier = event.get("complexity", "?")
                icons = {"trivial": "[*]", "moderate": "[~]", "complex": "[+]"}
                icon = icons.get(tier, "[?]")
                status.update(f"[info]{icon} Complexity:[/info] {tier}")

            elif etype == "planning":
                status.update("[info]Planning task DAG...[/info]")

            elif etype == "plan_ready":
                status.stop()
                console.print("\n[info]Plan:[/info]")
                console.print(Markdown(event["plan_markdown"]))
                status.start()
                status.update("[info]Executing...[/info]")

            elif etype == "node_start":
                status.update(f"[info]Executing:[/info] {event['title']} ({event['node_id']})")

            elif etype == "node_complete":
                console.print(f"[success]OK[/success] {event['node_id']}: {event['summary'][:80]}")
                if event.get("files"):
                    for f in event["files"]:
                        console.print(f"  [dim]-> {f}[/dim]")
                if verbose:
                    _print_tool_trace(event.get("tools"))
                    _print_verify_trace(event.get("verify"))

            elif etype == "node_failed":
                console.print(f"[error]FAIL[/error] {event['node_id']}: {event['error'][:100]}")

            elif etype == "node_retry":
                console.print(f"[warn]~>[/warn] Retrying {event['node_id']} (attempt {event['attempt']})")

            elif etype == "complete":
                status.stop()
                console.print(Panel(
                    f"[success]Done in {event['duration_s']}s - "
                    f"{event['nodes_completed']} nodes, {event['tokens_used']:,} tokens[/success]",
                    border_style="green",
                ))

            elif etype == "failed":
                status.stop()
                errors = "\n".join(event.get("errors", [event.get("reason", "unknown")]))
                console.print(Panel(f"[error]Task failed[/error]\n{errors}", border_style="red"))

            elif verbose:
                console.print(f"[dim]{event}[/dim]")


@app.command()
def repl(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed event stream"),
) -> None:
    """Interactive REPL - enter goals one at a time. Supports cd, /help, /cwd."""
    _setup()
    cwd = Path.cwd()

    console.print(Panel(
        "[bold cyan]SLM Agent REPL[/bold cyan]\n"
        "Type a coding goal, or use special commands:\n"
        "  [dim]cd <path>[/dim]      change working directory\n"
        "  [dim]/cwd[/dim]           show current directory\n"
        "  [dim]/provider[/dim]      show active model config\n"
        "  [dim]/help[/dim]          show this help\n"
        "  [dim]exit[/dim]           quit",
        border_style="cyan",
    ))

    while True:
        # Show last 2 path components to keep prompt compact
        parts = cwd.parts
        short = str(Path(*parts[-2:])) if len(parts) >= 2 else str(cwd)
        try:
            goal = console.input(f"\n[cyan]>[/cyan] [dim]({short})[/dim] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not goal:
            continue

        if goal.lower() in ("exit", "quit", "q"):
            break

        # ── cd command ────────────────────────────────────────────────────
        if goal.lower().startswith("cd "):
            target = goal[3:].strip()
            new_path = Path(target) if Path(target).is_absolute() else (cwd / target)
            new_path = new_path.resolve()
            if new_path.is_dir():
                cwd = new_path
                console.print(f"[info]cwd:[/info] {cwd}")
            else:
                console.print(f"[error]Not a directory: {new_path}[/error]")
            continue

        # ── /cwd ──────────────────────────────────────────────────────────
        if goal.lower() == "/cwd":
            console.print(f"[info]cwd:[/info] {cwd}")
            continue

        # ── /help ─────────────────────────────────────────────────────────
        if goal.lower() == "/help":
            console.print(
                "  [cyan]cd <path>[/cyan]     change working directory\n"
                "  [cyan]/cwd[/cyan]          show current directory\n"
                "  [cyan]/provider[/cyan]     show active model config\n"
                "  [cyan]exit[/cyan]          quit"
            )
            continue

        # ── /provider ─────────────────────────────────────────────────────
        if goal.lower() == "/provider":
            cfg = get_config()
            console.print(f"  orchestrator: [cyan]{cfg.models.orchestrator.base_url}[/cyan] / {cfg.models.orchestrator.model_name}")
            console.print(f"  coder:        [cyan]{cfg.models.coder.base_url}[/cyan] / {cfg.models.coder.model_name}")
            console.print(f"  critic:       [cyan]{cfg.models.critic.base_url}[/cyan] / {cfg.models.critic.model_name}")
            continue

        # ── Normal goal ───────────────────────────────────────────────────
        asyncio.run(_async_run(goal, cwd, verbose=verbose))

    console.print("Goodbye.")


def _do_copy_env(
    provider: str,
    env_path: Path,
    api_key: str | None,
    copy_env_fn,
) -> None:
    """Copy provider .env template and optionally patch in an API key."""
    try:
        result = copy_env_fn(provider, env_path)
        console.print(f"[success]OK[/success] Copied .env template to [cyan]{result}[/cyan]")
        if api_key and provider == "openrouter":
            import re
            text = env_path.read_text(encoding="utf-8")
            text = re.sub(r"sk-or-[A-Z_]+", api_key, text)
            env_path.write_text(text, encoding="utf-8")
            console.print("[success]OK[/success] API key written to .env")
    except Exception as e:
        console.print(f"[error]Failed to copy .env: {e}[/error]")
        raise typer.Exit(1)


@app.command()
def init() -> None:
    """First-run wizard: pick a provider, copy .env template, run doctor."""
    from src.cli.setup import (
        copy_env_template,
        verify_provider_connectivity,
        PROVIDERS,
    )

    console.print(Panel(
        "[bold cyan]Welcome to ChotoVai SLM Agent[/bold cyan]\n\n"
        "This wizard will configure your model provider.\n"
        "It takes about 2 minutes.",
        title="[info]slm-agent init[/info]",
        border_style="cyan",
    ))

    # ── Provider selection ────────────────────────────────────────────────
    provider_list = list(PROVIDERS.keys())
    console.print("\n[info]Available providers:[/info]")
    for i, name in enumerate(provider_list, 1):
        info = PROVIDERS[name]
        console.print(f"  [bold]{i}.[/bold] [cyan]{name}[/cyan] — {info['name']}")

    choice = typer.prompt("\nEnter provider name or number", default="lmstudio")

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(provider_list):
            provider = provider_list[idx]
        else:
            console.print(f"[error]Invalid number: {choice}[/error]")
            raise typer.Exit(1)
    elif choice in PROVIDERS:
        provider = choice
    else:
        console.print(f"[error]Unknown provider: {choice}[/error]")
        console.print(f"[info]Available:[/info] {', '.join(PROVIDERS.keys())}")
        raise typer.Exit(1)

    console.print(f"\n[success]Selected:[/success] {PROVIDERS[provider]['name']}")

    # ── API key prompt (cloud providers only) ────────────────────────────
    api_key: str | None = None
    if provider == "openrouter":
        api_key = typer.prompt("Enter your OpenRouter API key (sk-or-...)", hide_input=True)

    # ── Copy .env template ────────────────────────────────────────────────
    env_path = Path(".env")
    if env_path.exists():
        overwrite = typer.confirm(
            f"\n.env already exists at {env_path.resolve()}. Overwrite?",
            default=False,
        )
        if not overwrite:
            console.print("[info]Keeping existing .env — skipping copy.[/info]")
        else:
            _do_copy_env(provider, env_path, api_key, copy_env_template)
    else:
        _do_copy_env(provider, env_path, api_key, copy_env_template)

    # ── Next steps ────────────────────────────────────────────────────────
    service_cmd = PROVIDERS[provider]["service_cmd"]
    docs = PROVIDERS[provider]["docs"]
    console.print(Panel(
        f"[bold]Next steps:[/bold]\n\n"
        f"1. Start your model server:\n"
        f"   [cyan]{service_cmd}[/cyan]\n\n"
        f"2. Review [cyan].env[/cyan] and adjust model names if needed\n\n"
        f"3. Start coding:\n"
        f"   [cyan]slm-agent repl[/cyan]\n\n"
        f"Docs: [dim]{docs}[/dim]",
        title="[success]Setup complete[/success]",
        border_style="green",
    ))

    # ── Quick connectivity check ──────────────────────────────────────────
    console.print("\n[info]Checking connectivity...[/info]")
    if verify_provider_connectivity(provider):
        console.print(f"[success]OK[/success] {provider} is reachable — you're ready to go!")
    else:
        console.print(
            f"[warn]WARN[/warn] {provider} not reachable yet.\n"
            f"  Start the server first, then run [cyan]slm-agent doctor[/cyan] to verify."
        )


@app.command()
def setup(
    provider: str = typer.Argument(None, help="Model provider (ollama, vllm, openrouter, azure_openai)"),
    list_providers: bool = typer.Option(False, "--list", "-l", help="List available providers"),
    copy_env: bool = typer.Option(False, "--copy-env", help="Copy .env template for this provider"),
    output: str = typer.Option(".env", "--output", "-o", help="Output .env path"),
) -> None:
    """Configure API and model providers."""
    from src.cli.setup import (
        list_providers as list_providers_fn,
        print_setup_instructions,
        copy_env_template,
        verify_provider_connectivity,
        PROVIDERS,
    )
    
    if list_providers or not provider:
        console.print(Panel("[bold]Available Model Providers[/bold]", border_style="cyan"))
        providers = list_providers_fn()
        for name, info in providers.items():
            console.print(f"\n[bold cyan]{name}[/bold cyan]: {info['name']}")
            console.print(f"  [dim]{info['service_cmd']}[/dim]")
            console.print(f"  [dim]Docs: {info['docs']}[/dim]")
        
        console.print(f"\n[info]Usage:[/info] slm-agent setup <provider> [--copy-env]")
        return
    
    if provider not in PROVIDERS:
        console.print(f"[error]Unknown provider: {provider}[/error]")
        console.print(f"[info]Available:[/info] {', '.join(PROVIDERS.keys())}")
        return
    
    # Show setup instructions
    print_setup_instructions(provider)
    
    # Optionally copy .env
    if copy_env:
        try:
            env_path = copy_env_template(provider, output)
            console.print(f"\n[success]OK Copied .env template to {env_path}[/success]")
            console.print(f"[info]Next:[/info] Edit [cyan]{env_path}[/cyan] with your credentials/URLs")
        except Exception as e:
            console.print(f"[error]Failed to copy .env: {e}[/error]")
        
        # Try to verify connectivity
        console.print()
        console.print("[info]Checking connectivity...[/info]")
        if verify_provider_connectivity(provider):
            console.print(f"[success]OK {provider} is reachable[/success]")
        else:
            console.print(f"[warn]WARN {provider} not reachable (may not be running yet)[/warn]")


@app.command()
def doctor() -> None:
    """Check local configuration and connectivity (Codex-like setup sanity check)."""
    _setup()
    from src.core.config import get_config
    import httpx

    cfg = get_config()
    console.print(Panel("Environment sanity check", title="[info]doctor[/info]", border_style="cyan"))

    # venv hint
    import sys
    console.print(f"[info]Python:[/info] {sys.executable}")
    console.print(f"[info]Data dir:[/info] {cfg.data_dir}")

    # Model endpoints
    endpoints = {
        "coder": cfg.models.coder,
        "orchestrator": cfg.models.orchestrator,
        "critic": cfg.models.critic,
    }
    for name, ep in endpoints.items():
        console.print(f"\n[info]{name}[/info]")
        console.print(f"  base_url: {ep.base_url}")
        console.print(f"  model:    {ep.model_name}")
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(ep.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {ep.api_key}"})
                console.print(f"  /models:  {r.status_code}")
        except Exception as e:
            console.print(f"  [warn]connectivity:[/warn] {type(e).__name__}: {e}")

    console.print(
        "\n[info]Approval mode[/info]: "
        + (os.getenv("SLM_AGENT_APPROVAL_MODE", "scoped_auto"))
    )


if __name__ == "__main__":
    app()
