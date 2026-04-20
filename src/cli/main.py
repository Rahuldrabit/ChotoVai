"""
CLI entry point — Rich-based TUI for the SLM coding agent.
"""
from __future__ import annotations

import asyncio
import os
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
    cwd: str = typer.Option(".", help="Working directory for file operations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed event stream"),
) -> None:
    """Run the SLM coding agent on a goal."""
    _setup()
    asyncio.run(_async_run(goal, Path(cwd), verbose))


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
                icons = {"trivial": "⚡", "moderate": "📋", "complex": "🔬"}
                icon = icons.get(tier, "•")
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
                console.print(f"[success]✓[/success] {event['node_id']}: {event['summary'][:80]}")
                if event.get("files"):
                    for f in event["files"]:
                        console.print(f"  [dim]→ {f}[/dim]")
                if verbose:
                    _print_tool_trace(event.get("tools"))
                    _print_verify_trace(event.get("verify"))

            elif etype == "node_failed":
                console.print(f"[error]✗[/error] {event['node_id']}: {event['error'][:100]}")

            elif etype == "node_retry":
                console.print(f"[warn]↺[/warn] Retrying {event['node_id']} (attempt {event['attempt']})")

            elif etype == "complete":
                status.stop()
                console.print(Panel(
                    f"[success]✓ Done in {event['duration_s']}s — "
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
def repl() -> None:
    """Interactive REPL — enter goals one at a time."""
    _setup()
    console.print("[info]SLM Agent REPL[/info] (type 'exit' to quit)")
    while True:
        try:
            goal = console.input("\n[cyan]>[/cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if goal.lower() in ("exit", "quit", "q"):
            break
        if not goal:
            continue
        asyncio.run(_async_run(goal, Path("."), verbose=False))

    console.print("Goodbye.")


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
