# SLM-First Coding Agent System

This repository contains the reference implementation of an SLM-First (Small Language Model) Coding Agent System as described in the architectural report.

It demonstrates how to scaffold small, open-weight models (like Qwen2.5-Coder 7B) into a robust, deterministic execution loop (TAOR), enabling them to perform complex coding tasks that typically require much larger frontier models.

## Phase Delivery

Currently implemented: **Phase 6 Multi-Path Reasoning**
- Phase 0: Foundation (schemas, config, tracing, vllm client)
- Phase 1: Single-Agent Baseline (BaseAgent, MCP tools: read, write, grep, shell, tests)
- Phase 2: Memory Layer (working buffer, episodic store, plan state, context assembler)
- Phase 3: Orchestrator & FSM (Planner, Deterministic validators, FSM, CLI)
- Phase 4: Multi-Agent Swarm (Specialized Tester, Explorer, Critic, Coder)
- Phase 5: Adversarial SLM Debate (Debate Graph loops, Cognitive Router, Escalation)
- Phase 6: Multi-Path Reasoning (Tree of Thoughts for Intent/Planning, Graph of Thoughts for Debate)

## Getting Started

1. Set up dependencies using `uv` (or pip):
   ```bash
   uv venv
   # Depending on your platform, you might activate the venv here
   uv pip install -e .
   ```

2. Start the infrastructure (Postgres, Qdrant):
   ```bash
   docker-compose -f infra/docker-compose.yml up -d postgres qdrant
   ```

3. Configure your endpoint in `.env`:
   ```bash
   cp .env.example .env
   # Edit .env to point to your local vLLM / Ollama or OpenRouter endpoints
   ```

4. Run the CLI Interactive Mode:
   ```bash
   slm-agent repl
   ```
   Or run a single task:
   ```bash
   slm-agent run "Add a sum_list function to utils.py"
   ```

## Architecture

For a detailed dive into the system design, read the [Architecture Guide](ARCHITECTURE.md).

- `src/core/`: Pydantic schemas, config, and OpenTelemetry tracing
- `src/serving/`: Unifies model inference via OpenAI-compatible endpoints (vLLM support)
- `src/agents/`: Specialist swarm (Coder, Critic, Tester, Explorer, etc.) inheriting from `BaseAgent`.
- `src/memory/`: 5-layer memory (Working, Plan state, Episodic, Semantic, Procedural)
- `src/orchestrator/`: `planner.py`, `fsm.py`, `debate_graph.py`, and `cognitive_router.py` for control flow
- `src/validators/`: Deterministic checks (ruff, pyright, pytest)
- `tools/`: MCP tool implementations
