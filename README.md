# ChotoVai — SLM-First Coding Agent System

A multi-agent coding assistant that scaffolds small open-weight models (Qwen2.5-Coder 7B, Llama-3-8B) into a robust, deterministic execution loop capable of handling complex software engineering tasks — without requiring frontier-model access.

## 
Currently implemented:

| Description |
|-------------|
| Foundation — schemas, config, tracing, vLLM client |
| Single-Agent Baseline — BaseAgent, MCP tools (read, write, grep, shell, tests) |
| Memory Layer — working buffer, episodic store, plan state, context assembler |
| Orchestrator & FSM — Planner, deterministic validators, FSM, CLI |
| Multi-Agent Swarm — specialized Tester, Explorer, Critic, Coder, Refactorer |
| Adversarial SLM Debate — debate graph loops, cognitive router, escalation |
| Multi-Path Reasoning — Tree of Thoughts for intent/planning, Graph of Thoughts for debate |
| Dynamic Node Decomposition — broad goals auto-split into atomic subtasks at runtime |
| External Blackboard Memory — session scratchpad + code contracts shared across all agents |

**Also available:** VS Code extension (`chotovai-vscode/`) wrapping the agent system as a sidebar chat panel.

---

## Getting Started

### 1. Install dependencies

```bash
uv venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -e .
```

### 2. Start infrastructure

```bash
docker-compose -f infra/docker-compose.yml up -d postgres qdrant neo4j
```

### 3. Configure endpoints

```bash
cp .env.example .env
# Edit .env — point VLLM_BASE_URL at your local vLLM / Ollama server
# Set OPENROUTER_API_KEY for frontier-model escalation (optional)
```

### 4. Run

```bash
# Interactive REPL
slm-agent repl

# Single task
slm-agent run "Add a retry decorator to utils.py"
```

### VS Code Extension

```bash
cd chotovai-vscode
npm install
npm run compile        # or: node esbuild.js --watch
# Press F5 in VS Code to launch the extension host
```

The extension communicates with `bridge.py` over subprocess NDJSON — no HTTP server required.

---

## Project Layout

```
src/
  core/           Pydantic schemas, config, OpenTelemetry tracing
  serving/        OpenAI-compatible inference client (vLLM / Ollama / OpenRouter)
  agents/         Specialist swarm — Coder, Critic, Tester, Explorer, Refactorer, Summarizer
  memory/         Memory layers: episodic, scratchpad, contracts, context assembler
  orchestrator/   FSM, Planner, CognitiveRouter, DebateGraph, NodeDecomposer, Escalation
  validators/     Deterministic checks (ruff, pyright, pytest) + agentic ensemble
  protocols/      MCP client — tool registry and approval gating
  repo_intel/     Code graph, GraphRAG, community detection
  fine_tuning/    Debate trace collector + imagine trainer (decoupled from orchestration)

tools/            MCP tool implementations (read_file, write_file, grep, shell, git, …)
chotovai-vscode/  VS Code extension — sidebar chat panel + subprocess bridge
bridge.py         NDJSON bridge between VS Code extension and AgentFSM
```

---

## Key Design Decisions

**SLMs, not frontier models** — every agent targets 7–9B parameter models. Reliability comes from deterministic feedback loops, not model scale.

**LLM-first routing with code extraction** — incoming goals are classified (TRIVIAL/MODERATE/COMPLEX) by a fast LLM classifier with fallback heuristics. Code blocks are extracted once and separated from natural language intent, preventing code patterns from confusing classification or intent analysis. Code is preserved as explicit context for downstream reasoning.

**External Blackboard** — a per-session `scratchpad.md` (append-only reasoning log) and `contracts.json` (JSON symbol table) live outside the LLM context window. For smaller SLM runtimes, scratchpad tail injection into agent prompts is **scoped by config** (recommended: orchestrator-only), while contracts are injected for all roles. Agents write back via `scratchpad_append` and `contracts_update`, and roles that need it can query scratchpad via `read_scratchpad`.

**Dynamic Decomposition** — the `CognitiveRouter` detects broad nodes (>80-word description, >3 success criteria, or multi-concept title) and routes them through `NodeDecomposer`, which calls the planner model to split the node into 2–5 atomic child nodes injected live into the running DAG.

**Adversarial Debate** — the Coder and Critic run in a game-theoretic loop until the Critic scores the output ≥ threshold or retries are exhausted. The ensemble validator runs N critics in parallel and escalates to a frontier model when judges disagree.

**Fine-tuning decoupled** — debate traces are collected by `DebateTraceCollector` but the fine-tuning pipeline is intentionally disabled in the orchestration loop so it does not affect latency or correctness of the agent run.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design including component diagrams, memory layer detail, and execution flow walkthrough.
