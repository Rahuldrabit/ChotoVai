# ChotoVai — Architecture Guide

The **SLM-First Coding Agent** system scaffolds small open-weight models (Qwen2.5-Coder 7B, Llama-3-8B) into a robust, deterministic multi-agent loop. Since SLMs have limited context windows and weaker reasoning than frontier models, the architecture compensates with structured external state, deterministic feedback, adversarial debate, and dynamic task decomposition.

---

## System Diagram

```mermaid
graph TD
    User([User Request]) --> Intent[Intent Reasoner / Query Rewriter]
    Intent -->|Clarified goal| Planner[Planner Agent]

    subgraph Orchestration
        Planner -->|TaskDAG| Router[Cognitive Router]
        Router -->|DECOMPOSE| Decomposer[Node Decomposer]
        Decomposer -->|Child nodes injected| Router
        Router -->|DIRECT / VERIFY| SingleShot[Single-Shot Agent]
        Router -->|DEBATE| DebateGraph[Adversarial Debate Graph]
        Router -->|REFINE| Refine[Refine + AgenticValidator]
        Router -->|ESCALATE| Escalation[Frontier Model Escalation]
    end

    subgraph Adversarial Debate Loop
        DebateGraph --> Coder[Coder Agent]
        Coder -->|Code| DetVal[Deterministic Validators\nruff · pyright · pytest]
        DetVal --> Critic[Critic Agent]
        Critic -->|Score + hint| Decision{Score ≥ threshold?}
        Decision -->|No| Coder
        Decision -->|Yes| DebateDone[Return to Router]
    end

    subgraph External Blackboard
        Scratchpad[(scratchpad.md\nappend-only log)]
        Contracts[(contracts.json\nsymbol table)]
    end

    Coder -->|scratchpad_append\ncontracts_update| Scratchpad
    Coder -->|contracts_update| Contracts
    Critic -->|scratchpad_append| Scratchpad
    Explorer[Explorer Agent] -->|scratchpad_append| Scratchpad
    Refactorer[Refactorer Agent] -->|scratchpad_append\ncontracts_update| Scratchpad
    Tester[Tester Agent] -->|scratchpad_append\ncontracts_update| Scratchpad
    AgVal[AgenticValidator] -->|verdict to scratchpad| Scratchpad

    Scratchpad -->|tail injected into ContextPack (scoped by config)| ContextAssembler[Context Assembler]
    Contracts -->|compact summary injected| ContextAssembler
    ContextAssembler --> Coder
    ContextAssembler --> Critic
    ContextAssembler --> Explorer
    ContextAssembler --> Refactorer
    ContextAssembler --> Tester

    subgraph Persistent Memory
        EpisodicStore[(Episodic Store\nvector DB)]
        SemanticStore[(Semantic Store\nQdrant)]
        PlanState[(Plan State\nDAG + node status)]
    end

    ContextAssembler -.-> EpisodicStore
    ContextAssembler -.-> SemanticStore
    Router -.-> PlanState
```

---

## Core Design Principles

1. **Deterministic over probabilistic** — SLMs generate; compilers, linters, and test runners validate.
2. **Adversarial convergence** — Coder and Critic run in a game-theoretic loop; confidence-scored ensemble escalates to a frontier model on disagreement.
3. **External blackboard** — shared state lives outside any single context window. Agents write to a per-session scratchpad and contracts store; scratchpad tail injection into ContextPacks is **scoped by config** (recommended default: orchestrator-only for smaller SLM runtimes), while contracts are injected for all roles.
4. **Dynamic decomposition** — broad nodes are split at runtime into 2–5 atomic child nodes by the `NodeDecomposer`, preserving DAG dependency correctness.
5. **Fine-tuning decoupled** — debate traces are collected passively. The fine-tuning pipeline is intentionally disabled in the orchestration hot path.

---

## Components

### 1. Agent Swarm (`src/agents/`)

All agents inherit `BaseAgent` which implements the **TAOR loop** (Think → Act → Observe → Repeat). Each specialist overrides `system_prompt()`, `allowed_tools()`, and `card()`.

| Agent | Role | Allowed Tools | Blackboard Access |
|-------|------|---------------|-------------------|
| **CoderAgent** | Primary code writer | read_file, write_file, grep, shell, run_tests, scratchpad_append, contracts_update | Write scratchpad + contracts; does not require scratchpad read in scoped mode |
| **CriticAgent** | Code reviewer / SLM-as-Judge | read_file, grep, scratchpad_append, read_scratchpad | Read + write scratchpad |
| **TesterAgent** | Test suite writer | read_file, write_file, grep, run_tests, scratchpad_append, contracts_update, read_scratchpad | Full (read + write) |
| **ExplorerAgent** | Codebase traversal | code_graph, web_search, web_fetch, read_file, grep, scratchpad_append, read_scratchpad | Read + write scratchpad |
| **RefactorerAgent** | Systematic code transformations | read_file, write_file, grep, run_tests, shell, scratchpad_append, contracts_update, read_scratchpad | Full (read + write) |
| **SummarizerAgent** | Context compression | _(none — pure text)_ | Stateless, no blackboard |

**Blackboard collaboration rules:**
- Code-producing agents (Coder, Refactorer, Tester) call `contracts_update` after writing any class or function.
- All agents call `read_scratchpad` at the start of a task to avoid re-exploring what a prior agent already found.
- CriticAgent logs PASS/FAIL + reason to the scratchpad so the next REFINE iteration knows exactly what to fix.

---

### 2. Orchestration (`src/orchestrator/`)

#### FSM (`fsm.py`)
The top-level finite state machine. States: `PLANNING → EXECUTING → [DEBATING | VALIDATING | DECOMPOSING] → COMPLETE / FAILED`.

On session start the FSM creates per-session `scratchpad.md` and `contracts.json` and wires them into the `ContextAssembler` so all subsequent agent calls share the same blackboard.

#### Task Router & Code Extraction (`task_router.py`)
Routes incoming goals to the optimal execution tier (TRIVIAL, MODERATE, COMPLEX) based on semantic complexity.

**Context Prefetcher (stub-only):**
When the user mentions a repo file path in plain text (e.g., `src/orchestrator/fsm.py`), the TaskRouter performs a deterministic prefetch step:
- Resolve the mention to a repo-local file (blocks traversal like `../`)
- Extract stubs only (function/class/method names + short signatures + nearby doc comments)
- Render a compact “stub map” and attach it to the classification result

The Planner receives this stub map as `repo_summary` before DAG generation, so it can plan with codebase structure without injecting full file contents.

**Unified Code Extraction:**
When a user provides code blocks + natural language query (e.g., "refactor this function"), TaskRouter immediately extracts code blocks once and splits the input into:
- `nl_intent`: Natural language query with code blocks removed (for classification)
- `code_snippets`: List of extracted markdown-fenced code blocks (preserved for downstream reasoning)

This prevents code patterns (if statements, async keywords, semicolons) from confusing classification heuristics.

**Classification Pipeline (LLM-first):**
1. Primary: Send `nl_intent` to LLM classifier (max_tokens=5, one-word output: trivial/moderate/complex)
2. Fallback (if model unavailable): Heuristic scoring across 4 dimensions:
   - Verb type (creation/modification/reasoning)
   - Scope width (single item / module / system-wide)
   - Requirement count (logical connectors: AND, THEN, semicolons)
   - Structural complexity flags (if/async/thread/lock keywords)

**Routing Tiers:**
- **TRIVIAL** — simple creation tasks (e.g., "write hello_world in file X") → skip planning, build 1-node DAG
- **MODERATE** — modification + ambiguous intent (e.g., "add retry decorator") → lite intent rewrite (1 model call) + lite planner
- **COMPLEX** — reasoning/refactoring/architectural (e.g., "refactor for scalability") → full Tree of Thoughts intent analysis + full planner

#### Intent Reasoner (`intent_reasoner.py`)
Clarifies user intent and extracts structured task description.

**COMPLEX path:** Tree of Thoughts (ToT) with evaluation
1. Generate 3 distinct interpretations of intent (literal vs defensive vs architectural)
2. Evaluate all 3 with a judge model, select the best
3. Return structured IntentAnalysis (primary_intent, constraints, assumptions, code_snippets, rewritten_query)

**MODERATE path:** Single-call lite rewrite (no ToT selection)
1. Direct call to intent clarifier (no branching/selection overhead)
2. Return same IntentAnalysis schema
3. Used when classification uncertainty is lower

**Code Handling:** Both paths accept pre-extracted `code_snippets` as context. Code is appended to the user message so the model sees both intent and code together while maintaining a clean separation between intent classification and code analysis.

**Tracker (recursive summarization):** For very large user inputs (goal + pasted code), the orchestrator can first run a recursive summarizer to produce a compact, non-verbatim description. This prevents the Planner from receiving large raw code blobs while preserving file paths and symbol names.

#### Planner (`planner.py`)
Uses a **Tree of Thoughts** approach: drafts 3 architectural approaches, scores them, selects the best, then decomposes it into a flat `TaskDAG`. Nodes carry `cognitive_strategy` hints (debate, verify, direct, decompose, escalate, refine).

#### Cognitive Router (`cognitive_router.py`)
Selects the execution strategy for each DAG node in priority order:

1. Explicit `cognitive_strategy` set by Planner
2. Node is too broad (>80-word description, >3 success criteria, or multi-concept title) → **DECOMPOSE**
3. High retry count (near limit) → **ESCALATE**
4. Read-only roles (Explorer, Summarizer) → **DIRECT**
5. Code-producing roles (Coder, Refactorer) + debate enabled → **DEBATE**
6. Tester role → **VERIFY**
7. Critic role → **DIRECT**

#### Node Decomposer (`node_decomposer.py`)
When the router selects DECOMPOSE, the `NodeDecomposer` calls the orchestrator model with a structured prompt to split the broad node into 2–5 atomic child nodes. Children are injected into the live DAG via `TaskDAG.decompose_node()`:

- First child inherits the parent's upstream dependencies
- Each subsequent child depends on the previous one (sequential by default)
- Downstream nodes that depended on the parent are rewired to depend on the last child
- Parent is marked COMPLETE; the main execution loop picks up children naturally on the next iteration

#### Debate Graph (`debate_graph.py`)
Adversarial Coder ↔ Critic loop. The Coder generates code; deterministic validators (ruff, pyright, pytest) run first; results feed into the Critic which scores 0–10. Loop continues until score ≥ threshold or retries exhausted.

#### Escalation (`escalation.py`)
Fallback to a frontier model (via OpenRouter) when the debate stagnates or the judge ensemble disagrees.

---

### 3. External Blackboard Memory (`src/memory/`)

The blackboard solves the core SLM problem: state cannot fit in a single context window across a long decomposed session.

#### Session Scratchpad (`scratchpad.py`)
An **append-only markdown log** stored at `./data/sessions/<session_id>/scratchpad.md`.

Each entry is prefixed with an ISO timestamp, agent role, and plan node ID:
```
---
[2025-04-19T10:23:45Z] [coder] [node:N3a]
Chose recursive DFS over BFS because the graph is sparse and depth-first avoids allocating large frontier queues.
```

Key methods:
- `append(entry, role, node_id)` — thread-safe write
- `read_tail(max_chars)` — O(1) seek-from-end for context injection
- `read_node(node_id)` — retrieve all entries for a specific plan node
- `read_by_role(role, max_chars)` — retrieve entries from a specific agent role

**Janitor compaction (truncate-in-place):**
To hard-enforce bounded context growth across long sessions, the orchestrator can compact the scratchpad when it exceeds a configured size cap:
- Summarize older entries into a single “janitor” summary block
- Keep the most recent tail unchanged

#### Code Contracts Store (`contracts.py`)
A **JSON symbol table** stored at `./data/sessions/<session_id>/contracts.json`.

Each entry is an `EntityContract` (Pydantic-validated) with: name, kind (class/function/module), file path, method signatures, parameter types, return type, dependencies.

`get_all_compact()` returns a ~200-token human-readable summary of all registered entities — injected into every `ContextPack` as the "Code Contracts" section.

#### Context Assembler (`context_assembler.py`)
Assembles a `ContextPack` for each agent call from:
- Current plan node
- Top-3 episodic memories (vector search, filtered to PASS outcomes)
- Semantic rules relevant to the node
- Code snippets from 2-hop graph neighborhood (trimmed to token budget)
- Scratchpad tail (config-bounded tail injection)
- Compact contracts summary

Total: ~2–3K tokens, assembled fresh per call.

#### MCP Tools for the Blackboard

| Tool | Description |
|------|-------------|
| `scratchpad_append` | Write a reasoning entry tagged with role + node ID |
| `contracts_update` | Register or update a class/function interface contract |
| `read_scratchpad` | Query the scratchpad: `recent` tail, `by_role`, or `by_node` |

#### Other Memory Layers
- **Episodic Store** — short-term vector DB of past agent trajectories; top-k retrieved per task
- **Semantic Store** — Qdrant-backed repository knowledge (file summaries, module relationships)
- **Plan State** — current DAG + per-node status, persisted across FSM restarts

---

### 4. Validators (`src/validators/`)

#### Deterministic Validator
Runs `ruff` (linting), `pyright` (type checking), and `pytest` (tests) in subprocess. Binary pass/fail. Output fed directly into the Coder/Critic debate loop.

#### Agentic Validator (`agentic.py`)
Ensemble of N independent `CriticAgent` instances run in parallel (fan-out). Each scores 0–10. Aggregates mean score, standard deviation, and confidence.

- `confidence = 1 - (std_dev / 5.0)` — low confidence means judges disagree
- If `confidence < threshold` → escalate to frontier model to break the tie
- After each verdict, writes `"Validation PASS/FAIL — mean score X/10 — issues: ..."` to the session scratchpad so subsequent REFINE attempts see why the previous attempt failed

---

### 5. Serving Layer (`src/serving/`)

OpenAI-compatible abstraction. Agents call `get_client(role)` which returns a `VllmClient` pointed at the model registered for that role:

| Role | Default Model |
|------|--------------|
| CODER, REFACTORER, TESTER, EXPLORER | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| CRITIC | `google/gemma-2-9b-it` |
| ORCHESTRATOR (Planner, Decomposer) | Same as CODER |
| ESCALATION | Frontier model via OpenRouter |

---

### 6. MCP Tool Layer (`tools/` + `src/protocols/mcp_client.py`)

All agent tool calls go through `execute_tool()` in `mcp_client.py`, which:
1. Looks up the tool in `_TOOL_REGISTRY`
2. Applies the **approval gate** (configurable: `auto`, `scoped_auto`, `interactive`, `deny`)
3. High blast-radius operations (`git_push`, `pr_create`, `write_file` to config files) always prompt in `scoped_auto` mode
4. Returns a `ToolResult` — errors are surfaced to the agent, never crash the loop

**Available tools:**

| Category | Tools |
|----------|-------|
| File I/O | `read_file`, `write_file` |
| Search | `grep` |
| Execution | `shell`, `run_tests` |
| Graph | `code_graph` |
| Web | `web_search`, `web_fetch` |
| VCS | `git_status`, `git_diff`, `git_log`, `git_branch_create`, `git_commit`, `git_push` |
| PR/Issues | `pr_fetch`, `issue_fetch`, `pr_create` |
| Validation | `verify_repo` |
| Blackboard | `scratchpad_append`, `contracts_update`, `read_scratchpad` |

---

### 7. Repository Intelligence (`src/repo_intel/`)

- **Code Graph (`code_graph.py`)** — Kuzu embedded graph DB; stores AST-level function/class/import relationships. Supports: `find_function`, `callers`, `callees`, `module_neighborhood`, `top_modules`
- **GraphRAG (`graph_rag.py`)** — Combines graph traversal with vector search for semantic code retrieval
- **Community Detection (`community.py`)** — Identifies bounded contexts and module clusters in large repos

---

### 8. Fine-Tuning Pipeline (`src/fine_tuning/`)

**Decoupled from orchestration.** The `DebateTraceCollector` and `ImagineTrainer` exist but the collector is disabled in `_execute_debate()` by default — it does not run during agent sessions.

- **DebateTraceCollector** — harvests successful Coder → Critic trajectories as training pairs
- **ImagineTrainer** — generates synthetic tasks from the codebase for RL fine-tuning

To re-enable: uncomment the collector block in `src/orchestrator/fsm.py → _execute_debate()`.

---

### 9. VS Code Extension (`chotovai-vscode/`)

A sidebar chat panel that wraps the Python agent system. Architecture:

```
VS Code UI (WebviewViewProvider)
    ↕ postMessage / onDidReceiveMessage
ChatViewProvider (TypeScript)
    ↕ start/cancel
AgentRunner (TypeScript)
    ↕ subprocess stdin/stdout — NDJSON protocol
bridge.py (Python)
    ↕ async generator
AgentFSM (Python)
```

**Communication protocol:** Each FSM event is serialized as `json.dumps(event) + "\n"` on stdout. Rich TUI output goes to stderr (invisible to the extension). Event types: `plan_ready`, `node_start`, `node_complete`, `tool_call`, `tool_result`, `complete`, `error`.

**Windows process cleanup:** `AgentRunner.cancel()` uses `taskkill /pid <pid> /T /F` as a fallback since `SIGTERM` is unreliable on Windows.

---

## Execution Flow

```
User goal
  → IntentReasoner (ToT): rewrite + clarify
  → Planner (ToT): 3 architectures → best → TaskDAG
  → FSM.run():
      for each PENDING node in DAG:
        CognitiveRouter.select(node):
          DECOMPOSE → NodeDecomposer → inject children → continue
          DIRECT / VERIFY → single agent TAOR pass
          DEBATE → DebateGraph Coder↔Critic loop
          REFINE → agent pass + AgenticValidator ensemble
          ESCALATE → frontier model
        scratchpad.append("node_complete: <summary>")
  → all nodes COMPLETE → emit final answer
```

Every agent call:
1. ContextAssembler builds ContextPack (plan node + episodic memories + scratchpad tail + contracts)
2. BaseAgent.run() executes TAOR loop: model → parse tool call → execute → feed result back
3. Agent optionally writes to blackboard via `scratchpad_append` / `contracts_update`
4. FINAL_ANSWER emitted → AgentResult returned to FSM
