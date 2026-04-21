"""
Core configuration — layered loading:
  1. Defaults hardcoded here
  2. config.yaml (optional, project-level)
  3. .env file (optional, local overrides)
  4. Environment variables (highest priority)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpoint(BaseSettings):
    """A single model endpoint binding."""

    model_config = SettingsConfigDict(extra="ignore")

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-required"  # vLLM local; override for cloud APIs
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout_s: float = 120.0


class ModelRegistry(BaseSettings):
    """Per-role model endpoint configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    orchestrator: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        temperature=0.1,
        max_tokens=8192,
    ))
    coder: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.2,
    ))
    tester: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.1,
    ))
    critic: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="google/gemma-2-9b-it",  # Different family → less judge drift
        temperature=0.0,
    ))
    summarizer: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="microsoft/Phi-4-mini-instruct",
        temperature=0.1,
        max_tokens=1024,
    ))
    explorer: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.0,
    ))
    escalation: ModelEndpoint = Field(default_factory=lambda: ModelEndpoint(
        base_url="https://api.anthropic.com/v1",
        api_key="",  # Must be set via env ESCALATION__API_KEY
        model_name="claude-opus-4-5",
        temperature=0.2,
        max_tokens=16384,
    ))


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # Vector DB (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_episodic: str = "episodic_memory"
    qdrant_collection_code: str = "code_index"

    # Relational DB (Postgres)
    postgres_dsn: str = "postgresql+asyncpg://slm:slm@localhost:5432/slm_agent"

    # Working memory
    working_memory_compaction_threshold: float = 0.92  # % of context window
    working_memory_max_tokens: int = 6000

    # Episodic
    episodic_top_k: int = 3
    episodic_retention_days: int = 90

    # Embedding model (runs locally)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Scratchpad injection/compaction (Janitor)
    scratchpad_tail_max_chars: int = 1500
    scratchpad_max_chars_before_compact: int = 50_000
    scratchpad_keep_recent_chars: int = 12_000
    scratchpad_summary_target_chars: int = 3500

    # Recursive summarization (Tracker)
    tracker_trigger_chars: int = 20_000
    tracker_chunk_chars: int = 6000
    tracker_target_chars: int = 6000

    # Externality guardrail: if True, only ORCHESTRATOR sees scratchpad/episodic.
    # Specialists still receive contracts_context and current-node working context.
    contracts_only_externality: bool = True


class RepoIntelConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    graph_db_path: str = "./data/code_graph.kuzu"  # KuzuDB embedded
    graph_db_backend: Literal["kuzu", "neo4j"] = "kuzu"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # Indexing
    supported_languages: list[str] = ["python", "javascript", "typescript", "go", "rust", "java"]
    max_file_size_kb: int = 500  # Skip files larger than this
    index_cache_dir: str = "./data/repo_index"

    # Stub-only prefetch (Context Prefetcher)
    prefetch_max_files: int = 10
    prefetch_max_symbols_per_file: int = 200
    prefetch_max_render_chars: int = 20_000


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    max_iterations: int = 10         # Per planning node
    max_total_tokens: int = 200_000  # Per task
    wall_clock_timeout_s: float = 300.0  # 5 minutes per task
    retry_limit: int = 3
    best_of_n: int = 5              # Candidate generation count
    escalation_threshold: float = 0.6   # Confidence below this → escalate
    escalation_rate_target: float = 0.1  # 10% target for escalation

    # ── Adversarial debate settings ─────────────────────────────────────
    debate_enabled: bool = True          # Feature flag — set False to bypass debate
    debate_max_turns: int = 5            # Maximum Coder↔Critic rounds before deadlock
    debate_acceptance_threshold: int = 9  # Critic score >= this → Coder wins
    # Deadlock resolution: "judge_ensemble" | "escalate" | "best_of_n"
    debate_deadlock_strategy: str = "judge_ensemble"
    # Confidence threshold for judge ensemble: std_dev > this → escalate to frontier
    debate_ensemble_confidence_threshold: float = 0.5
    # Token limit to prevent unbounded TTL loops
    debate_max_token_budget: int = 50_000


class ObservabilityConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = True
    log_level: str = "INFO"


class Config(BaseSettings):
    """Root configuration — loads from environment variables with SLM_AGENT_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="SLM_AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "slm-agent"
    environment: Literal["development", "staging", "production"] = "development"
    data_dir: Path = Path("./data")

    models: ModelRegistry = Field(default_factory=ModelRegistry)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    repo_intel: RepoIntelConfig = Field(default_factory=RepoIntelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="after")
    def ensure_data_dirs(self) -> "Config":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.repo_intel.index_cache_dir).mkdir(parents=True, exist_ok=True)
        return self


# Module-level singleton — import this everywhere
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset singleton — useful in tests."""
    global _config
    _config = None
