"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# where `corecoder rag` keeps its persisted index when nothing else is given
KB_DIR = Path.home() / ".corecoder" / "kb"


def _load_dotenv():
    """Load .env from cwd, walking up to home dir."""
    from dotenv import load_dotenv

    # search cwd first, then parent dirs up to ~
    env_path = Path("../.env")
    if not env_path.exists():
        cur = Path.cwd()
        home = Path.home()
        while cur != home and cur != cur.parent:
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent
    load_dotenv(env_path, override=False)


def _opt_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"
    fallback_models: list[str] = field(default_factory=list)
    budget_usd: float | None = None
    # sandbox backend for the bash tool: none | docker | bwrap | seccomp
    sandbox: str = "none"
    sandbox_image: str = "python:3.12-slim"
    sandbox_net: bool = False

    # --- RAG 智能客服 -------------------------------------------------------
    # kb_index is the persisted bundle written by `corecoder rag --save` /
    # `python -m corecoder.rag.index_doc`; kb_dir is the default corpus to
    # ingest when no --knowledge/--file is given.
    kb_dir: str = ""
    kb_index: str = str(KB_DIR / "index.json")
    kb_top_k: int = 5
    kb_chunk_size: int = 1000
    kb_chunk_overlap: int = 100
    # retrieval score below this -> refuse instead of letting the LLM improvise.
    # RRF scores top out around 1/60 per ranked list, so ~0.03 means "matched
    # in one list only, at rank 0"; 0.0 disables the gate.
    kb_threshold: float = 0.0
    # hash | openai | litellm. "hash" is the offline default: no key, no network.
    kb_embedding: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-5.5"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=os.getenv("CORECODER_PROVIDER", "openai"),
            sandbox=os.getenv("CORECODER_SANDBOX", "none"),
            sandbox_image=os.getenv("CORECODER_SANDBOX_IMAGE", "python:3.12-slim"),
            sandbox_net=os.getenv("CORECODER_SANDBOX_NET", "0") == "1",
            # 逗号分隔的降级链，例如 "deepseek-chat,gpt-4o-mini"
            fallback_models=[m.strip() for m in os.getenv("CORECODER_FALLBACK_MODELS", "").split(",") if m.strip()],
            # 空字符串/未设置视为"无预算"
            budget_usd=float(os.getenv("CORECODER_BUDGET_USD")) if os.getenv("CORECODER_BUDGET_USD") else None,
            kb_dir=os.getenv("CORECODER_KB_DIR", ""),
            kb_index=os.getenv("CORECODER_KB_INDEX", str(KB_DIR / "index.json")),
            kb_top_k=int(os.getenv("CORECODER_KB_TOP_K", "5")),
            kb_chunk_size=int(os.getenv("CORECODER_KB_CHUNK_SIZE", "1000")),
            kb_chunk_overlap=int(os.getenv("CORECODER_KB_CHUNK_OVERLAP", "100")),
            kb_threshold=_opt_float("CORECODER_KB_THRESHOLD", 0.0) or 0.0,
            kb_embedding=os.getenv("CORECODER_RAG_EMBEDDING", "hash").strip().lower(),
            embedding_model=os.getenv("CORECODER_EMBEDDING_MODEL") or None,
            embedding_base_url=os.getenv("CORECODER_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            embedding_api_key=os.getenv("CORECODER_EMBEDDING_API_KEY") or api_key or None,
        )
