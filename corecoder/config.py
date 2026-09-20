"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass, field


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
        )
