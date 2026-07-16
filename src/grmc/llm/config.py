"""LLM feature-flag configuration (default OFF)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def llm_enabled_from_env() -> bool:
    """Return True only when explicitly enabled via environment."""
    return _truthy(os.environ.get("GRMC_LLM")) or _truthy(
        os.environ.get("GRMC_LLM_ENABLED")
    )


@dataclass
class LLMConfig:
    """Conservative defaults for optional verification calls."""

    enabled: bool = False
    provider: str = "openai_compatible"  # openai_compatible | mock
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_s: float = 30.0
    max_tokens: int = 800
    # Hard caps applied to any LLM-proposed confidences
    concept_conf_cap: float = 0.50
    contradiction_conf_cap: float = 0.35

    @classmethod
    def from_env(cls, *, force_enabled: Optional[bool] = None) -> "LLMConfig":
        enabled = llm_enabled_from_env() if force_enabled is None else force_enabled
        provider = (os.environ.get("GRMC_LLM_PROVIDER") or "openai_compatible").lower()
        base_url = os.environ.get("GRMC_LLM_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        ) or "https://api.openai.com/v1"
        # Prefer project-specific key, then common OpenAI/xAI env names
        api_key = (
            os.environ.get("GRMC_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or ""
        )
        model = (
            os.environ.get("GRMC_LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        # Convenience: xAI console users often set XAI_API_KEY only
        if os.environ.get("XAI_API_KEY") and not os.environ.get("GRMC_LLM_BASE_URL"):
            if "x.ai" not in base_url and not os.environ.get("OPENAI_BASE_URL"):
                base_url = "https://api.x.ai/v1"
                if model == "gpt-4o-mini" and not os.environ.get("GRMC_LLM_MODEL"):
                    model = "grok-2-latest"

        return cls(
            enabled=enabled,
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
        )
