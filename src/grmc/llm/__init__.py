"""Optional LLM helpers for report-only reflection enrichment.

Default: fully disabled. Enable via env ``GRMC_LLM=1`` or CLI ``--llm``.
"""

from .config import LLMConfig, llm_enabled_from_env
from .verification import LLMVerifier

__all__ = ["LLMConfig", "LLMVerifier", "llm_enabled_from_env"]
