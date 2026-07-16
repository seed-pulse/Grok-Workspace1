"""Minimal OpenAI-compatible chat client + mock for tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic offline client used in tests / fallback demos."""

    def __init__(self, canned: Optional[Dict[str, Any]] = None):
        self.canned = canned or {}
        self.calls: List[Dict[str, str]] = []

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> Dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if self.canned:
            return dict(self.canned)
        # Heuristic-ish mock: echo request kind
        if "contradiction" in system.lower() or "contradiction" in user.lower():
            return {
                "verified": [],
                "rejected": [],
                "notes": "mock: no verification applied",
            }
        return {
            "concepts": [],
            "notes": "mock: no concepts extracted",
        }


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> Dict[str, Any]:
        if not self.config.api_key:
            raise LLMError("LLM enabled but no API key (set GRMC_LLM_API_KEY)")

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.model,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Best-effort JSON mode for providers that support it
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self.config.timeout_s) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise LLMError(f"LLM HTTP failure: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {data!r}") from exc

        text = (content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON: {text[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM JSON root must be an object")
        return parsed


def build_client(config: LLMConfig) -> LLMClient:
    if config.provider in {"mock", "test"}:
        return MockLLMClient()
    return OpenAICompatibleClient(config)
