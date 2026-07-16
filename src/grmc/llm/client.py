"""Minimal OpenAI-compatible chat client + mock for tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    last_usage: Optional[Dict[str, Any]] = None
    last_raw_content: str = ""

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic offline client used in tests / fallback demos."""

    def __init__(self, canned: Optional[Dict[str, Any]] = None):
        self.canned = canned or {}
        self.calls: List[Dict[str, str]] = []
        self.last_usage = None
        self.last_raw_content = ""

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> Dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if self.canned:
            result = dict(self.canned)
        elif "contradiction" in system.lower() or "contradiction" in user.lower():
            result = {
                "items": [],
                "notes": "mock: no verification applied",
            }
        else:
            result = {
                "concepts": [],
                "notes": "mock: no concepts extracted",
            }
        self.last_raw_content = json.dumps(result)
        # Fake small usage for audit tests
        self.last_usage = {
            "prompt_tokens": max(1, len(system + user) // 4),
            "completion_tokens": max(1, len(self.last_raw_content) // 4),
            "total_tokens": 0,
        }
        self.last_usage["total_tokens"] = (
            self.last_usage["prompt_tokens"] + self.last_usage["completion_tokens"]
        )
        return result


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_usage = None
        self.last_raw_content = ""

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

        self.last_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        text = (content or "").strip()
        self.last_raw_content = text
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
