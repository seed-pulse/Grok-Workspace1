"""Append-only LLM call audit log (JSONL + optional SQLite mirror).

Never enables LLM by itself — only records when verification actually runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class LLMCallRecord:
    call_id: str = field(default_factory=lambda: f"llm_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    purpose: str = "unknown"  # concept_extract | contradiction_review | other
    model: str = ""
    provider: str = ""
    success: bool = False
    error: str = ""
    latency_ms: int = 0
    # Token accounting: prefer provider usage; else estimate
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    total_tokens_est: int = 0
    usage_source: str = "estimate"  # estimate | provider
    report_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Not billing-grade."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class LLMAuditLog:
    """JSONL audit under ``{data_dir}/llm_audit/calls.jsonl``."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.log_dir = self.data_dir / "llm_audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "calls.jsonl"

    def append(self, record: LLMCallRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 50) -> List[LLMCallRecord]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records: List[LLMCallRecord] = []
        for line in lines[-max(0, limit) :]:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(LLMCallRecord(**{
                    k: data.get(k)
                    for k in LLMCallRecord.__dataclass_fields__
                    if k in data
                }))
            except Exception:
                continue
        return list(reversed(records))

    def summary(self) -> Dict[str, Any]:
        records = self.list_recent(limit=10000)
        total = len(records)
        ok = sum(1 for r in records if r.success)
        fail = total - ok
        tokens = sum(r.total_tokens_est for r in records)
        by_purpose: Dict[str, int] = {}
        for r in records:
            by_purpose[r.purpose] = by_purpose.get(r.purpose, 0) + 1
        return {
            "path": str(self.path.resolve()),
            "calls": total,
            "success": ok,
            "failed": fail,
            "total_tokens_est": tokens,
            "by_purpose": by_purpose,
        }


class TimedLLMCall:
    """Context helper to time and log a call."""

    def __init__(
        self,
        audit: Optional[LLMAuditLog],
        *,
        purpose: str,
        model: str,
        provider: str,
        report_id: Optional[str] = None,
    ):
        self.audit = audit
        self.purpose = purpose
        self.model = model
        self.provider = provider
        self.report_id = report_id
        self._t0 = 0.0
        self.record = LLMCallRecord(
            purpose=purpose,
            model=model,
            provider=provider,
            report_id=report_id,
        )

    def __enter__(self) -> "TimedLLMCall":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.latency_ms = int((time.perf_counter() - self._t0) * 1000)
        if exc is not None:
            self.record.success = False
            self.record.error = str(exc)
        if self.audit is not None:
            self.audit.append(self.record)
        return False  # never swallow

    def mark_success(
        self,
        *,
        prompt: str = "",
        completion: str = "",
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        self.record.success = True
        if prompt_tokens is not None or completion_tokens is not None:
            self.record.prompt_tokens_est = int(prompt_tokens or 0)
            self.record.completion_tokens_est = int(completion_tokens or 0)
            self.record.usage_source = "provider"
        else:
            self.record.prompt_tokens_est = estimate_tokens(prompt)
            self.record.completion_tokens_est = estimate_tokens(completion)
            self.record.usage_source = "estimate"
        self.record.total_tokens_est = (
            self.record.prompt_tokens_est + self.record.completion_tokens_est
        )
