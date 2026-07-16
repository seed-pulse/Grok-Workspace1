"""Report-only LLM enrichment for reflection.

Never writes graph nodes/edges. Always applies confidence caps.
Failures fall back to heuristic results unchanged (plus a note).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models.reflection_report import (
    ConceptCandidate,
    ContradictionFlag,
    ReflectionReport,
)
from .audit import LLMAuditLog, TimedLLMCall
from .client import LLMClient, LLMError, MockLLMClient, build_client
from .config import LLMConfig


class LLMVerifier:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[LLMClient] = None,
        audit: Optional[LLMAuditLog] = None,
        data_dir: Optional[str | Path] = None,
    ):
        self.config = config or LLMConfig.from_env()
        self.client = client or (
            build_client(self.config) if self.config.enabled else MockLLMClient()
        )
        if audit is not None:
            self.audit = audit
        elif data_dir is not None:
            self.audit = LLMAuditLog(data_dir)
        else:
            self.audit = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def enrich_report(
        self,
        report: ReflectionReport,
        episodes: Sequence[Dict[str, Any]],
    ) -> ReflectionReport:
        """Mutate-in-place report fields only (still mutates_memory=False)."""
        if not self.config.enabled:
            report.metadata["llm"] = {"enabled": False}
            return report

        report.metadata["llm"] = {
            "enabled": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "concept_conf_cap": self.config.concept_conf_cap,
            "contradiction_conf_cap": self.config.contradiction_conf_cap,
            "audit": str(self.audit.path) if self.audit else None,
        }
        call_ids: List[str] = []

        # 1) Concept enrichment
        try:
            concepts, call_id = self._extract_concepts(
                episodes, report_id=report.report_id
            )
            if call_id:
                call_ids.append(call_id)
            if concepts:
                report.concept_candidates = self._merge_concepts(
                    report.concept_candidates, concepts
                )
                report.notes.append(
                    f"LLM concept enrichment applied ({len(concepts)} labels, "
                    f"cap={self.config.concept_conf_cap})."
                )
                report.metadata["llm"]["concepts_from_llm"] = len(concepts)
        except LLMError as exc:
            report.notes.append(f"LLM concept extraction failed; heuristic kept ({exc}).")
            report.metadata.setdefault("llm", {})["concept_error"] = str(exc)

        # 2) Contradiction verification / scoring
        try:
            if report.potential_contradictions:
                verified, call_id = self._verify_contradictions(
                    report.potential_contradictions,
                    episodes,
                    report_id=report.report_id,
                )
                if call_id:
                    call_ids.append(call_id)
                report.potential_contradictions = verified
                report.notes.append(
                    "LLM contradiction review applied (still low-confidence; "
                    "human review required)."
                )
                report.metadata["llm"]["contradictions_reviewed"] = len(verified)
        except LLMError as exc:
            report.notes.append(
                f"LLM contradiction review failed; heuristic flags kept ({exc})."
            )
            report.metadata.setdefault("llm", {})["contradiction_error"] = str(exc)

        report.metadata["llm"]["call_ids"] = call_ids
        report.mutates_memory = False  # hard invariant
        report.metadata["llm"]["mutates_memory"] = False
        return report

    def _episode_blob(self, episodes: Sequence[Dict[str, Any]], limit: int = 12) -> str:
        lines = []
        for ep in list(episodes)[:limit]:
            eid = ep.get("episode_id", "?")
            summary = (ep.get("summary") or ep.get("content_summary") or "")[:400]
            lines.append(f"- {eid}: {summary}")
        return "\n".join(lines)

    def _call_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        temperature: float,
        report_id: Optional[str],
    ) -> tuple[Dict[str, Any], str]:
        with TimedLLMCall(
            self.audit,
            purpose=purpose,
            model=self.config.model,
            provider=self.config.provider,
            report_id=report_id,
        ) as timed:
            try:
                data = self.client.complete_json(
                    system, user, temperature=temperature
                )
                usage = getattr(self.client, "last_usage", None) or {}
                raw = getattr(self.client, "last_raw_content", "") or json.dumps(data)
                timed.mark_success(
                    prompt=system + "\n" + user,
                    completion=raw if isinstance(raw, str) else json.dumps(raw),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
                timed.record.metadata["purpose_detail"] = purpose
                return data, timed.record.call_id
            except Exception:
                # TimedLLMCall.__exit__ records failure
                raise

    def _extract_concepts(
        self,
        episodes: Sequence[Dict[str, Any]],
        *,
        report_id: Optional[str] = None,
    ) -> tuple[List[ConceptCandidate], Optional[str]]:
        system = (
            "You extract conservative concept labels for a long-term AI memory system. "
            "Return JSON only: {\"concepts\": [{\"label\": str, \"confidence\": float, "
            "\"episode_ids\": [str], \"rationale\": str}]}. "
            "Rules: prefer precise multi-word labels; avoid stopwords; confidence "
            f"must be <= {self.config.concept_conf_cap}; do not invent facts; "
            "empty list is OK."
        )
        user = (
            "Episodes:\n"
            + self._episode_blob(episodes)
            + "\n\nExtract up to 12 high-signal concepts."
        )
        data, call_id = self._call_json(
            purpose="concept_extract",
            system=system,
            user=user,
            temperature=0.1,
            report_id=report_id,
        )
        raw = data.get("concepts") or []
        out: List[ConceptCandidate] = []
        if not isinstance(raw, list):
            return out, call_id
        for item in raw[:12]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label or len(label) < 2:
                continue
            try:
                conf = float(item.get("confidence", 0.35))
            except (TypeError, ValueError):
                conf = 0.35
            conf = max(0.0, min(conf, self.config.concept_conf_cap))
            eids = item.get("episode_ids") or []
            if not isinstance(eids, list):
                eids = []
            out.append(
                ConceptCandidate(
                    label=label,
                    frequency=max(1, len(eids)),
                    supporting_episode_ids=[str(x) for x in eids][:20],
                    confidence=conf,
                    source="llm",
                )
            )
        return out, call_id

    def _merge_concepts(
        self,
        heuristic: List[ConceptCandidate],
        llm_concepts: List[ConceptCandidate],
    ) -> List[ConceptCandidate]:
        by_label: Dict[str, ConceptCandidate] = {}
        for c in heuristic:
            by_label[c.label.lower()] = c
        for c in llm_concepts:
            key = c.label.lower()
            if key in by_label:
                old = by_label[key]
                eps = list(
                    dict.fromkeys(old.supporting_episode_ids + c.supporting_episode_ids)
                )
                by_label[key] = ConceptCandidate(
                    label=old.label if len(old.label) >= len(c.label) else c.label,
                    frequency=max(old.frequency, c.frequency, len(eps)),
                    supporting_episode_ids=eps[:20],
                    confidence=min(
                        self.config.concept_conf_cap,
                        max(old.confidence, c.confidence),
                    ),
                    source="llm+heuristic" if old.source != "llm" else "llm",
                )
            else:
                by_label[key] = c
        merged = sorted(
            by_label.values(),
            key=lambda x: (x.frequency, x.confidence),
            reverse=True,
        )
        return merged[:25]

    def _verify_contradictions(
        self,
        flags: List[ContradictionFlag],
        episodes: Sequence[Dict[str, Any]],
        *,
        report_id: Optional[str] = None,
    ) -> tuple[List[ContradictionFlag], Optional[str]]:
        system = (
            "You review soft contradiction flags for an AI memory system. "
            "Return JSON only: {\"items\": [{\"index\": int, \"keep\": bool, "
            "\"confidence\": float, \"reason\": str}]}. "
            "Be conservative: keep=false unless there is a clear tension. "
            f"confidence must be <= {self.config.contradiction_conf_cap}."
        )
        payload = []
        for i, f in enumerate(flags[:15]):
            payload.append(
                {
                    "index": i,
                    "episode_id_a": f.episode_id_a,
                    "episode_id_b": f.episode_id_b,
                    "summary_a": f.summary_a,
                    "summary_b": f.summary_b,
                    "reason": f.reason,
                    "method": f.method,
                    "confidence": f.confidence,
                }
            )
        user = json.dumps(
            {"flags": payload, "context_episodes": self._episode_blob(episodes, 8)},
            ensure_ascii=False,
        )
        data, call_id = self._call_json(
            purpose="contradiction_review",
            system=system,
            user=user,
            temperature=0.0,
            report_id=report_id,
        )
        items = data.get("items") or []
        if not isinstance(items, list) or not items:
            return (
                [
                    self._cap_flag(f, f.confidence, f.reason + " [llm:no-op]")
                    for f in flags
                ],
                call_id,
            )

        by_idx: Dict[int, Dict[str, Any]] = {}
        for it in items:
            if isinstance(it, dict) and "index" in it:
                try:
                    by_idx[int(it["index"])] = it
                except (TypeError, ValueError):
                    continue

        out: List[ContradictionFlag] = []
        for i, f in enumerate(flags):
            review = by_idx.get(i)
            if not review:
                out.append(self._cap_flag(f, f.confidence, f.reason))
                continue
            keep = bool(review.get("keep", True))
            if not keep:
                continue
            try:
                conf = float(review.get("confidence", f.confidence))
            except (TypeError, ValueError):
                conf = f.confidence
            reason = str(review.get("reason") or f.reason)
            out.append(
                self._cap_flag(
                    f,
                    conf,
                    reason + " [llm-reviewed]",
                    method=f"{f.method}+llm",
                )
            )
        return out, call_id

    def _cap_flag(
        self,
        flag: ContradictionFlag,
        conf: float,
        reason: str,
        method: Optional[str] = None,
    ) -> ContradictionFlag:
        conf = max(0.0, min(float(conf), self.config.contradiction_conf_cap))
        return ContradictionFlag(
            episode_id_a=flag.episode_id_a,
            episode_id_b=flag.episode_id_b,
            summary_a=flag.summary_a,
            summary_b=flag.summary_b,
            reason=reason,
            confidence=conf,
            requires_human_review=True,
            method=method or flag.method,
            similarity=flag.similarity,
        )
