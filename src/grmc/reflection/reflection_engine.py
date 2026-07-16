"""
Reflection Engine v0.1 (Conservative, report-only)

Aligned with GRMC principles:
- Prefer missing a signal over injecting a wrong high-confidence belief.
- Human oversight for any knowledge-graph mutation (this engine never mutates).
- Start with simple heuristics; LLM-assisted reflection comes later.

The only side effect optional here is writing a JSON report to disk for audit.
Memory stores and graphs are never modified by ``reflect``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from ..models.reflection_report import (
    ConceptCandidate,
    ContradictionFlag,
    ReflectionReport,
)


# Shared content tokens shorter than this are ignored when pairing episodes.
_MIN_TOKEN_LEN = 3
_MAX_PAIRWISE_EPISODES = 12  # O(n^2) bound for contradiction scan

# Conservative negation / polarity cues (EN + JP). Weak signals only.
_NEGATION_CUES = (
    "not",
    "never",
    "no",
    "doesn't",
    "don't",
    "isn't",
    "aren't",
    "cannot",
    "can't",
    "without",
    "ない",
    "ません",
    "ではない",
    "じゃない",
    "不要",
    "禁止",
    "避ける",
    "やめる",
)

# Opposing concept pairs — only used as soft flags when both poles appear.
_OPPOSING_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("essential", "unnecessary"),
    ("required", "optional"),
    ("always", "never"),
    ("true", "false"),
    ("important", "unimportant"),
    ("必要", "不要"),
    ("必須", "任意"),
    ("正しい", "誤り"),
    ("賛成", "反対"),
    ("永続", "一時"),
    ("自動", "手動"),
)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for EN + JP mixed text.

    - Latin words: split on non-alphanumeric
    - CJK runs: keep contiguous runs of length >= 2 as tokens
    This is intentionally simple (no MeCab / spaCy dependency in Phase 0).
    """
    if not text:
        return []
    tokens: List[str] = []
    # Latin / digit tokens
    for m in re.finditer(r"[A-Za-z0-9_]{%d,}" % _MIN_TOKEN_LEN, text):
        tokens.append(m.group(0).lower())
    # CJK sequences — keep whole runs only (2-grams are too noisy without a
    # proper morphological analyzer; better false-negative than junk concepts).
    for m in re.finditer(r"[\u3040-\u30ff\u3400-\u9fff]{2,}", text):
        tokens.append(m.group(0))
    return tokens


def _has_negation(text: str) -> bool:
    lowered = _normalize(text)
    return any(cue in lowered for cue in _NEGATION_CUES)


def _content_overlap(a: str, b: str) -> List[str]:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    # Drop pure negation-like tokens from overlap signal
    noise = {c for c in _NEGATION_CUES if len(c) >= _MIN_TOKEN_LEN}
    return sorted((ta & tb) - noise)


class ReflectionEngine:
    """Produce conservative reflection reports without mutating memory."""

    def __init__(self, memory_manager: Any, report_dir: Optional[str] = None):
        self.memory_manager = memory_manager
        self.store = memory_manager.store
        self.last_reflection_time: Optional[datetime] = None
        self.last_report: Optional[ReflectionReport] = None

        if report_dir is None:
            base = getattr(self.store, "persist_dir", Path("./grmc_data"))
            report_dir = str(Path(base) / "reflections")
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        recent_limit: int = 30,
        topic: Optional[str] = None,
        top_k_for_topic: int = 15,
        persist: bool = True,
    ) -> ReflectionReport:
        """Run a reflection pass and return a structured report.

        Modes:
        - topic is None  → analyze recent episodes (chronological approximation)
        - topic set      → retrieve semantically related episodes for that topic
        """
        if topic:
            episodes = self._episodes_for_topic(topic, top_k=top_k_for_topic)
            mode: str = "topic"
        else:
            episodes = self._recent_episodes(limit=recent_limit)
            mode = "recent"

        report = ReflectionReport(
            report_id=f"refl_{uuid4().hex[:12]}",
            timestamp=datetime.utcnow(),
            mode=mode,  # type: ignore[arg-type]
            topic=topic,
            episodes_analyzed=len(episodes),
            episode_ids=[e["episode_id"] for e in episodes],
            confidence_level="conservative",
            mutates_memory=False,
            engine_version="0.1.0",
        )

        self._attach_limitations(report, mode=mode, requested_limit=recent_limit)

        if not episodes:
            report.potential_issues.append(
                "No episodes available to reflect on. Ingest memory first "
                "(`grmc ingest` or scripts/example_ingest.py)."
            )
            report.suggested_actions.append(
                "Ingest a small set of conversation notes, then re-run `grmc reflect`."
            )
            report.notes.append(
                "Reflection completed with empty input. No graph mutations attempted "
                "(engine never mutates memory)."
            )
            return self._finalize(report, persist=persist)

        # --- Concept candidates (observational) ---
        frequencies, candidates = self._extract_concept_candidates(episodes)
        report.concept_frequencies = dict(frequencies.most_common(40))
        report.concept_candidates = candidates

        # --- Contradiction / tension heuristics (low confidence) ---
        report.potential_contradictions = self._scan_contradictions(episodes)

        # --- Suggested actions (never auto-applied) ---
        report.suggested_actions.extend(self._suggest_actions(report))

        # --- Issues / notes ---
        if not report.concept_candidates:
            report.potential_issues.append(
                "No concept candidates extracted. Heuristic tokenizer may be weak "
                "for this content; consider adding extracted_concepts on ingest."
            )
        if report.potential_contradictions:
            report.potential_issues.append(
                f"{len(report.potential_contradictions)} potential tension(s) flagged "
                "at low confidence. Human review required before any belief update."
            )
        else:
            report.notes.append(
                "No contradiction heuristic fired. This is NOT proof of consistency — "
                "only that weak surface checks found nothing."
            )

        report.notes.append(
            "Report-only mode: knowledge graph was not modified. "
            "Approve any graph write through a future human-in-the-loop gate."
        )
        report.notes.append(
            f"Analyzed {len(episodes)} episode(s) in mode={mode!r}"
            + (f" topic={topic!r}" if topic else "")
            + "."
        )

        return self._finalize(report, persist=persist)

    def simple_contradiction_check(self, text_a: str, text_b: str) -> bool:
        """Public, intentionally weak pairwise check (for tests & CLI demos)."""
        flag = self._pair_contradiction(
            episode_id_a="a",
            episode_id_b="b",
            summary_a=text_a,
            summary_b=text_b,
        )
        return flag is not None

    # ------------------------------------------------------------------
    # Episode selection
    # ------------------------------------------------------------------

    def _recent_episodes(self, limit: int) -> List[Dict[str, Any]]:
        if hasattr(self.store, "list_recent"):
            return self.store.list_recent(limit=limit)
        return []

    def _episodes_for_topic(self, topic: str, top_k: int) -> List[Dict[str, Any]]:
        """Use semantic retrieve when available; fall back to recent scan."""
        if hasattr(self.memory_manager, "retrieve"):
            results = self.memory_manager.retrieve(topic, top_k=top_k)
            # retrieve() returns distance; keep compatible shape
            return results
        return self._recent_episodes(limit=top_k)

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def _extract_concept_candidates(
        self, episodes: Sequence[Dict[str, Any]]
    ) -> Tuple[Counter, List[ConceptCandidate]]:
        token_counter: Counter = Counter()
        token_sources: Dict[str, List[str]] = {}
        field_concepts: Dict[str, List[str]] = {}

        for ep in episodes:
            eid = ep.get("episode_id") or "unknown"
            # Prefer explicit concepts if present
            explicit = ep.get("extracted_concepts") or []
            if not explicit:
                meta = ep.get("metadata") or {}
                raw = meta.get("extracted_concepts")
                if raw:
                    if isinstance(raw, list):
                        explicit = raw
                    else:
                        explicit = [p for p in str(raw).split("|") if p]

            for concept in explicit:
                label = concept.strip()
                if not label:
                    continue
                field_concepts.setdefault(label, []).append(eid)

            summary = ep.get("summary") or ep.get("content_summary") or ""
            for tok in _tokenize(summary):
                if len(tok) < _MIN_TOKEN_LEN and not re.search(
                    r"[\u3040-\u30ff\u3400-\u9fff]", tok
                ):
                    continue
                token_counter[tok] += 1
                token_sources.setdefault(tok, []).append(eid)

        candidates: List[ConceptCandidate] = []

        # Explicit fields: slightly higher (still conservative) confidence
        for label, eids in sorted(field_concepts.items(), key=lambda x: -len(x[1])):
            candidates.append(
                ConceptCandidate(
                    label=label,
                    frequency=len(eids),
                    supporting_episode_ids=sorted(set(eids)),
                    confidence=min(0.55, 0.3 + 0.05 * len(set(eids))),
                    source="episode_field",
                )
            )

        # Heuristic tokens that appear more than once, or strongly once with length
        for tok, freq in token_counter.most_common(50):
            if freq < 2 and len(tok) < 6:
                continue
            # Skip if already covered by explicit field with same label
            if any(c.label == tok for c in candidates):
                continue
            eids = token_sources.get(tok, [])
            candidates.append(
                ConceptCandidate(
                    label=tok,
                    frequency=freq,
                    supporting_episode_ids=sorted(set(eids))[:20],
                    confidence=min(0.45, 0.2 + 0.04 * freq),
                    source="heuristic",
                )
            )

        # Cap list to keep reports readable
        candidates = sorted(
            candidates, key=lambda c: (c.frequency, c.confidence), reverse=True
        )[:25]
        return token_counter, candidates

    # ------------------------------------------------------------------
    # Contradiction heuristics
    # ------------------------------------------------------------------

    def _scan_contradictions(
        self, episodes: Sequence[Dict[str, Any]]
    ) -> List[ContradictionFlag]:
        flags: List[ContradictionFlag] = []
        subset = list(episodes)[:_MAX_PAIRWISE_EPISODES]

        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                flag = self._pair_contradiction(
                    episode_id_a=a.get("episode_id", f"i{i}"),
                    episode_id_b=b.get("episode_id", f"j{j}"),
                    summary_a=a.get("summary") or "",
                    summary_b=b.get("summary") or "",
                )
                if flag:
                    flags.append(flag)

        # Soft global polarity: opposing terms appear across the batch
        corpus = " ".join((e.get("summary") or "") for e in episodes)
        lowered = _normalize(corpus)
        for left, right in _OPPOSING_PAIRS:
            if left in lowered and right in lowered:
                flags.append(
                    ContradictionFlag(
                        episode_id_a="(corpus)",
                        episode_id_b="(corpus)",
                        summary_a=f"pole:{left}",
                        summary_b=f"pole:{right}",
                        reason=(
                            f"Both opposing terms {left!r} and {right!r} appear in the "
                            "analyzed set. May indicate unresolved tension — not proof."
                        ),
                        confidence=0.2,
                        requires_human_review=True,
                    )
                )

        return flags

    def _pair_contradiction(
        self,
        episode_id_a: str,
        episode_id_b: str,
        summary_a: str,
        summary_b: str,
    ) -> Optional[ContradictionFlag]:
        if not summary_a or not summary_b:
            return None

        overlap = _content_overlap(summary_a, summary_b)
        if len(overlap) < 1:
            return None

        neg_a = _has_negation(summary_a)
        neg_b = _has_negation(summary_b)
        if neg_a == neg_b:
            return None

        # Shared substance + differing polarity → soft flag
        return ContradictionFlag(
            episode_id_a=episode_id_a,
            episode_id_b=episode_id_b,
            summary_a=summary_a[:200],
            summary_b=summary_b[:200],
            reason=(
                "Shared content tokens "
                f"{overlap[:8]} with differing negation polarity. "
                "Heuristic only — verify before updating any belief."
            ),
            confidence=0.25,
            requires_human_review=True,
        )

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def _attach_limitations(
        self, report: ReflectionReport, mode: str, requested_limit: int
    ) -> None:
        report.limitations.extend(
            [
                "ChromaDB has no native chronological index; 'recent' is "
                "client-side sort on metadata.timestamp written at ingest.",
                "Concept extraction is heuristic (regex tokenizer), not LLM-based.",
                "Contradiction detection is weak surface heuristics at low confidence.",
                "No knowledge graph is written by this engine (mutates_memory=False).",
                "Embeddings are used for topic mode only when MemoryManager.retrieve "
                "is available; pairwise contradiction does not yet use embeddings.",
            ]
        )
        if mode == "recent":
            report.limitations.append(
                f"Requested recent_limit={requested_limit}; actual analyzed count "
                "depends on how many episodes exist and have timestamps."
            )

    def _suggest_actions(self, report: ReflectionReport) -> List[str]:
        actions = [
            "Review concept_candidates manually before promoting any to graph nodes.",
            "Do not raise confidence above 0.6 without additional evidence or human sign-off.",
        ]
        if report.potential_contradictions:
            actions.append(
                "Inspect each potential_contradiction; either resolve in a new episode "
                "note, or mark as open question (future: open_questions store)."
            )
        if report.concept_candidates:
            top = ", ".join(c.label for c in report.concept_candidates[:5])
            actions.append(
                f"Consider explicit extracted_concepts on ingest for: {top}"
            )
        actions.append(
            "Next iteration: add embedding-similarity contradiction check + optional "
            "LLM verification behind a feature flag (still report-only by default)."
        )
        actions.append(
            "Next iteration: SQLite (or append-only log) for true recent-episode index "
            "and reflection history queries."
        )
        return actions

    def _finalize(
        self, report: ReflectionReport, persist: bool
    ) -> ReflectionReport:
        self.last_reflection_time = report.timestamp
        self.last_report = report
        if persist:
            path = self._persist_report(report)
            report.metadata["report_path"] = str(path)
        return report

    def _persist_report(self, report: ReflectionReport) -> Path:
        """Write report JSON for audit trail (does not touch episode memory)."""
        ts = report.timestamp.strftime("%Y%m%dT%H%M%SZ")
        path = self.report_dir / f"{ts}_{report.report_id}.json"
        path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        # Also keep a pointer to the latest report for CLI status
        latest = self.report_dir / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "report_id": report.report_id,
                    "path": str(path),
                    "timestamp": report.timestamp.isoformat(),
                    "episodes_analyzed": report.episodes_analyzed,
                    "mutates_memory": report.mutates_memory,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
