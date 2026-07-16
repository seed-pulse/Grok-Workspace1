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
    EdgeSuggestion,
    ReflectionReport,
)
from ..core.embedder import cosine_similarity


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

    def __init__(
        self,
        memory_manager: Any,
        report_dir: Optional[str] = None,
        llm_enabled: Optional[bool] = None,
        llm_verifier: Any = None,
    ):
        self.memory_manager = memory_manager
        self.store = memory_manager.store
        self.last_reflection_time: Optional[datetime] = None
        self.last_report: Optional[ReflectionReport] = None
        self._llm_enabled_override = llm_enabled
        self._llm_verifier = llm_verifier

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
        llm: Optional[bool] = None,
    ) -> ReflectionReport:
        """Run a reflection pass and return a structured report.

        Modes:
        - topic is None  → analyze recent episodes (chronological approximation)
        - topic set      → retrieve semantically related episodes for that topic

        ``llm``: None → env default (off unless GRMC_LLM=1); True/False force.
        LLM path is report-only and never writes the graph.
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
            engine_version="0.6.0",
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
        # Embedding pairwise tensions (report-only; still low confidence)
        emb_flags = self._scan_embedding_tensions(episodes)
        report.potential_contradictions.extend(emb_flags)

        # Optional LLM enrichment (feature flag; default off)
        use_llm = self._resolve_llm_flag(llm)
        if use_llm:
            report = self._apply_llm(report, episodes)
        else:
            report.metadata["llm"] = {"enabled": False}
            report.notes.append("LLM verification off (default). Heuristics only.")

        # Soft edge ideas when graph nodes already exist (still not written)
        report.edge_suggestions = self._suggest_edges(report, episodes)

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
            "Concept/edge candidates may be enqueued as pending proposals; "
            "only `grmc approve <id>` writes nodes or edges."
        )
        if report.edge_suggestions:
            report.notes.append(
                f"{len(report.edge_suggestions)} soft edge suggestion(s) — "
                "review via `grmc propose` if enqueued; never auto-applied."
            )
        report.notes.append(
            f"Analyzed {len(episodes)} episode(s) in mode={mode!r}"
            + (f" topic={topic!r}" if topic else "")
            + "."
        )

        report.mutates_memory = False
        return self._finalize(report, persist=persist)

    def _resolve_llm_flag(self, llm: Optional[bool]) -> bool:
        if llm is not None:
            return bool(llm)
        if self._llm_enabled_override is not None:
            return bool(self._llm_enabled_override)
        try:
            from ..llm.config import llm_enabled_from_env

            return llm_enabled_from_env()
        except Exception:
            return False

    def _apply_llm(
        self, report: ReflectionReport, episodes: Sequence[Dict[str, Any]]
    ) -> ReflectionReport:
        try:
            if self._llm_verifier is not None:
                verifier = self._llm_verifier
            else:
                from ..llm.config import LLMConfig
                from ..llm.verification import LLMVerifier

                cfg = LLMConfig.from_env(force_enabled=True)
                verifier = LLMVerifier(config=cfg)
            return verifier.enrich_report(report, episodes)
        except Exception as exc:
            report.notes.append(
                f"LLM path failed open; using heuristics only ({exc})."
            )
            report.metadata["llm"] = {
                "enabled": True,
                "error": str(exc),
                "fallback": "heuristic",
            }
            report.mutates_memory = False
            return report

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
        # Prefer MemoryManager / SQLite (indexed timestamp). Vector store is not SoR.
        if hasattr(self.memory_manager, "list_recent"):
            return self.memory_manager.list_recent(limit=limit)
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

    def _scan_embedding_tensions(
        self, episodes: Sequence[Dict[str, Any]]
    ) -> List[ContradictionFlag]:
        """High embedding similarity + differing polarity → soft tension flag.

        Report-only. Confidence stays low. Requires MemoryManager.embedder.
        """
        embedder = getattr(self.memory_manager, "embedder", None)
        if embedder is None:
            return []

        subset = list(episodes)[:_MAX_PAIRWISE_EPISODES]
        texts = [(e.get("summary") or e.get("content_summary") or "") for e in subset]
        if sum(1 for t in texts if t.strip()) < 2:
            return []

        try:
            vectors = [embedder.encode(t) for t in texts]
        except Exception:
            return []

        flags: List[ContradictionFlag] = []
        # Threshold: high similarity. Hashing embedder is noisier → slightly lower.
        name = getattr(embedder, "name", "")
        thr = 0.72 if "hashing" in str(name) else 0.82

        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not texts[i] or not texts[j]:
                    continue
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim < thr:
                    continue
                neg_a = _has_negation(texts[i])
                neg_b = _has_negation(texts[j])
                if neg_a == neg_b:
                    continue
                flags.append(
                    ContradictionFlag(
                        episode_id_a=subset[i].get("episode_id", f"i{i}"),
                        episode_id_b=subset[j].get("episode_id", f"j{j}"),
                        summary_a=texts[i][:200],
                        summary_b=texts[j][:200],
                        reason=(
                            f"High embedding similarity ({sim:.3f}) with differing "
                            "negation polarity. Soft tension only — not proof of conflict."
                        ),
                        confidence=min(0.3, 0.15 + 0.1 * sim),
                        requires_human_review=True,
                        method="embedding_polarity",
                        similarity=sim,
                    )
                )
        return flags

    def _suggest_edges(
        self,
        report: ReflectionReport,
        episodes: Sequence[Dict[str, Any]],
    ) -> List[EdgeSuggestion]:
        """Suggest low-confidence edges only when both endpoints already exist.

        Never creates nodes. Basic types: contradicts | related_to | supports.
        """
        sqlite = getattr(self.memory_manager, "sqlite", None)
        if sqlite is None:
            return []

        suggestions: List[EdgeSuggestion] = []
        seen: set[tuple[str, str, str]] = set()

        def add_sugg(
            src_node,
            tgt_node,
            edge_type: str,
            conf: float,
            reason: str,
            ep_ids: List[str],
        ) -> None:
            key = (src_node.node_id, tgt_node.node_id, edge_type)
            if key in seen or src_node.node_id == tgt_node.node_id:
                return
            # Skip if edge already exists or pending
            if sqlite.find_edge(src_node.node_id, tgt_node.node_id, edge_type):
                return
            if sqlite.find_pending_edge_proposal(
                src_node.node_id, tgt_node.node_id, edge_type
            ):
                return
            seen.add(key)
            suggestions.append(
                EdgeSuggestion(
                    source_label=src_node.label,
                    target_label=tgt_node.label,
                    source_node_id=src_node.node_id,
                    target_node_id=tgt_node.node_id,
                    edge_type=edge_type,
                    confidence=min(0.3, conf),
                    reason=reason,
                    supporting_episode_ids=ep_ids[:8],
                    requires_human_review=True,
                )
            )

        # Map episode_id → nodes grounded by that episode
        nodes = sqlite.list_graph_nodes(limit=200)
        ep_to_nodes: Dict[str, List[Any]] = {}
        for node in nodes:
            for eid in node.supporting_episodes or []:
                ep_to_nodes.setdefault(eid, []).append(node)
            for link in sqlite.list_links_for_node(node.node_id):
                ep_to_nodes.setdefault(link.episode_id, []).append(node)

        for flag in report.potential_contradictions:
            ea, eb = flag.episode_id_a, flag.episode_id_b
            if ea.startswith("(") or eb.startswith("("):
                continue
            for na in ep_to_nodes.get(ea, []):
                for nb in ep_to_nodes.get(eb, []):
                    if na.node_id == nb.node_id:
                        continue
                    add_sugg(
                        na,
                        nb,
                        "contradicts",
                        conf=min(0.28, flag.confidence + 0.05),
                        reason=(
                            f"Episodes {ea}↔{eb} flagged ({flag.method}); "
                            f"nodes '{na.label}' and '{nb.label}' may be in tension."
                        ),
                        ep_ids=[ea, eb],
                    )

        # Co-mentioned concepts that both exist as nodes → soft related_to
        # Stricter v0.6: require >=2 shared episodes and non-trivial labels.
        labels_present = {
            c.label.lower(): c
            for c in report.concept_candidates
            if c.supporting_episode_ids and len(c.label) >= 4
        }
        node_by_label = {n.label.lower(): n for n in nodes}
        shared_labels = [lab for lab in labels_present if lab in node_by_label]
        for i in range(len(shared_labels)):
            for j in range(i + 1, len(shared_labels)):
                la, lb = shared_labels[i], shared_labels[j]
                ca, cb = labels_present[la], labels_present[lb]
                common = set(ca.supporting_episode_ids) & set(cb.supporting_episode_ids)
                if len(common) < 2:
                    continue
                add_sugg(
                    node_by_label[la],
                    node_by_label[lb],
                    "related_to",
                    conf=0.18,
                    reason=(
                        f"Concepts co-occur in {len(common)} episode(s) "
                        f"{sorted(common)[:3]}; soft related_to only."
                    ),
                    ep_ids=sorted(common)[:6],
                )

        # Prefer contradicts over related_to for same pair when both present
        by_pair: Dict[tuple[str, str], List[EdgeSuggestion]] = {}
        for s in suggestions:
            key = tuple(sorted([s.source_node_id or "", s.target_node_id or ""]))
            by_pair.setdefault(key, []).append(s)
        refined: List[EdgeSuggestion] = []
        for pair_suggs in by_pair.values():
            types = {s.edge_type for s in pair_suggs}
            if "contradicts" in types and "related_to" in types:
                refined.extend(s for s in pair_suggs if s.edge_type != "related_to")
            else:
                refined.extend(pair_suggs)
        return refined[:15]

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
            method="negation_overlap",
        )

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def _attach_limitations(
        self, report: ReflectionReport, mode: str, requested_limit: int
    ) -> None:
        report.limitations.extend(
            [
                "Recent episodes come from SQLite (timestamp index). "
                "ChromaDB is used only for semantic/topic retrieval.",
                "Concept extraction is heuristic (regex tokenizer), not LLM-based.",
                "Contradiction detection mixes surface heuristics + optional "
                "embedding polarity checks; both stay low confidence.",
                "No knowledge graph is written by this engine (mutates_memory=False).",
                "Graph writes require human `grmc approve` on pending proposals.",
                "Edge suggestions only appear when both endpoint nodes already exist.",
                "LLM verification is opt-in (GRMC_LLM=1 or --llm); failures fall back "
                "to heuristics; still mutates_memory=False.",
            ]
        )
        if mode == "recent":
            report.limitations.append(
                f"Requested recent_limit={requested_limit}; actual analyzed count "
                "depends on how many episodes exist in SQLite."
            )

    def _suggest_actions(self, report: ReflectionReport) -> List[str]:
        actions = [
            "Review concept_candidates manually before promoting any to graph nodes.",
            "Do not raise confidence above 0.6 without additional evidence or human sign-off.",
        ]
        if report.potential_contradictions:
            actions.append(
                "Inspect each potential_contradiction; either resolve in a new episode "
                "note, or leave as open tension (do not auto-raise node confidence)."
            )
        if report.edge_suggestions:
            actions.append(
                "Review edge_suggestions / pending edge proposals carefully — "
                "default types are contradicts/related_to at low confidence."
            )
        if report.concept_candidates:
            top = ", ".join(c.label for c in report.concept_candidates[:5])
            actions.append(
                f"Consider explicit extracted_concepts on ingest for: {top}"
            )
        actions.append("Run `grmc eval` periodically to watch over-confidence drift.")
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
