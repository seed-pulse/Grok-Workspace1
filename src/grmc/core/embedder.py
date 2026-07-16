"""Pluggable embedding backends.

Phase 0 default is sentence-transformers when available.
A deterministic hashing fallback keeps local demos working when torch/ST
is broken or unavailable (lower quality retrieval — clearly labeled).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str

    def encode(self, text: str) -> List[float]:
        ...


class HashingEmbedder:
    """Lightweight bag-of-token hashing embedder (no ML deps).

    Not a substitute for real sentence embeddings — used only as a fallback
    so the rest of GRMC remains usable for development and reflection demos.
    """

    name = "hashing-fallback-v1"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[A-Za-z0-9_]{2,}|[\u3040-\u30ff\u3400-\u9fff]{1,}", text or "")
        if not tokens:
            tokens = ["__empty__"]
        for tok in tokens:
            digest = hashlib.sha256(tok.lower().encode("utf-8")).digest()
            # Two indices for mild multi-hash
            for offset in (0, 8):
                idx = int.from_bytes(digest[offset : offset + 4], "little") % self.dim
                sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
                vec[idx] += sign
        # L2 normalize for cosine space
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def encode(self, text: str) -> List[float]:
        return self._model.encode(text or "").tolist()


def create_embedder(prefer: str = "auto") -> Embedder:
    """Create an embedder.

    prefer:
      - "auto": try sentence-transformers, else hashing fallback
      - "st" / "sentence-transformers": require ST
      - "hash" / "hashing": always use hashing fallback
    """
    prefer = (prefer or "auto").lower()
    if prefer in ("hash", "hashing", "fallback"):
        return HashingEmbedder()

    if prefer in ("st", "sentence-transformers", "auto"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            if prefer != "auto":
                raise
            # Broken torch/numpy installs are common; stay usable via hashing.
            _ = exc
            return HashingEmbedder()

    raise ValueError(f"Unknown embedder preference: {prefer}")
