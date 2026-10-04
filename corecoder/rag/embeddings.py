"""Embedding models for RAG retrieval.

The retriever needs *some* way to turn text into vectors. We deliberately do
not hard-depend on openai/numpy here (CoreCoder's rule: no extra deps for the
core). So the default is a deterministic hashing embedder that needs nothing
but the stdlib:

  - tokenise on word/character boundaries (Chinese text included)
  - hash each token into a fixed-size bag-of-words vector
  - L2-normalise

It is crude (no semantics beyond exact n-gram overlap) but it is:

  * offline  - no API key, no network, works in CI and tests
  * cheap    - O(tokens) per document, trivial for a few MB of docs
  * stable   - same text -> same vector, everywhere

If numpy is importable at runtime we use it for the dot products (much faster
on real corpora). The cosine distance contract stays identical, so swapping
the backend never changes RAG behaviour, only its speed.

If you have an OpenAI-compatible embeddings endpoint you can set
CORECODER_RAG_EMBEDDING=openai and pass an LLM-shaped client (any object with
``.embeddings.create``) - the provider-neutral client in corecoder.llm is
already an OpenAI client, so ``LLM.client`` works as-is.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

DEFAULT_DIM = 256
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|.", re.UNICODE)
_HASHER = hashlib.blake2b


class Embedder(Protocol):
    """Anything that maps a list of texts to a list of vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedding:
    """Zero-dependency hashing embedder.

    Vector component ``j`` is the L2-normalised count of tokens whose hash
    falls in bucket ``j``. Collisions just add noise; the n-gram signal that
    survives is what makes retrieval work at all.
    """

    def __init__(self, dim: int = DEFAULT_DIM, with_ngrams: bool = True):
        self.dim = int(dim or DEFAULT_DIM)
        self.with_ngrams = with_ngrams

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        for i, tok in enumerate(tokens):
            self._bump(vec, tok)
            if self.with_ngrams and i + 1 < len(tokens):
                self._bump(vec, tok + " " + tokens[i + 1])
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def _bump(self, vec: list[float], token: str) -> None:
        h = int.from_bytes(_HASHER(token.encode("utf-8")).digest()[:8], "big")
        vec[h % self.dim] += 1.0


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Both vectors are expected pre-normalised, so this is
    a dot product; the fallback also handles unnormalised inputs."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# optional numpy fast path
# ---------------------------------------------------------------------------

def _use_numpy() -> bool:
    """Runtime opt-in for the vectorised backend. Off by default so tests and
    CI stay dependency-free; set CORECODER_RAG_NUMPY=1 to enable."""
    if os.getenv("CORECODER_RAG_NUMPY", "0") != "1":
        return False
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


class _NumpyCosine:
    """Vectorised cosine similarity (stacked rows)."""

    def __init__(self):
        import numpy as np  # noqa: F401

        self.np = np

    def matrix(self, rows: list[list[float]], query: list[float]) -> list[float]:
        np = self.np
        m = np.asarray(rows, dtype=np.float64)
        q = np.asarray(query, dtype=np.float64)
        return (m @ q).tolist()  # rows are already L2-normalised


def similarity_matrix(rows: list[list[float]], query: list[float]) -> list[float]:
    """Cosine scores of every row against the query vector.

    Uses the numpy backend when enabled, else a plain Python loop over
    :func:`cosine`.
    """
    if _use_numpy():
        return _NumpyCosine().matrix(rows, query)
    return [cosine(r, query) for r in rows]


# ---------------------------------------------------------------------------
# API-backed embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedding:
    """Embedder that delegates to ``LLM.embed`` - i.e. the OpenAI-compatible
    ``/embeddings`` endpoint the user already configured for chat.

    Nothing about credentials lives here: the LLM owns the key, the base URL,
    the retry policy and (for the litellm backend) the provider quirks. This
    class only adds batching, because a corpus of a few thousand chunks must
    not become a few thousand HTTP calls.

    ``dim`` is unknown until the first response, so it is filled in lazily.
    Vectors are L2-normalised on the way out to match HashEmbedding's contract,
    which lets :func:`similarity_matrix` treat both backends identically.
    """

    def __init__(self, llm, model: str | None = None, batch_size: int = 64):
        self.llm = llm
        self.model = model or getattr(llm, "embedding_model", None)
        self.batch_size = max(1, int(batch_size))
        self.dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # providers reject empty strings; a space keeps positions aligned
            vecs = self.llm.embed([t if t.strip() else " " for t in batch], model=self.model)
            for v in vecs:
                out.append(_l2(v))
        if out:
            self.dim = len(out[0])
        return out


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def make_embedder(kind: str = "hash", llm=None, *, dim: int = DEFAULT_DIM, model: str | None = None):
    """Pick an embedder by name: ``hash`` (offline default) or ``openai``.

    ``openai`` needs an object with ``.embed`` - pass the CoreCoder ``LLM``.
    An unknown name or a missing LLM degrades to the hashing embedder rather
    than raising: retrieval still works, just lexically, and a support bot that
    boots offline beats one that crashes on a typo'd env var.
    """
    kind = (kind or "hash").strip().lower()
    if kind in ("openai", "litellm", "api") and llm is not None and hasattr(llm, "embed"):
        return OpenAIEmbedding(llm, model=model)
    return HashEmbedding(dim=dim)


def signature(embedder) -> dict:
    """Identify an embedder well enough to detect an incompatible index.

    Vectors from two different embedders are not comparable, and mixing them
    silently degrades every score - so the bundle records this fingerprint and
    :meth:`KnowledgeBase.load` refuses to reuse vectors that don't match it.
    """
    if isinstance(embedder, HashEmbedding):
        return {"kind": "hash", "dim": embedder.dim, "ngrams": embedder.with_ngrams}
    if isinstance(embedder, OpenAIEmbedding):
        return {
            "kind": "openai",
            "model": embedder.model or getattr(embedder.llm, "embedding_model", None)
            or getattr(embedder.llm, "model", None),
            "dim": embedder.dim,
        }
    return {"kind": type(embedder).__name__, "dim": getattr(embedder, "dim", None)}


__all__ = [
    "DEFAULT_DIM",
    "Embedder",
    "HashEmbedding",
    "OpenAIEmbedding",
    "cosine",
    "make_embedder",
    "signature",
    "similarity_matrix",
]