"""Retrieval layer: chunking, indexing and hybrid search.

The knowledge base stores documents as chunks (``Chunk``). The
:class:`Retriever` builds two indexes over them:

  * an inverted-index BM25 scorer for pure lexical retrieval
  * an embedding matrix for vector retrieval

``search()`` merges both with Reciprocal Rank Fusion, which needs no
parameter tuning and behaves sanely when one of the two signals is absent
(e.g. embeddings disabled, or the corpus was loaded before an API key
existed).

Chunking is deliberately simple: split on blank lines into paragraphs, hard
wrap anything longer than ``chunk_size``, then merge up to the budget with a
tail overlap. Good enough for markdown-ish docs, manuals and FAQ-style text;
no dependency on langchain's text splitters.

Both indexes grow incrementally (:meth:`Retriever.add` embeds only the new
chunks), so loading a 200-file corpus is one pass instead of 200 full
re-embeddings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .embeddings import HashEmbedding, cosine, similarity_matrix

# "编码" -> "编", "码" (character bigrams give Chinese its lexical signal)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)
_PARA_RE = re.compile(r"\n\s*\n")
# sentence ends, used to break up a paragraph that has no blank line in it.
# The lookbehind keeps the punctuation with the sentence it terminates.
_SENT_RE = re.compile(r"(?<=[。！？!?；;：:.\n])")

RRF_K = 60  # the constant from the original RRF paper; tuned, not tunable


def tokenize(text: str) -> list[str]:
    """Word/char tokens, lowercased. CJK characters become single tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    """One retrievable unit of text. ``id`` is a stable pkey within a store."""

    id: str
    text: str
    source: str = ""  # file path / URL / chapter, for citation
    meta: dict = field(default_factory=dict)

    # -- convenience view, avoids re-tokenising everywhere ------------------
    tokens: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_text(
        cls,
        id: str,
        text: str,
        source: str = "",
        meta: dict | None = None,
        pre_tokenized: bool = False,
    ) -> "Chunk":
        return cls(
            id=id,
            text=text,
            source=source,
            meta=meta or {},
            tokens=[] if pre_tokenized else tokenize(text),
        )

    @property
    def tag(self) -> str:
        """Short human-readable label used in the citation line."""
        if not self.source:
            return self.id
        page = self.meta.get("page")
        return f"{self.source}#p{page}" if page else self.source


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    *,
    source: str = "",
    chunk_size: int = 1000,
    overlap: int = 100,
    id_prefix: str = "",
    meta: dict | None = None,
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    Paragraphs are merged until they reach ``chunk_size``; every chunk after
    the first keeps the last ``overlap`` chars of its predecessor so a
    sentence that straddles a cut survives somewhere. A paragraph longer than
    the whole budget is wrapped on sentence boundaries first (see :func:`_fit`),
    so no chunk ever exceeds ``chunk_size`` by an unbounded amount.
    """
    chunk_size = max(32, int(chunk_size or 0))
    overlap = max(0, min(int(overlap or 0), chunk_size // 2))

    units: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if para:
            units.extend(_fit(para, chunk_size))

    chunks: list[Chunk] = []
    buf = ""
    idx = 0
    for unit in units:
        if buf and len(buf) + 1 + len(unit) > chunk_size:
            chunks.append(_make_chunk(buf, idx, id_prefix, source, meta))
            idx += 1
            # carry the tail over, but only while it still leaves room - an
            # overlap that alone busts the budget is worse than no overlap
            tail = buf[-overlap:].strip() if overlap else ""
            buf = tail if tail and len(tail) + 1 + len(unit) <= chunk_size else ""
        buf = f"{buf}\n{unit}".strip() if buf else unit
    if buf:
        chunks.append(_make_chunk(buf, idx, id_prefix, source, meta))
    return chunks


def _fit(para: str, chunk_size: int) -> list[str]:
    """Break one oversized paragraph into pieces that fit ``chunk_size``.

    Sentences are packed greedily; a single sentence longer than the whole
    budget is sliced on a fixed window (rare in prose, common in machine-made
    logs and CSV dumps). Sub-second returns ``[para]`` untouched.
    """
    if len(para) <= chunk_size:
        return [para]
    pieces: list[str] = []
    for sent in (s.strip() for s in _SENT_RE.split(para)):
        if not sent:
            continue
        while len(sent) > chunk_size:
            pieces.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if sent:
            pieces.append(sent)
    return pieces or [para[:chunk_size]]


def _make_chunk(buf: str, idx: int, id_prefix: str, source: str, meta: dict | None) -> Chunk:
    cid = f"{id_prefix}{idx}" if id_prefix else str(idx)
    return Chunk.from_text(cid, buf.strip(), source=source, meta={**(meta or {}), "chunk": idx})


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Index:
    """Inverted-index BM25 scorer that grows with the corpus.

    Standard Okapi BM25 with k1=1.5, b=0.75. ``add()`` appends postings in
    place, so ingesting a document is O(its own tokens) rather than
    O(whole corpus); ``score(query_tokens)`` stays O(|query| * df).
    """

    def __init__(self, chunks: Iterable[Chunk] = (), k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.dl: list[int] = []
        self._tokens = 0
        self.add(chunks)

    @property
    def n(self) -> int:
        """Document count, floored at 1 so an empty index can't divide by zero."""
        return max(len(self.dl), 1)

    @property
    def avgdl(self) -> float:
        return self._tokens / len(self.dl) if self.dl else 0.0

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index more documents; their positions follow the existing ones."""
        offset = len(self.dl)
        for i, c in enumerate(chunks):
            tokens = c.tokens if c.tokens else tokenize(c.text)
            self.dl.append(len(tokens))
            self._tokens += len(tokens)
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            for t, f in freq.items():
                self.postings.setdefault(t, []).append((offset + i, f))

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        """doc index -> BM25 score (only docs sharing >=1 term)."""
        scores: dict[int, float] = {}
        if not self.dl:
            return scores
        n, avgdl = self.n, self.avgdl
        k1, b = self.k1, self.b
        # per-token IDF is constant for the query, so weight each hit once
        idf_cache: dict[str, float] = {}
        for t in set(query_tokens):
            posting = self.postings.get(t)
            if not posting:
                continue
            if t not in idf_cache:
                df = len(posting)
                idf_cache[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_i, f in posting:
                dl = self.dl[doc_i]
                norm = 1 - b + b * dl / avgdl if avgdl else 1.0
                scores[doc_i] = scores.get(doc_i, 0.0) + idf_cache[t] * f * (k1 + 1) / (f + k1 * norm)
        return scores


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(
        self,
        chunks: Iterable[Chunk] | None = None,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        top_k: int = 5,
        *,
        use_embeddings: bool = True,
        min_score: float = 0.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        rrf_k: int = RRF_K,
    ):
        self.top_k = max(1, top_k)
        self.chunks: list[Chunk] = []
        # either an object with .embed (HashEmbedding / OpenAIEmbedding) or a
        # bare callable list[str] -> list[list[float]]; both are accepted.
        # The guard must NOT be `callable(embedder)`: embedder *objects* expose
        # .embed, not __call__, so a callable() check would silently discard the
        # very embedders that matter (a custom-dim HashEmbedding, or the whole
        # OpenAIEmbedding API path) and fall back to a default HashEmbedding.
        self.embedder = (
            embedder
            if embedder is not None and (hasattr(embedder, "embed") or callable(embedder))
            else HashEmbedding()
        )
        self.use_embeddings = use_embeddings
        self.min_score = float(min_score or 0.0)  # 0 disables the relevance gate
        self.chunk_size = max(32, int(chunk_size or 1000))
        self.chunk_overlap = max(0, int(chunk_overlap or 0))
        self.rrf_k = max(1, int(rrf_k or RRF_K))
        self._matrix: list[list[float]] = []
        self._bm25 = BM25Index()
        if chunks:
            self.add(chunks)

    # -- corpus mutation ----------------------------------------------------

    def add(self, chunks: Iterable[Chunk]) -> int:
        """Index more chunks. Embeds only the new ones; returns how many landed."""
        new = [c for c in chunks if c and c.text.strip()]
        if not new:
            return 0
        self.chunks.extend(new)
        self._bm25.add(new)
        if self.use_embeddings:
            self._matrix.extend(self._embed([c.text for c in new]))
        return len(new)

    def remove(self, chunk_ids: set[str] | Iterable[str]) -> int:
        """Drop chunks by id; returns how many were removed.

        The vectors are filtered positionally instead of recomputed, so
        removing a source doesn't bill an embedding call per surviving chunk.
        """
        dead = set(chunk_ids)
        if not dead:
            return 0
        keep = [i for i, c in enumerate(self.chunks) if c.id not in dead]
        removed = len(self.chunks) - len(keep)
        if not removed:
            return 0
        self.chunks = [self.chunks[i] for i in keep]
        self._matrix = [self._matrix[i] for i in keep if i < len(self._matrix)]
        self._bm25 = BM25Index(self.chunks)  # postings hold absolute indexes
        return removed

    def clear(self) -> None:
        self.chunks.clear()
        self._matrix = []
        self._bm25 = BM25Index()

    def chunk_for(self, cid: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == cid), None)

    @property
    def sources(self) -> list[str]:
        """Distinct source labels currently indexed, sorted."""
        return sorted({c.source for c in self.chunks if c.source})

    # -- query --------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid BM25 + vector search, merged with RRF.

        Returns ``(chunk, score)`` sorted by relevance, best first. Chunks
        scoring below ``min_score`` are dropped - the gate that turns "the
        retrieval was noise" into "知识库中未找到相关信息" instead of a
        confident hallucination.
        """
        k = self.top_k if k is None else max(1, k)
        gate = self.min_score if min_score is None else float(min_score)
        if not self.chunks:
            return []
        qt = tokenize(query)

        ranks: list[dict[int, float]] = [self._bm25.score(qt)] if qt else []
        if self.use_embeddings and self._matrix:
            qv = self._embed([query])[0]
            sims = similarity_matrix(self._matrix[: len(self.chunks)], qv)
            vec = {i: s for i, s in enumerate(sims) if s > 0.0}
            if vec:
                ranks.append(vec)
        ranks = [r for r in ranks if r]
        if not ranks:
            return []  # empty query, no shared terms, no vectors -> nothing to rank

        # normalise RRF to 0..1: a chunk that is #1 in every list scores 1.0,
        # so --threshold means "how much the retrievers agree" rather than
        # "1/(k+rank) times however many retrievers were enabled".
        fused = self._rrf(ranks)
        ceiling = len(ranks) / (self.rrf_k + 1)
        ranked = sorted(
            ((i, min(s / ceiling, 1.0)) for i, s in fused.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if gate > 0.0:
            ranked = [item for item in ranked if item[1] >= gate]
        return [(self.chunks[i], s) for i, s in ranked[:k]]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed through whichever protocol the embedder implements."""
        emb = self.embedder
        if hasattr(emb, "embed"):
            return emb.embed(texts)
        return list(emb(texts))

    # -- rank helpers -------------------------------------------------------

    def _rrf(self, ranked: list[dict[int, float]], limit: int | None = None) -> dict[int, float]:
        """Reciprocal Rank Fusion: 1/(k+rank) summed over each ranked list.

        ``limit`` caps how deep each list contributes (default 4*top_k): a chunk
        at rank 200 of the BM25 list adds ~1/260 of noise to the fusion and can
        never win a top-5 anyway, so trimming it is both faster and cleaner.
        Empty lists are dropped by the caller, so every list here contributes.
        """
        depth = self.top_k * 4 if limit is None else limit
        fused: dict[int, float] = {}
        for r in ranked:
            order = sorted(r.items(), key=lambda kv: kv[1], reverse=True)[:depth]
            for rank, (doc, score) in enumerate(order):
                if score <= 0.0:
                    continue
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.rrf_k + rank)
        return fused


__all__ = [
    "BM25Index",
    "Chunk",
    "Retriever",
    "RRF_K",
    "chunk_document",
    "tokenize",
]
