"""Knowledge base: load documents, persist the index, serve RAG retrievals.

A :class:`KnowledgeBase` owns the corpus and the :class:`Retriever`. It can be
built from:

  * a directory of documents - any format :mod:`corecoder.rag.loader` reads
    (txt / md / rst / csv / json / py / pdf / docx), via ``load_dir``
  * individual files (``add_file``) or raw strings (``add_text``)
  * a previously persisted bundle (``load`` / ``save``)

Persistence is one JSON bundle: a version header, the embedder fingerprint,
the per-source manifest and the chunks - plus the vectors, when the embedder
is one whose output is reproducible (the hashing default) or expensive to
regenerate (an API embedder). :meth:`load` reuses the stored vectors only if
the fingerprint matches the embedder it is handed; otherwise it re-embeds,
because mixing two vector spaces would quietly poison every score.

Sources are tracked so a corpus can be *synced* rather than rebuilt: adding a
file that is already indexed replaces its chunks, and an unchanged file
(mtime + content hash match) is skipped without spending an embedding call.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .embeddings import make_embedder, signature
from .loader import LoadedDoc, LoaderError, load_dir as loader_load_dir, load_file
from .retriever import Chunk, Retriever, chunk_document

BUNDLE_VERSION = 2  # 1 = bare {id: {...}} map, 2 = versioned header + vectors


@dataclass
class Answer:
    """Retrieval result for one query: context + where it came from."""

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def score(self) -> float:
        """Best retrieval score, 0.0 when nothing matched."""
        return self.scores[0] if self.scores else 0.0

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def low_confidence(self) -> bool:
        """True when the best match is too weak to trust an answer to.

        The gate is only meaningful when one was configured: with
        ``threshold == 0`` retrieval never filters, so nothing is "low".
        """
        return self.threshold > 0.0 and (not self.chunks or self.score < self.threshold)

    @property
    def sources(self) -> list[str]:
        """Distinct citation tags, in the order they were retrieved."""
        seen: list[str] = []
        for c in self.chunks:
            tag = c.tag
            if tag not in seen:
                seen.append(tag)
        return seen

    @property
    def context(self) -> str:
        """Flat context block for prompt assembly (source-tagged)."""
        return "\n\n".join(f"[{c.tag}]\n{c.text}" for c in self.chunks)

    def citations(self) -> list[dict]:
        """Machine-readable provenance for the UI / API layer."""
        return [
            {
                "id": c.id,
                "source": c.source,
                "tag": c.tag,
                "score": round(s, 6),
                "snippet": c.text[:200],
                "meta": c.meta,
            }
            for c, s in zip(self.chunks, self.scores)
        ]


class KnowledgeBase:
    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        top_k: int = 5,
        use_embeddings: bool = True,
        embedder=None,
        llm=None,
        embedding_kind: str = "hash",
        min_score: float = 0.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        """Build a KB.

        ``embedder`` wins over ``embedding_kind``; the latter exists so the CLI
        can say "hash" or "openai" and let :func:`make_embedder` wire it to
        ``llm`` (the same OpenAI-compatible client chat already uses).
        """
        self.chunk_size = max(32, int(chunk_size or 1000))
        self.chunk_overlap = max(0, int(chunk_overlap or 0))
        if retriever is None:
            if embedder is None and embedding_kind and embedding_kind != "hash":
                embedder = make_embedder(embedding_kind, llm=llm)
            retriever = Retriever(
                embedder=embedder,
                top_k=top_k,
                use_embeddings=use_embeddings,
                min_score=min_score,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        self.retriever = retriever
        # source -> {"chunks": n, "hash": digest, "mtime": s, "added_at": s}
        self.sources: dict[str, dict] = {}
        self.last_errors: list[str] = []   # skips from the last load_dir()
        self._seq = 0
        self._last_saved: str | None = None

    # -- ingestion ----------------------------------------------------------

    def add_text(self, text: str, source: str = "memory", **meta) -> list[Chunk]:
        """Chunk ``text`` and index it under ``source`` (replacing any prior copy)."""
        chunks = chunk_document(
            text,
            source=source,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            id_prefix=self._next_prefix(),
            meta=meta or None,
        )
        if not chunks:
            return []
        self.replace_source(source, chunks)
        self._remember(source, text, len(chunks))
        return chunks

    def add_doc(self, doc: LoadedDoc) -> list[Chunk]:
        """Index one LoadedDoc, keeping its page/paragraph metadata."""
        return self.add_text(doc.text, source=doc.source, **(doc.meta or {}))

    def add_file(self, path: str | Path, **meta) -> list[Chunk]:
        """Add one file (txt/md/csv/json/pdf/docx). Returns the chunks created."""
        p = Path(path).expanduser()
        docs = load_file(p, **{k: v for k, v in meta.items() if k in ("strip_markdown",)})
        extra = {k: v for k, v in meta.items() if k not in ("strip_markdown",)}
        made: list[Chunk] = []
        for d in docs:
            made.extend(self.add_text(d.text, source=d.source, **{**(d.meta or {}), **extra}))
        return made

    def load_dir(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        verbose: bool = False,
        only_new: bool = True,
    ) -> int:
        """Load every supported file under ``directory``. Returns chunk count.

        Unchanged files (same size+mtime, same text hash) are skipped when
        ``only_new`` is set, which is what makes a repeated ingest of a large
        corpus cheap. Load failures are reported, never fatal.
        """
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {root}")

        errors: list[str] = []
        docs = loader_load_dir(
            root, recursive=recursive, on_error=lambda p, msg: errors.append(f"{p}: {msg}")
        )
        total = 0
        for doc in docs:
            if doc.is_empty:
                continue
            digest = _hash(doc.text)
            prev = self.sources.get(doc.source)
            if only_new and prev and prev.get("hash") == digest:
                continue
            made = self.add_text(doc.text, source=doc.source, **(doc.meta or {}))
            if not made:
                continue
            self.sources[doc.source].update(hash=digest, mtime=_mtime(doc.source))
            total += len(made)
            if verbose:
                print(f"  + {doc.source} -> {len(made)} chunks")
        if verbose and errors:
            for e in errors:
                print(f"  ! {e}")
        self.last_errors = errors
        return total

    def replace_source(self, source: str, chunks: list[Chunk]) -> None:
        """Swap a source's chunks in atomically (id namespace is the source)."""
        self.remove_source(source)
        self.retriever.add(chunks)
        self.sources[source] = {"chunks": len(chunks)}

    def remove_source(self, source: str) -> int:
        """Drop everything indexed under ``source``; returns chunks removed."""
        dead = {c.id for c in self.retriever.chunks if c.source == source}
        removed = self.retriever.remove(dead) if dead else 0
        self.sources.pop(source, None)
        return removed

    # -- query --------------------------------------------------------------

    def ask(self, query: str, k: int | None = None, *, min_score: float | None = None) -> Answer:
        """Retrieve the top-k chunks for a question, threshold applied."""
        gate = self.retriever.min_score if min_score is None else float(min_score)
        results = self.retriever.search(query, k=k, min_score=min_score)
        return Answer(
            query=query,
            chunks=[c for c, _ in results],
            scores=[s for _, s in results],
            threshold=gate,
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path, *, with_vectors: bool = True, indent: int | None = None) -> Path:
        """Write the bundle. Vectors are included when the index has them.

        ``indent`` defaults to None (compact): a vector dump pretty-printed is
        megabytes of whitespace for no readability gain - humans read the
        manifest, not the floats.
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        r = self.retriever
        vectors = [list(map(float, v)) for v in r._matrix[: len(r.chunks)]] if with_vectors and r._matrix else []
        bundle = {
            "version": BUNDLE_VERSION,
            "saved_at": time.time(),
            "embedder": signature(r.embedder),
            "use_embeddings": bool(r.use_embeddings),
            "params": {
                "top_k": r.top_k,
                "min_score": r.min_score,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "sources": self.sources,
            "chunks": [
                {"id": c.id, "text": c.text, "source": c.source, "meta": c.meta}
                for c in r.chunks
            ],
            "vectors": vectors,
        }
        p.write_text(json.dumps(bundle, ensure_ascii=False, indent=indent), encoding="utf-8")
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder=None,
        llm=None,
        embedding_kind: str = "hash",
        reuse_vectors: bool = True,
        **kwargs,
    ) -> "KnowledgeBase":
        """Rebuild from a bundle written by :meth:`save`.

        Unknown-but-newer versions load anyway: the chunk list is the part we
        need, and refusing to start over a formatting bump would be spiteful.
        Vectors are reused only when the stored embedder fingerprint matches
        the one we are handed - otherwise they are recomputed.
        """
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        kb = cls(embedder=embedder, llm=llm, embedding_kind=embedding_kind, **kwargs)

        if isinstance(raw, dict) and "chunks" in raw:
            # an explicit argument beats the stored one: the caller is the
            # person holding the keyboard right now
            params = dict(raw.get("params") or {})
            for key, attr in (("top_k", "top_k"), ("min_score", "min_score")):
                if key not in kwargs:
                    setattr(kb.retriever, attr, params.get(key, getattr(kb.retriever, attr)))
            for key in ("chunk_size", "chunk_overlap"):
                if key not in kwargs and key in params:
                    setattr(kb, key, int(params[key]))
            kb.retriever.top_k = max(1, int(kb.retriever.top_k))
            stored = signature(kb.retriever.embedder)
            compatible = reuse_vectors and raw.get("embedder") == stored
            chunks = [Chunk.from_text(c["id"], c["text"], source=c.get("source", ""), meta=c.get("meta") or {}) for c in raw["chunks"]]
            kb.retriever.add(chunks)
            vectors = raw.get("vectors") or []
            if compatible and len(vectors) == len(chunks):
                kb.retriever._matrix = [list(map(float, v)) for v in vectors]
            elif kb.retriever.use_embeddings and kb.retriever._matrix:
                # we embedded during add(); keep only the aligned prefix
                kb.retriever._matrix = kb.retriever._matrix[: len(chunks)]
            kb.sources = dict(raw.get("sources") or {})
            kb._seq = _next_seq(c["id"] for c in raw["chunks"]) or len(kb.retriever.chunks)
            return kb

        # version 1: a bare {chunk_id: {text, source, meta}} map
        chunks = [
            Chunk.from_text(cid, d["text"], source=d.get("source", ""), meta=d.get("meta") or {})
            for cid, d in (raw or {}).items()
        ]
        kb.retriever.add(chunks)
        for c in chunks:
            kb.sources.setdefault(c.source, {"chunks": 0})
            kb.sources[c.source]["chunks"] += 1
        kb._seq = _next_seq(c.id for c in chunks) or len(chunks)
        return kb

    def save_if_changed(self, path: str | Path) -> bool:
        """Persist only when the corpus moved since the last write."""
        p = Path(path).expanduser()
        stamp = _hash(json.dumps([c.id for c in self.retriever.chunks]))
        if p.exists() and self._last_saved == stamp:
            return False
        self.save(p)
        self._last_saved = stamp
        return True

    # -- helpers ------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.retriever.chunks)

    @property
    def source_names(self) -> list[str]:
        return sorted(self.sources)

    def stats(self) -> dict:
        r = self.retriever
        return {
            "chunks": self.size,
            "sources": len(self.sources),
            "vectors": len(r._matrix),
            "embedder": signature(r.embedder),
            "top_k": r.top_k,
            "min_score": r.min_score,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _next_prefix(self) -> str:
        """Namespace chunk ids by a monotonic counter, so removals can't collide."""
        prefix = f"c{self._seq}-"
        self._seq += 1
        return prefix

    def _remember(self, source: str, text: str, n_chunks: int) -> None:
        self.sources[source] = {
            "chunks": n_chunks,
            "hash": _hash(text),
            "mtime": _mtime(source),
            "added_at": time.time(),
        }


_ID_SEQ_RE = re.compile(r"^c(\d+)-")


def _next_seq(ids: Iterable[str]) -> int:
    """Smallest chunk-id counter that can't collide with the ids already stored."""
    seen = [int(m.group(1)) for i in ids if (m := _ID_SEQ_RE.match(str(i)))]
    return max(seen) + 1 if seen else 0


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _mtime(source: str) -> float | None:
    """Best-effort: sources are often paths, but a PDF page tag is not."""
    try:
        return Path(source).stat().st_mtime
    except OSError:
        return None


__all__ = ["Answer", "BUNDLE_VERSION", "KnowledgeBase", "LoaderError"]
