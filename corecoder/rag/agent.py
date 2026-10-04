"""RAG-aware Q&A agent, reusing CoreCoder's LLM class.

This is deliberately NOT a second agent loop. It builds on the exact LLM
interface the coding agent uses (``llm.chat(messages, tools=..., on_token=)``)
so that:

  * the same model, API key, base URL, budget and fallback chain just work
  * the caller keeps streaming, token accounting and BudgetExceededError
  * ``RAGAgent`` can plug into the existing REPL instead of inventing one

The loop is: question -> retrieve top-k -> build the prompt from
:data:`corecoder.prompt.RAG_PROMPT` -> chat -> remember the turn. Retrieval is
per-turn and the history carries only the visible Q/A, so a follow-up resolves
against earlier answers without dragging stale chunks forward.

When retrieval comes back empty (or below the configured score gate) the LLM
is never called: the refusal costs nothing and can't hallucinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from corecoder.llm import LLM, LLMResponse
from corecoder.prompt import RAG_PROMPT

from .knowledge import Answer, KnowledgeBase

NO_ANSWER = "知识库中未找到相关信息，建议联系人工客服。"

# the context block rides in the user message, not the system prompt: the
# system turn stays identical for every question, so a provider that caches
# prefixes still hits its cache, and the history keeps one clean Q/A pair
# per round instead of a retrieval dump nobody asked for.
_TEMPLATE = "{question}\n\n知识库内容：\n{context}"


@dataclass
class Turn:
    """One answered question, kept for history and for the UI."""

    question: str
    answer: str
    chunks: list = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    grounded: bool = True
    threshold: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "scores": [round(s, 6) for s in self.scores],
            "threshold": self.threshold,
            "grounded": self.grounded,
        }


class RAGAgent:
    def __init__(
        self,
        llm: LLM,
        knowledge: KnowledgeBase | None = None,
        *,
        top_k: int | None = None,
        system: str | None = None,
        knowledge_dir: str | None = None,
        stream: bool = False,
        min_score: float | None = None,
        max_history: int = 6,
        no_answer: str = NO_ANSWER,
    ):
        self.llm = llm
        self.knowledge = knowledge or KnowledgeBase(
            top_k=top_k or 5, llm=llm, embedding_kind=_env_kind()
        )
        if top_k is not None:
            self.knowledge.retriever.top_k = max(1, top_k)
        if min_score is not None:
            self.knowledge.retriever.min_score = float(min_score)
        if knowledge_dir and not self.knowledge.size:
            self.load_dir(knowledge_dir)
        self.system = system or RAG_PROMPT
        self.stream = stream
        self.max_history = max(0, int(max_history))
        self.no_answer = no_answer
        self.turns: list[Turn] = []

    # -- knowledge management ----------------------------------------------

    def load_dir(self, directory: str, verbose: bool = False) -> int:
        """Load a knowledge directory; returns the number of chunks added."""
        return self.knowledge.load_dir(directory, verbose=verbose)

    def add(self, text: str, source: str = "memory") -> int:
        """Index a snippet at runtime (chat-time feedback, hot fixes)."""
        return len(self.knowledge.add_text(text, source=source))

    def forget(self, source: str) -> int:
        """Drop every chunk from ``source``; returns how many went away."""
        return self.knowledge.remove_source(source)

    @property
    def history(self) -> list[dict]:
        """Past turns in OpenAI message form, oldest first."""
        msgs: list[dict] = []
        for t in self.turns[-self.max_history :] if self.max_history else []:
            msgs.append({"role": "user", "content": t.question})
            msgs.append({"role": "assistant", "content": t.answer})
        return msgs

    def reset(self) -> None:
        """Forget the conversation (the corpus stays indexed)."""
        self.turns.clear()

    # -- chat ----------------------------------------------------------------

    def chat(
        self,
        question: str,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """One grounded answer. Retrieves context, then asks the LLM."""
        return self.answer(question, on_token=on_token, tools=tools).answer

    def answer(
        self,
        question: str,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
        k: int | None = None,
    ) -> Turn:
        """Like :meth:`chat`, but returns the retrieval alongside the text."""
        question = (question or "").strip()
        if not question:
            return Turn(question=question, answer="", grounded=False)

        found: Answer = self.knowledge.ask(question, k=k)
        if found.low_confidence:
            # no retrieval, no request: the refusal is free and unfakeable
            turn = Turn(
                question=question,
                answer=self.no_answer,
                scores=found.scores,
                threshold=found.threshold,
                grounded=False,
            )
            self.turns.append(turn)
            return turn

        messages = self.messages_for(question, found)
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        if on_token is not None or self.stream:
            kwargs["on_token"] = on_token or (lambda t: None)

        resp: LLMResponse = self.llm.chat(messages=messages, **kwargs)
        turn = Turn(
            question=question,
            answer=resp.content or "",
            chunks=found.chunks,
            scores=found.scores,
            sources=found.sources,
            threshold=found.threshold,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )
        self.turns.append(turn)
        return turn

    def messages_for(self, question: str, found: Answer) -> list[dict]:
        """system + past turns + (context, question). Rebuilt every call so a
        stale retrieval can never leak into the next one."""
        user = _TEMPLATE.format(question=question, context=found.context)
        return [
            {"role": "system", "content": self.system},
            *self.history,
            {"role": "user", "content": user},
        ]

    def last_citations(self) -> list[dict]:
        """Provenance for the most recent answered turn."""
        for t in reversed(self.turns):
            if t.chunks:
                return [
                    {"source": c.source or c.id, "tag": c.tag, "score": s, "snippet": c.text[:200]}
                    for c, s in zip(t.chunks, t.scores)
                ]
        return []

    def stats(self) -> dict:
        return {
            "model": self.llm.model,
            "turns": len(self.turns),
            **self.knowledge.stats(),
        }


def _env_kind() -> str:
    """Default embedder for a KB built without one; env-driven, offline-safe."""
    import os

    return (os.getenv("CORECODER_RAG_EMBEDDING") or "hash").strip().lower()


__all__ = ["NO_ANSWER", "RAGAgent", "Turn"]
