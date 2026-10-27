"""Tests for the RAG 智能客服 subpackage (corecoder.rag).

Everything here runs offline: the hashing embedder is deterministic and the
LLM is a fake, so no API key or network is needed.
"""

import json
from pathlib import Path

import pytest

from corecoder.llm import LLMResponse
from corecoder.rag import (
    BM25Index,
    Chunk,
    HashEmbedding,
    KnowledgeBase,
    RAGAgent,
    Retriever,
    chunk_document,
    cosine,
    tokenize,
)
from corecoder.rag.agent import NO_ANSWER
from corecoder.rag.embeddings import make_embedder, signature, similarity_matrix
from corecoder.rag.loader import LoaderError, load_dir, load_file

# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def test_hash_embedding_deterministic_and_normalised():
    emb = HashEmbedding(dim=128)
    v1 = emb.embed(["如何配置 API Key"])[0]
    v2 = emb.embed(["如何配置 API Key"])[0]
    assert v1 == v2
    assert len(v1) == 128
    assert abs(cosine(v1, v2) - 1.0) < 1e-9


def test_hash_embedding_overlap_beats_unrelated():
    emb = HashEmbedding()
    a, b, c = emb.embed(["退款流程 七天无理由", "退款流程 原路退回", "量子物理 薛定谔方程"])
    assert cosine(a, b) > cosine(a, c)


def test_cosine_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


def test_similarity_matrix_matches_cosine():
    emb = HashEmbedding(dim=64)
    rows = emb.embed(["alpha beta", "gamma delta"])
    q = emb.embed(["alpha"])[0]
    sims = similarity_matrix(rows, q)
    assert sims == pytest.approx([cosine(r, q) for r in rows])


def test_make_embedder_falls_back_to_hash():
    # unknown kind / missing llm must degrade to the offline embedder
    assert isinstance(make_embedder("hash"), HashEmbedding)
    assert isinstance(make_embedder("openai"), HashEmbedding)  # no llm given
    assert isinstance(make_embedder("bogus-kind"), HashEmbedding)


def test_signature_distinguishes_embedders():
    assert signature(HashEmbedding(dim=256)) != signature(HashEmbedding(dim=128))
    assert signature(HashEmbedding())["kind"] == "hash"


# ---------------------------------------------------------------------------
# tokenisation & chunking
# ---------------------------------------------------------------------------


def test_tokenize_words_and_cjk():
    toks = tokenize("如何退款 API-Key v2")
    assert "退款" in toks and "如" in toks and "何" in toks
    assert "api" in toks and "key" in toks and "v2" in toks


def test_chunk_document_respects_budget_and_overlaps():
    paras = [f"第{i}段。" + "内容" * 40 for i in range(6)]
    chunks = chunk_document("\n\n".join(paras), chunk_size=100, overlap=20, source="doc.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.source == "doc.md" for c in chunks)
    # the overlap leaves a shared tail between neighbours
    assert any(tail in chunks[i + 1].text for i in range(len(chunks) - 1)
               for tail in [chunks[i].text[-15:]])


def test_chunk_document_empty_text():
    assert chunk_document("   \n\n  ") == []


def test_chunk_document_oversized_single_paragraph():
    chunks = chunk_document("x" * 5000, chunk_size=200, overlap=0)
    assert len(chunks) > 1 and all(len(c.text) <= 200 for c in chunks)


# ---------------------------------------------------------------------------
# BM25 + retriever
# ---------------------------------------------------------------------------


def _chunks():
    return [
        Chunk.from_text("a", "退款流程：七天内可申请无理由退款，原路退回。", source="refund.md"),
        Chunk.from_text("b", "配送时间：默认顺丰发货，三个工作日送达。", source="ship.md"),
        Chunk.from_text("c", "API key configuration goes into the .env file.", source="env.md"),
    ]


def test_bm25_scores_matching_doc_highest():
    idx = BM25Index(_chunks())
    scores = idx.score(tokenize("退款"))
    assert max(scores, key=scores.get) == 0


def test_retriever_hybrid_search_ranks_relevant_first():
    r = Retriever(_chunks(), top_k=3)
    hits = r.search("如何退款？")
    assert hits and hits[0][0].id == "a"
    assert all(0.0 < s <= 1.0 for _, s in hits)


def test_retriever_min_score_gate():
    r = Retriever(_chunks(), min_score=1.1)  # unreachable -> everything filtered
    assert r.search("退款") == []


def test_retriever_add_remove():
    r = Retriever()
    assert r.add(_chunks()) == 3
    assert r.remove({"b"}) == 1
    assert [c.id for c in r.chunks] == ["a", "c"]
    assert len(r._matrix) == 2  # vectors stay aligned without re-embedding
    assert r.remove({"nope"}) == 0


def test_retriever_empty_query():
    r = Retriever(_chunks())
    assert r.search("   ") == []


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_load_file_txt_md_json(tmp_path):
    (tmp_path / "a.txt").write_text("纯文本知识。", encoding="utf-8")
    (tmp_path / "b.md").write_text("# 标题\n\n[链接文字](http://x) **加粗**", encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps({"fee": {"price": 10}}), encoding="utf-8")

    assert load_file(tmp_path / "a.txt")[0].text == "纯文本知识。"
    md = load_file(tmp_path / "b.md")[0].text
    assert "链接文字" in md and "http://x" not in md and "#" not in md.splitlines()[0]
    assert "fee.price: 10" in load_file(tmp_path / "c.json")[0].text


def test_load_file_errors(tmp_path):
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "x.xyz").write_text("data", encoding="utf-8")
    with pytest.raises(LoaderError):
        load_file(tmp_path / "empty.txt")
    with pytest.raises(LoaderError):
        load_file(tmp_path / "x.xyz")
    with pytest.raises(LoaderError):
        load_file(tmp_path / "missing.txt")


def test_load_dir_collects_errors(tmp_path):
    (tmp_path / "ok.txt").write_text("好的内容", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("", encoding="utf-8")
    (tmp_path / "skip.xyz").write_text("junk", encoding="utf-8")
    errors: list = []
    docs = load_dir(tmp_path, on_error=lambda p, m: errors.append(str(p)))
    assert [d.source for d in docs] == [str(tmp_path / "ok.txt")]
    assert len(errors) == 1  # bad.txt raised; skip.xyz was filtered silently


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------


def test_kb_add_and_ask():
    kb = KnowledgeBase()
    kb.add_text("退款七天内无理由，原路退回。", source="refund.md")
    kb.add_text("默认顺丰发货，三个工作日送达。", source="ship.md")
    ans = kb.ask("怎么退款")
    assert not ans.is_empty
    assert ans.chunks[0].source == "refund.md"
    assert ans.sources == ["refund.md"]
    assert "[refund.md]" in ans.context
    assert ans.citations()[0]["source"] == "refund.md"


def test_kb_replace_source_on_readd():
    kb = KnowledgeBase()
    kb.add_text("旧内容 A", source="s")
    kb.add_text("新内容 B", source="s")
    assert kb.size == 1
    assert "新内容" in kb.retriever.chunks[0].text


def test_kb_load_dir_incremental(tmp_path):
    (tmp_path / "one.md").write_text("第一篇知识。", encoding="utf-8")
    kb = KnowledgeBase()
    first = kb.load_dir(tmp_path)
    assert first > 0
    assert kb.load_dir(tmp_path) == 0  # unchanged -> skipped
    (tmp_path / "two.md").write_text("第二篇知识。", encoding="utf-8")
    assert kb.load_dir(tmp_path) > 0


def test_kb_save_load_roundtrip(tmp_path):
    kb = KnowledgeBase(chunk_size=200, chunk_overlap=20)
    kb.add_text("退款政策：七天无理由。", source="refund.md")
    kb.add_text("Shipping takes three business days.", source="ship.md")
    bundle = kb.save(tmp_path / "kb.json")

    kb2 = KnowledgeBase.load(bundle)
    assert kb2.size == kb.size
    assert kb2.sources.keys() == kb.sources.keys()
    # same embedder fingerprint -> vectors reused verbatim, not recomputed
    assert kb2.retriever._matrix == kb.retriever._matrix
    assert kb2.ask("退款").chunks[0].source == "refund.md"
    assert kb2.chunk_size == 200 and kb2.chunk_overlap == 20


def test_kb_load_reembeds_on_fingerprint_mismatch(tmp_path):
    kb = KnowledgeBase()
    kb.add_text("一些用于测试的文本内容。", source="s")
    bundle = kb.save(tmp_path / "kb.json")

    kb2 = KnowledgeBase.load(bundle, embedder=HashEmbedding(dim=64))
    assert kb2.size == kb.size
    assert len(kb2.retriever._matrix[0]) == 64  # new space, not the stored 256-d


def test_kb_load_v1_bare_map(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"0": {"text": "老格式的内容", "source": "legacy"}}), encoding="utf-8")
    kb = KnowledgeBase.load(p)
    assert kb.size == 1
    assert "legacy" in kb.sources


def test_kb_remove_source_and_save_if_changed(tmp_path):
    kb = KnowledgeBase()
    kb.add_text("甲", source="a")
    kb.add_text("乙", source="b")
    path = tmp_path / "kb.json"
    assert kb.save_if_changed(path) is True
    assert kb.save_if_changed(path) is False
    assert kb.remove_source("a") == 1
    assert kb.save_if_changed(path) is True
    assert "a" not in kb.sources


# ---------------------------------------------------------------------------
# RAGAgent (fake LLM, no network)
# ---------------------------------------------------------------------------


class FakeLLM:
    """The slice of the LLM interface RAGAgent actually uses."""

    def __init__(self, reply: str = "根据知识库：七天内可退款。"):
        self.model = "fake-model"
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(messages)
        if on_token:
            on_token(self.reply)
        return LLMResponse(content=self.reply, prompt_tokens=11, completion_tokens=7)


def _kb():
    kb = KnowledgeBase()
    kb.add_text("退款流程：七天内可申请无理由退款，原路退回。", source="refund.md")
    kb.add_text("配送时间：默认顺丰发货，三个工作日送达。", source="ship.md")
    return kb


def test_agent_grounded_answer():
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, knowledge=_kb())
    turn = agent.answer("如何退款？")
    assert turn.answer == llm.reply
    assert turn.grounded and turn.sources == ["refund.md"]
    assert turn.prompt_tokens == 11 and turn.completion_tokens == 7
    # the context rides in the user message, system prompt stays the RAG one
    msgs = llm.calls[0]
    assert msgs[0]["role"] == "system"
    assert "知识库内容" in msgs[-1]["content"] and "refund.md" in msgs[-1]["content"]
    assert agent.last_citations()[0]["source"] == "refund.md"


def test_agent_refuses_without_llm_when_gate_fails():
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, knowledge=_kb(), min_score=1.1)  # nothing can pass
    turn = agent.answer("如何退款？")
    assert turn.answer == NO_ANSWER
    assert turn.grounded is False
    assert llm.calls == []  # refusal is free: the model was never asked


def test_agent_refuses_on_empty_kb_with_gate():
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, knowledge=KnowledgeBase(), min_score=0.5)
    assert agent.answer("随便问点什么").answer == NO_ANSWER
    assert llm.calls == []


def test_agent_empty_question_skips_everything():
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, knowledge=_kb())
    turn = agent.answer("   ")
    assert turn.answer == "" and llm.calls == []


def test_agent_history_and_reset():
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, knowledge=_kb(), max_history=2)
    agent.chat("如何退款？")
    agent.chat("那配送呢？")
    second = llm.calls[1]
    # system + Q1/A1 history + current question
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "如何退款？"
    agent.reset()
    assert agent.history == [] and agent.turns == []


def test_agent_runtime_add_and_forget():
    agent = RAGAgent(llm=FakeLLM(), knowledge=_kb())
    assert agent.add("会员积分规则：消费一元积一分。", source="vip.md") == 1
    assert "vip.md" in agent.knowledge.sources
    assert agent.forget("vip.md") == 1
    assert "vip.md" not in agent.knowledge.sources


def test_agent_stats():
    agent = RAGAgent(llm=FakeLLM(), knowledge=_kb())
    st = agent.stats()
    assert st["model"] == "fake-model"
    assert st["chunks"] == agent.knowledge.size and st["sources"] == 2


# ---------------------------------------------------------------------------
# index_doc (offline indexer entry point)
# ---------------------------------------------------------------------------


def test_index_doc_build_list_rm(tmp_path, monkeypatch, capsys):
    from corecoder.rag.index_doc import main

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "faq.md").write_text("# 退款\n\n七天内无理由退款。", encoding="utf-8")
    index = tmp_path / "kb.json"

    assert main([str(docs), "--index", str(index)]) == 0
    assert index.is_file()
    built = json.loads(index.read_text(encoding="utf-8"))
    assert built["chunks"] and built["embedder"]["kind"] == "hash"

    # second run: nothing changed, index untouched
    before = index.read_text(encoding="utf-8")
    assert main([str(docs), "--index", str(index)]) == 0
    assert index.read_text(encoding="utf-8") == before

    # --list prints the manifest
    assert main(["--index", str(index), "--list"]) == 0
    assert "faq.md" in capsys.readouterr().out

    # --rm drops a source and persists
    src = next(iter(json.loads(before)["sources"]))
    assert main(["--index", str(index), "--rm", src]) == 0
    assert json.loads(index.read_text(encoding="utf-8"))["chunks"] == []

    # nothing to do -> usage error
    assert main(["--index", str(tmp_path / "nope.json")]) == 2


def test_index_doc_api_embedding_needs_key(tmp_path, monkeypatch):
    from corecoder.rag.index_doc import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CORECODER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CORECODER_EMBEDDING_API_KEY", raising=False)
    assert main(["--embedding", "openai", "--index", str(tmp_path / "kb.json"),
                 "-f", __file__]) == 1
