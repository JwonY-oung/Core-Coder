"""Standalone indexer: build/refresh the KB bundle without asking the LLM.

``corecoder rag`` ingests and chats in one go; this module is the half that
only ingests. It exists so a corpus can be (re)built offline - on a build box,
in CI, before the API key is even provisioned - and the resulting bundle
reused via ``corecoder rag --index``:

    python -m corecoder.rag.index_doc docs/ --index ~/.corecoder/kb/index.json
    python -m corecoder.rag.index_doc -f manual.pdf -f faq.md --index kb.json
    python -m corecoder.rag.index_doc --index kb.json --list
    python -m corecoder.rag.index_doc --index kb.json --rm old.md

No LLM, no network: the hashing embedder is the default (``--embedding openai``
is accepted but needs an API key, same as the chat CLI). The bundle format is
exactly :meth:`KnowledgeBase.save`, so chat and indexing stay compatible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .knowledge import KnowledgeBase
from .loader import LoaderError


def build_parser(prog: str = "python -m corecoder.rag.index_doc") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="离线构建/更新 RAG 知识库索引（不发起任何 LLM 请求）",
    )
    p.add_argument("paths", nargs="*", metavar="DIR", help="要摄入的知识目录，可多个")
    p.add_argument("-f", "--file", action="append", metavar="FILE", help="追加单个文件，可重复")
    p.add_argument("--index", metavar="PATH", help="索引 JSON 路径（默认 $CORECODER_KB_INDEX）")
    p.add_argument("--embedding", choices=("hash", "openai"), default=None,
                   help="向量来源（默认 $CORECODER_RAG_EMBEDDING 或 hash）")
    p.add_argument("--no-embeddings", action="store_true", help="关闭向量，仅 BM25")
    p.add_argument("--chunk-size", type=int, help="切片长度（默认 1000）")
    p.add_argument("--chunk-overlap", type=int, help="切片重叠（默认 100）")
    p.add_argument("-k", "--top-k", type=int, help="写入 bundle 的默认召回数")
    p.add_argument("--threshold", type=float, help="写入 bundle 的默认相关度阈值")
    p.add_argument("--list", action="store_true", help="列出索引中的来源后退出")
    p.add_argument("--rm", action="append", metavar="SOURCE", help="从索引中删除某来源，可重复")
    p.add_argument("--rebuild", action="store_true", help="忽略已有索引，从零重建")
    p.add_argument("-v", "--verbose", action="store_true", help="逐文件打印进度")
    return p


def main(argv: list[str] | None = None) -> int:
    # Config only for env-driven defaults; it never touches the network.
    from corecoder.config import Config

    args = build_parser().parse_args(argv)
    config = Config.from_env()
    index = args.index or config.kb_index

    use_emb = not args.no_embeddings
    kind = "hash" if not use_emb else (args.embedding or config.kb_embedding)
    want_api = kind in ("openai", "litellm", "api")

    # an API embedder must be wired in from the very start: bolting it onto a
    # KB that already embedded with hash would mix two vector spaces
    llm = None
    if want_api:
        llm = _make_embed_llm(config)
        if llm is None:
            print("--embedding openai 需要 API Key（OPENAI_API_KEY / CORECODER_API_KEY）")
            return 1

    kwargs: dict = {
        "llm": llm,
        "embedding_kind": kind,
        "use_embeddings": use_emb,
        "top_k": args.top_k or config.kb_top_k,
        "min_score": args.threshold if args.threshold is not None else config.kb_threshold,
        "chunk_size": args.chunk_size or config.kb_chunk_size,
        "chunk_overlap": args.chunk_overlap if args.chunk_overlap is not None else config.kb_chunk_overlap,
    }

    kb: KnowledgeBase | None = None
    if not args.rebuild and Path(index).is_file():
        try:
            kb = KnowledgeBase.load(index, **kwargs)
            print(f"已加载索引 {index} -> {kb.size} chunks / {len(kb.sources)} 来源")
        except Exception as e:  # noqa: BLE001 - a corrupt bundle is rebuilt, not fatal
            print(f"索引 {index} 读取失败（{e}），改为重新构建")
            kb = None
    if kb is None:
        kb = KnowledgeBase(**kwargs)

    if args.list:
        _print_sources(kb)
        return 0

    removed = 0
    for src in args.rm or []:
        n = kb.remove_source(src)
        removed += n
        print(f"已删除 {src}（{n} chunks）" if n else f"未找到来源: {src}")

    if not args.paths and not args.file and not args.rm:
        print("没有要摄入的内容：请给出目录、-f FILE 或 --rm SOURCE（--help 查看用法）")
        return 2

    changed = bool(removed)
    for f in args.file or []:
        try:
            n = len(kb.add_file(f))
        except (LoaderError, OSError) as e:
            print(f"{f}: {e}")
            continue
        changed = changed or n > 0
        print(f"已加载 {f} -> {n} chunks" if n else f"{f}: 没有可用内容")

    for d in args.paths:
        try:
            n = kb.load_dir(d, verbose=args.verbose)
        except NotADirectoryError as e:
            print(str(e))
            continue
        changed = changed or n > 0
        print(f"已加载 {d} -> 新增 {n} chunks")
        for err in kb.last_errors:
            print(f"跳过: {err}")

    if changed:
        kb.save(index)
        print(f"索引已保存: {index}（{kb.size} chunks / {len(kb.sources)} 来源）")
    else:
        print("内容无变化，索引未改动。")
    return 0


def _make_embed_llm(config):
    """An LLM-shaped client for embeddings only; None when no key is around."""
    key = config.api_key or config.embedding_api_key
    if not key:
        return None
    from corecoder.llm import LLM

    return LLM(
        model=config.model,
        api_key=key,
        base_url=config.base_url,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
    )


def _print_sources(kb: KnowledgeBase) -> None:
    if not kb.sources:
        print("索引为空。")
        return
    for src in kb.source_names:
        info = kb.sources.get(src) or {}
        print(f"  {src}  ({info.get('chunks', '?')} chunks)")


if __name__ == "__main__":
    raise SystemExit(main())
