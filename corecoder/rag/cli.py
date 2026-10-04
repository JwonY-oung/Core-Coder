"""CLI entry for the RAG 智能客服 (``corecoder rag``).

Reuses the exact LLM construction from the main CLI: Config.from_env() +
LLM (or LiteLLM), so model / API key / base URL / budget / fallback /
embedding endpoint behave identically to the coding agent.

Usage:
    corecoder rag --knowledge docs -q "如何配置 API Key?"
    corecoder rag --file manual.md                 # 交互式问答
    corecoder rag --knowledge docs --index ~/.corecoder/kb/index.json
    corecoder rag --index INDEX --list             # 查看已索引来源
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from corecoder.config import Config
from corecoder.llm import LLM, LiteLLM

from .agent import NO_ANSWER, RAGAgent
from .knowledge import KnowledgeBase

console = Console()

HELP_LINES = (
    "命令: /sources 列出来源  /add FILE 添加  /rm SOURCE 删除  "
    "/save 保存索引  /tokens 用量  /reset 清空对话  quit 退出"
)


def add_rag_args(parser) -> None:
    """Register the ``corecoder rag`` options on a subparser.

    Lives here (not in corecoder/cli.py) so the RAG surface area stays inside
    the RAG package; the main CLI only imports this one function.
    """
    parser.add_argument("--knowledge", metavar="DIR", help="知识库目录（txt/md/rst/csv/json/pdf/docx）")
    parser.add_argument("-f", "--file", action="append", metavar="FILE",
                        help="追加单个知识文件，可重复")
    parser.add_argument("--index", metavar="PATH",
                        help="索引 JSON 缓存路径（默认 $CORECODER_KB_INDEX）")
    parser.add_argument("--save-index", action="store_true",
                        help="摄入完成后把索引写入 --index 路径")
    parser.add_argument("--list", action="store_true", help="列出已索引来源后退出")
    parser.add_argument("-q", "--question", help="一次性提问（非交互模式）")
    parser.add_argument("-m", "--model", help="模型名（默认 $CORECODER_MODEL）")
    parser.add_argument("--base-url", help="API base URL（默认 $OPENAI_BASE_URL）")
    parser.add_argument("--api-key", help="API key（默认 $OPENAI_API_KEY）")
    parser.add_argument("--fallback", help="降级模型，逗号分隔")
    parser.add_argument("--budget", type=float, help="本次会话美元硬预算上限")
    parser.add_argument("-k", "--top-k", type=int, help="召回片段数（默认 5）")
    parser.add_argument("--threshold", type=float,
                        help="相关度阈值，低于它直接拒答（默认 0=不过滤）")
    parser.add_argument("--chunk-size", type=int, help="切片长度（默认 1000 字符）")
    parser.add_argument("--chunk-overlap", type=int, help="切片重叠（默认 100）")
    parser.add_argument("--embedding", choices=("hash", "openai"),
                        help="向量来源：hash=离线（默认），openai=走 embedding API")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="关闭向量检索，仅用 BM25")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--no-cite", action="store_true", help="不打印来源")
    parser.add_argument("-v", "--verbose", action="store_true", help="逐文件打印摄入进度")


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def build_llm(args, config: Config | None = None):
    """Construct an LLM exactly like corecoder.cli.main does, plus embeddings."""
    config = config or Config.from_env()
    for attr, dest in (
        ("model", "model"),
        ("base_url", "base_url"),
        ("api_key", "api_key"),
    ):
        val = getattr(args, attr, None)
        if val:
            setattr(config, dest, val)
    if getattr(args, "fallback", None):
        config.fallback_models = [m.strip() for m in args.fallback.split(",") if m.strip()]
    if getattr(args, "budget", None) is not None:
        config.budget_usd = args.budget
    if getattr(args, "top_k", None):
        config.kb_top_k = args.top_k
    if getattr(args, "threshold", None) is not None:
        config.kb_threshold = args.threshold
    if getattr(args, "chunk_size", None):
        config.kb_chunk_size = args.chunk_size
    if getattr(args, "chunk_overlap", None) is not None:
        config.kb_chunk_overlap = args.chunk_overlap
    if getattr(args, "embedding", None):
        config.kb_embedding = args.embedding

    kind = config.kb_embedding
    want_api = kind in ("openai", "litellm", "api")
    key = config.api_key or (config.embedding_api_key if want_api else None)
    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=key or "",
        base_url=config.base_url,
        fallback_models=config.fallback_models,
        budget_usd=config.budget_usd,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    llm._on_switch = lambda old, new, err: console.print(
        f"[yellow]模型降级: {old} → {new} ({type(err).__name__})[/yellow]"
    )
    return llm, config


def build_kb(args, llm, config: Config) -> KnowledgeBase:
    """Load the persisted index if there is one, then ingest --file/--knowledge."""
    use_emb = not getattr(args, "no_embeddings", False)
    kind = "hash" if not use_emb else config.kb_embedding
    kb = KnowledgeBase(
        llm=llm,
        embedding_kind=kind,
        top_k=config.kb_top_k,
        min_score=config.kb_threshold,
        chunk_size=config.kb_chunk_size,
        chunk_overlap=config.kb_chunk_overlap,
    )

    index = Path(getattr(args, "index", None) or config.kb_index)
    if index.is_file():
        try:
            kb = KnowledgeBase.load(
                index,
                llm=llm,
                embedding_kind=kind,
                top_k=config.kb_top_k,
                min_score=config.kb_threshold,
            )
            console.print(
                f"[dim]已加载索引 {index} -> {kb.size} chunks / "
                f"{len(kb.sources)} 来源[/dim]"
            )
        except Exception as e:  # noqa: BLE001 - a corrupt bundle must not kill the bot
            console.print(f"[yellow]索引 {index} 读取失败（{e}），改为重新构建[/yellow]")

    changed = False
    for f in getattr(args, "file", None) or []:
        try:
            n = len(kb.add_file(f))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the load
            console.print(f"[red]{f}: {e}[/red]")
            continue
        changed = changed or n > 0
        console.print(
            f"[dim]已加载 {f} -> {n} chunks[/dim]" if n
            else f"[yellow]{f}: 没有可用内容[/yellow]"
        )

    if getattr(args, "knowledge", None):
        n = kb.load_dir(args.knowledge, verbose=bool(getattr(args, "verbose", False)))
        changed = changed or n > 0
        console.print(f"[dim]已加载 {args.knowledge} -> 新增 {n} chunks[/dim]")
        for err in kb.last_errors:
            console.print(f"[dim yellow]跳过: {err}[/dim yellow]")

    if changed and getattr(args, "save_index", False):
        kb.save(index)
        console.print(f"[dim]索引已保存: {index}[/dim]")
    return kb


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def run_rag(args) -> int:
    """Entry called by corecoder.cli.main when args.command == "rag"."""
    config = Config.from_env()
    # an existing bundle is enough to start, whether it came from --index or
    # the default location ($CORECODER_KB_INDEX / ~/.corecoder/kb/index.json)
    index = Path(getattr(args, "index", None) or config.kb_index)
    has_index = index.is_file()
    if not has_index and not args.knowledge and not args.file:
        if config.kb_dir and Path(config.kb_dir).is_dir():
            args.knowledge = config.kb_dir
        else:
            console.print(
                "[red]请提供知识库：--knowledge DIR 或 --file FILE"
                "（或已有的 --index INDEX）[/red]"
            )
            return 2

    llm, config = build_llm(args)
    if not config.api_key and not config.embedding_api_key:
        console.print("[red]未找到 API Key。请设置 OPENAI_API_KEY 或 CORECODER_API_KEY。[/]")
        return 1

    kb = build_kb(args, llm, config)
    if kb.size == 0:
        console.print("[red]知识库为空（支持 txt/md/rst/csv/json/pdf/docx）。[/red]")
        return 1

    agent = RAGAgent(
        llm=llm,
        knowledge=kb,
        top_k=config.kb_top_k,
        min_score=config.kb_threshold,
        stream=not getattr(args, "no_stream", False),
    )
    kind = kb.stats()["embedder"].get("kind", "?")
    console.print(
        Panel(
            f"[bold]RAG 智能客服[/bold]\n"
            f"模型: [cyan]{llm.model}[/cyan]  向量: [cyan]{kind}[/cyan]"
            + f"\n知识库: [cyan]{kb.size}[/cyan] chunks / "
            f"[cyan]{len(kb.sources)}[/cyan] 来源, "
            f"召回 top-[cyan]{config.kb_top_k}[/cyan]"
            + (f", 阈值 [cyan]{config.kb_threshold:g}[/cyan]" if config.kb_threshold else "")
            + ("\n" + HELP_LINES if not args.question else ""),
            border_style="green",
        )
    )

    if getattr(args, "list", False):
        _print_sources(kb)
        return 0
    if args.question:
        return _ask_once(agent, args.question, show_cites=not getattr(args, "no_cite", False))
    return _repl(agent, kb, config, args)


def _ask_once(agent: RAGAgent, question: str, *, show_cites: bool = True) -> int:
    """One-shot Q&A, streaming to stdout."""
    streamed: list[str] = []

    def on_token(tok: str) -> None:
        streamed.append(tok)
        print(tok, end="", flush=True)

    try:
        turn = agent.answer(question, on_token=on_token)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]Error: {e}[/red]")
        return 1

    if not streamed:  # refusal, or a provider that didn't stream
        _render(turn.answer)
    if show_cites and turn.sources:
        console.print("[dim]来源: " + ", ".join(f"[{s}]" for s in turn.sources) + "[/dim]")
    print()
    return 0


def _repl(agent: RAGAgent, kb: KnowledgeBase, config: Config, args) -> int:
    """Interactive Q&A loop with a few knowledge-management commands."""
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.history import FileHistory

    hist = Path.home() / ".corecoder_rag_history"
    index_path = Path(getattr(args, "index", None) or config.kb_index)
    console.print(f"[dim]{HELP_LINES}[/dim]")
    while True:
        try:
            line = pt_prompt(
                "Q > ", history=FileHistory(str(hist)), multiline=True,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit", "/quit", "/exit"):
            return 0

        if line.startswith("/"):
            _command(agent, kb, config, index_path, line)
            continue

        try:
            turn = agent.answer(line, on_token=lambda t: print(t, end="", flush=True))
            print()
            if turn.sources:
                console.print(
                    "[dim]来源: " + ", ".join(f"[{s}]" for s in turn.sources) + "[/dim]"
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[red]Error: {e}[/red]")


def _command(agent: RAGAgent, kb: KnowledgeBase, config: Config, index_path: Path, line: str) -> None:
    """Slash commands inside the RAG REPL."""
    cmd, _, rest = line.partition(" ")
    cmd, rest = cmd.lower(), rest.strip()

    if cmd == "/sources":
        _print_sources(kb)
    elif cmd == "/add":
        if not rest:
            console.print("[yellow]用法: /add FILE[/yellow]")
            return
        try:
            n = len(kb.add_file(rest))
            console.print(f"[green]{rest}: {n} chunks[/green]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{e}[/red]")
    elif cmd == "/rm":
        removed = kb.remove_source(rest) if rest else 0
        console.print(
            f"[green]已删除 {rest}（{removed} chunks）[/green]" if removed
            else f"[yellow]未找到来源: {rest or '(空)'}[/yellow]"
        )
    elif cmd == "/save":
        kb.save(index_path)
        console.print(f"[green]索引已保存: {index_path}[/green]")
    elif cmd == "/reset":
        agent.reset()
        console.print("[yellow]对话已清空。[/yellow]")
    elif cmd == "/tokens":
        p, c = agent.llm.total_prompt_tokens, agent.llm.total_completion_tokens
        cost = agent.llm.estimated_cost
        line_out = f"Tokens: [cyan]{p}[/cyan] + [cyan]{c}[/cyan] = [bold]{p + c}[/bold]"
        console.print(line_out + (f"  (~${cost:.4f})" if cost is not None else ""))
        console.print(agent.llm.budget_status())
    elif cmd == "/stats":
        st = agent.stats()
        console.print(
            f"模型 [cyan]{st['model']}[/cyan] | chunks [cyan]{st['chunks']}[/cyan] "
            f"| 来源 [cyan]{st['sources']}[/cyan] | 向量 [cyan]{st['vectors']}[/cyan] "
            f"| top_k [cyan]{st['top_k']}[/cyan] | 阈值 [cyan]{st['min_score']:g}[/cyan]"
        )
    elif cmd == "/help":
        console.print(Panel(HELP_LINES + "\n" + _RAG_NOTE, border_style="dim"))
    else:
        console.print(f"[yellow]未知命令: {cmd}（/help 查看用法）[/yellow]")


_RAG_NOTE = (
    "回答只依据知识库内容；检索不到时输出：\n" + NO_ANSWER
)


def _print_sources(kb: KnowledgeBase) -> None:
    if not kb.sources:
        console.print("[dim]索引为空。[/dim]")
        return
    for src in kb.source_names:
        info = kb.sources.get(src) or {}
        console.print(f"  [cyan]{src}[/cyan]  [dim]{info.get('chunks', '?')} chunks[/dim]")


def _render(text: str) -> None:
    if not text:
        return
    try:
        console.print(Markdown(text))
    except Exception:  # noqa: BLE001 - rich chokes on exotic markup; fall back to raw
        console.print(text)


__all__ = ["build_kb", "build_llm", "run_rag"]
