"""RAG knowledge-base tool for agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from corecoder.rag.knowledge import KnowledgeBase

from .base import Tool


class RAGTool(Tool):
    """Query / manage a RAG knowledge base."""

    name = "rag"
    #read_only = True
    description = (
        "查询内部知识库，包含 FAQ、退款政策、配送说明等文档。"
        "当用户询问政策、退款、配送、常见问题等内部信息时必须调用。"
        "不要用于通用知识、编程问题或闲聊。"
        "如果检索结果 found=false,如实回答没有即可,不要瞎编."
        "action='search' 用于检索。"
       # "Search a RAG knowledge base or add documents to it. "
       # "Use 'search' to query indexed content, 'add_text' to add raw text, "
        "'add_file' to index a file, 'list_sources' to see what's loaded, "
        "'remove_source' to delete a source, and 'stats' for index info."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "add_text", "add_file", "list_sources", "remove_source", "stats"],
                "description": "Action to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search query (required when action=search).",
            },
            "text": {
                "type": "string",
                "description": "Text content to add (required when action=add_text).",
            },
            "source": {
                "type": "string",
                "description": "Source identifier/tag for the added text or file.",
            },
            "file_path": {
                "type": "string",
                "description": "File path to index (required when action=add_file).",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, knowledge_dir: str | None = None, **kb_kwargs):
        self._kb: KnowledgeBase | None = None
        self._knowledge_dir = knowledge_dir
        self._kb_kwargs = kb_kwargs

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = KnowledgeBase(**self._kb_kwargs)
            if self._knowledge_dir and Path(self._knowledge_dir).is_dir():
                self._kb.load_dir(self._knowledge_dir, verbose=False)
        return self._kb

    def execute(self, action: str, **kwargs) -> str:
        kb = self.kb

        if action == "search":
            query = kwargs.get("query", "")
            top_k = kwargs.get("top_k", 5)
            if not query:
                return '{"error": "query is required for search"}'
            result = kb.ask(query, k=top_k)
            if result.low_confidence:
                return json.dumps({"found": False, "message": "知识库中未找到相关信息"}, ensure_ascii=False)
            items = []
            for chunk, score in zip(result.chunks, result.scores):
                items.append({
                    "source": chunk.source,
                    "tag": chunk.tag,
                    "score": round(score, 6),
                    "snippet": chunk.text[:500],
                })
            return json.dumps({"found": True, "results": items}, ensure_ascii=False)

        elif action == "add_text":
            text = kwargs.get("text", "")
            source = kwargs.get("source", "memory")
            if not text:
                return '{"error": "text is required for add_text"}'
            n = kb.add_text(text, source=source)
            return json.dumps({"added": True, "chunks": n, "source": source}, ensure_ascii=False)

        elif action == "add_file":
            file_path = kwargs.get("file_path", "")
            source = kwargs.get("source", file_path)
            if not file_path:
                return '{"error": "file_path is required for add_file"}'
            try:
                n = kb.add_file(file_path, source=source)
                return json.dumps({"added": True, "chunks": n, "source": source}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        elif action == "list_sources":
            sources = kb.source_names
            return json.dumps({"sources": sources, "count": len(sources)}, ensure_ascii=False)

        elif action == "remove_source":
            source = kwargs.get("source", "")
            if not source:
                return '{"error": "source is required for remove_source"}'
            removed = kb.remove_source(source)
            return json.dumps({"removed": True, "chunks_removed": removed, "source": source}, ensure_ascii=False)

        elif action == "stats":
            return json.dumps(kb.stats(), ensure_ascii=False)

        else:
            return json.dumps({"error": f"unknown action: {action}"}, ensure_ascii=False)
