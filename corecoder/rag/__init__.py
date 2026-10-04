"""RAG (Retrieval-Augmented Generation) subpackage.

Zero-dependency retrieval core on top of the existing ``corecoder.llm.LLM``:

    from corecoder.rag import KnowledgeBase, RAGAgent, make_embedder

    kb = KnowledgeBase()
    kb.load_dir("docs/")
    agent = RAGAgent(llm=llm, knowledge=kb)
    print(agent.chat("如何配置 API Key?"))

Modules:
  * embeddings - hash-based embedding + optional numpy fast path
  * retriever  - chunking, BM25 index, hybrid RRF search
  * knowledge  - KnowledgeBase (load dir/file/text, JSON persistence)
  * agent      - RAGAgent reusing LLM.chat, no second agent loop
"""

from .agent import RAGAgent
from .embeddings import HashEmbedding, OpenAIEmbedding, cosine, make_embedder, signature
from .knowledge import Answer, KnowledgeBase
from .retriever import BM25Index, Chunk, Retriever, chunk_document, tokenize

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "HashEmbedding",
    "KnowledgeBase",
    "OpenAIEmbedding",
    "RAGAgent",
    "Retriever",
    "chunk_document",
    "cosine",
    "make_embedder",
    "signature",
    "tokenize",
]