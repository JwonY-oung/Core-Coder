"""Tool registry."""
from pathlib import Path

from .agent import AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .fetch import FetchUrlTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .rag_tool import RAGTool
from .read import ReadFileTool
from .todo import TodoWriteTool
from .write import WriteFileTool

# corecoder/tools/__init__.py
_DOCS_DIR = Path(__file__).resolve().parent.parent / "rag" / "docs"

ALL_TOOLS = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    TodoWriteTool(),
    AgentTool(),
    FetchUrlTool(),
    RAGTool(knowledge_dir=str(_DOCS_DIR)),
]

