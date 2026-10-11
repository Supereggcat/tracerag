"""TraceRAG: vectorless, reasoning-driven retrieval with an auditable trace."""

from tracerag.answer import build_ledger, synthesize
from tracerag.ingest import load_blocks
from tracerag.llm import LLMConfig, LLMError
from tracerag.models import Answer, Citation, LedgerEntry, Node, TraceStep
from tracerag.pipeline import ask, index_file, index_path
from tracerag.retrieve import RetrievalConfig, TreeWalker
from tracerag.store import Corpus
from tracerag.tree import build_tree

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Citation",
    "Corpus",
    "LLMConfig",
    "LLMError",
    "LedgerEntry",
    "Node",
    "RetrievalConfig",
    "TraceStep",
    "TreeWalker",
    "ask",
    "build_ledger",
    "build_tree",
    "index_file",
    "index_path",
    "load_blocks",
    "synthesize",
]
