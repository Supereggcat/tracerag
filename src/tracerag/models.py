"""Typed records shared across TraceRAG.

Every object that crosses a module boundary is defined here so the pipeline
stays inspectable: the tree builder returns ``Node`` objects, the retriever
returns ``TraceStep`` objects, and the answerer returns an ``Answer`` that
carries both its citations and the ledger that audits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Block:
    """A contiguous run of source text belonging to one outline path."""

    order: int
    page: int
    heading_path: tuple[str, ...]
    text: str


@dataclass
class Node:
    """One section of the document tree."""

    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    text: str = ""
    summary: str = ""
    token_estimate: int = 0
    parent_id: str | None = None
    children: list["Node"] = field(default_factory=list)

    def walk(self) -> Iterator["Node"]:
        """Yield this node and every descendant, depth first."""

        yield self
        for child in self.children:
            yield from child.walk()

    def leaves(self) -> Iterator["Node"]:
        if not self.children:
            yield self
            return
        for child in self.children:
            yield from child.leaves()

    def is_leaf(self) -> bool:
        return not self.children

    def find(self, node_id: str) -> "Node | None":
        for node in self.walk():
            if node.id == node_id:
                return node
        return None

    def size(self) -> int:
        return sum(1 for _ in self.walk())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "summary": self.summary,
            "token_estimate": self.token_estimate,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Node":
        node = cls(
            id=payload["id"],
            title=payload["title"],
            level=int(payload["level"]),
            page_start=int(payload.get("page_start", 1)),
            page_end=int(payload.get("page_end", payload.get("page_start", 1))),
            text=payload.get("text", ""),
            summary=payload.get("summary", ""),
            token_estimate=int(payload.get("token_estimate", 0)),
            parent_id=payload.get("parent_id"),
        )
        node.children = [cls.from_dict(item) for item in payload.get("children", [])]
        return node


@dataclass
class TraceStep:
    """One decision the retriever made while walking the tree."""

    index: int
    node_id: str
    title: str
    level: int
    score: float
    scorer: str
    decision: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "score": round(self.score, 4),
            "scorer": self.scorer,
            "decision": self.decision,
            "rationale": self.rationale,
        }


@dataclass
class Citation:
    """A numbered reference from the answer back into the document tree."""

    ref: int
    node_id: str
    title: str
    page_start: int
    page_end: int
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "node_id": self.node_id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


@dataclass
class LedgerEntry:
    """One sentence of the answer plus whether its citations back it up."""

    sentence: str
    citation_refs: list[int]
    support: float
    supported: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "citation_refs": list(self.citation_refs),
            "support": round(self.support, 4),
            "supported": self.supported,
            "note": self.note,
        }


@dataclass
class Usage:
    """Token and cost accounting for a single query."""

    model: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    def add(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": self.latency_ms,
        }


@dataclass
class Answer:
    """The full result of a query: prose, citations, audit trail, trace."""

    question: str
    doc_id: str
    text: str
    mode: str
    citations: list[Citation] = field(default_factory=list)
    ledger: list[LedgerEntry] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def supported_ratio(self) -> float:
        if not self.ledger:
            return 0.0
        return sum(1 for entry in self.ledger if entry.supported) / len(self.ledger)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "doc_id": self.doc_id,
            "text": self.text,
            "mode": self.mode,
            "supported_ratio": round(self.supported_ratio, 4),
            "citations": [item.to_dict() for item in self.citations],
            "ledger": [item.to_dict() for item in self.ledger],
            "trace": [item.to_dict() for item in self.trace],
            "selected": list(self.selected),
            "usage": self.usage.to_dict(),
        }
