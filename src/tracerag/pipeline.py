"""Glue: index a file, then answer questions against it."""

from __future__ import annotations

import time
from pathlib import Path

from tracerag.answer import build_ledger, synthesize
from tracerag.ingest import load_blocks
from tracerag.llm import LLMClient
from tracerag.models import Answer
from tracerag.retrieve import RetrievalConfig, TreeWalker
from tracerag.store import Corpus, DocMeta, build_meta
from tracerag.tree import build_tree

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".text", ".log", ".rst", ".html", ".htm"}


def index_file(
    source: str | Path,
    *,
    data_dir: str | Path = "data/index",
    title: str | None = None,
    corpus: Corpus | None = None,
) -> DocMeta:
    """Ingest one file, build its tree and persist it in the corpus."""

    path = Path(source)
    store = corpus or Corpus(data_dir)
    payload = path.read_bytes()
    blocks = load_blocks(path)
    if not blocks:
        raise ValueError(f"no extractable text in {path.name}")
    tree = build_tree(blocks, title=title)
    doc_id = store.make_doc_id(path.name, payload)
    meta = build_meta(
        doc_id=doc_id,
        title=tree.title,
        source_name=path.name,
        suffix=path.suffix.lower(),
        payload=payload,
        tree=tree,
        block_count=len(blocks),
        pages=max((block.page for block in blocks), default=1),
    )
    store.save(
        doc_id=doc_id,
        meta=meta,
        tree=tree,
        source_bytes=payload,
        suffix=path.suffix.lower(),
    )
    return meta


def index_path(
    source: str | Path,
    *,
    data_dir: str | Path = "data/index",
    corpus: Corpus | None = None,
) -> list[DocMeta]:
    """Index a single file, or every supported file in a directory."""

    path = Path(source)
    if path.is_file():
        return [index_file(path, data_dir=data_dir, corpus=corpus)]
    if not path.is_dir():
        raise FileNotFoundError(f"no such file or directory: {path}")
    store = corpus or Corpus(data_dir)
    metas = [
        index_file(child, data_dir=data_dir, corpus=store)
        for child in sorted(path.rglob("*"))
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not metas:
        raise ValueError(f"no supported documents found under {path}")
    return metas


def ask(
    question: str,
    doc_id: str,
    *,
    data_dir: str | Path = "data/index",
    config: RetrievalConfig | None = None,
    llm: LLMClient | None = None,
    corpus: Corpus | None = None,
) -> Answer:
    """Answer ``question`` from one indexed document."""

    store = corpus or Corpus(data_dir)
    settings = config or RetrievalConfig()
    if not store.exists(doc_id):
        raise KeyError(f"unknown document: {doc_id}")

    tree = store.load_tree(doc_id)
    walker = TreeWalker(settings, llm)
    started = time.perf_counter()
    retrieval = walker.run(question, tree)
    text, citations = synthesize(
        question,
        retrieval.selected,
        llm=walker.llm,
        usage=retrieval.usage,
        mode=settings.mode,
    )
    ledger = build_ledger(text, citations, retrieval.selected)
    retrieval.usage.latency_ms = int((time.perf_counter() - started) * 1000)
    retrieval.usage.cost_usd = _estimate(retrieval.usage)

    for citation in citations:
        citation.score = retrieval.scores.get(citation.node_id, 0.0)

    return Answer(
        question=question,
        doc_id=doc_id,
        text=text,
        mode=settings.mode,
        citations=citations,
        ledger=ledger,
        trace=retrieval.trace,
        selected=[node.id for node in retrieval.selected],
        usage=retrieval.usage,
    )


def _estimate(usage) -> float:
    from tracerag.llm import estimate_cost

    return estimate_cost(usage.model, usage.prompt_tokens, usage.completion_tokens)
