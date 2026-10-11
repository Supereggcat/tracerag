"""A deliberately naive comparison point.

The evaluation harness scores the tree walker against this baseline so that
the claim "structure-aware reasoning beats flat similarity" is measured on
the repository's own corpus rather than asserted.

The baseline is what a lot of production RAG stacks reduce to: concatenate
the document, cut it into fixed windows with a little overlap, and rank the
windows with BM25. It is not handicapped on purpose. The heading lines stay
in the flattened text, exactly as they would for a chunker reading the raw
file, so any advantage the tree walker shows comes from structure and
reasoning rather than from hiding information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tracerag.lexical import BM25
from tracerag.models import Node
from tracerag.text import estimate_tokens

DEFAULT_WINDOW = 1000
DEFAULT_OVERLAP = 150
MIN_SECTION_SHARE = 0.15


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    sections: list[str] = field(default_factory=list)
    tokens: int = 0


@dataclass
class Span:
    start: int
    end: int
    title: str
    page_start: int
    page_end: int


def flatten_document(root: Node) -> tuple[str, list[Span]]:
    """Concatenate the document in reading order, headings included."""

    parts: list[str] = []
    spans: list[Span] = []
    cursor = 0
    for node in root.walk():
        text = node.text.strip()
        if not text:
            continue
        heading = node.title.strip()
        block = f"{heading}\n{text}" if heading and node.level > 0 else text
        if cursor:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block)
        cursor += len(block)
        spans.append(
            Span(
                start=start,
                end=cursor,
                title=node.title,
                page_start=node.page_start,
                page_end=node.page_end,
            )
        )
    return "".join(parts), spans


def naive_chunks(
    root: Node, *, window: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Cut the flattened document into fixed-width windows."""

    flat, spans = flatten_document(root)
    if not flat.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(flat):
        end = min(start + window, len(flat))
        if end < len(flat):
            cut = flat.rfind("\n", start + window // 2, end)
            if cut != -1:
                end = cut
        text = flat[start:end].strip()
        if text:
            sections, page_start, page_end = _attribute(spans, start, end, window)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    sections=sections,
                    tokens=estimate_tokens(text),
                )
            )
        if end >= len(flat):
            break
        start = max(end - overlap, start + 1)
    return chunks


def naive_retrieve(question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    if not chunks:
        return []
    scorer = BM25([chunk.text for chunk in chunks])
    scores = scorer.rank(question)
    ranked = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
    positive = [index for index in ranked[:top_k] if scores[index] > 0]
    if positive:
        return [chunks[index] for index in positive]
    return [chunks[ranked[0]]]


def _attribute(
    spans: list[Span], start: int, end: int, window: int
) -> tuple[list[str], int, int]:
    """Work out which sections a window was drawn from."""

    overlaps: dict[str, int] = {}
    pages: list[tuple[int, int]] = []
    for span in spans:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap <= 0:
            continue
        overlaps[span.title] = overlaps.get(span.title, 0) + overlap
        pages.append((span.page_start, span.page_end))
    if not overlaps:
        return [], 1, 1

    width = max(end - start, 1)
    threshold = MIN_SECTION_SHARE * min(width, window)
    ranked = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    sections = [title for title, count in ranked if count >= threshold] or [ranked[0][0]]
    page_start = min(item[0] for item in pages)
    page_end = max(item[1] for item in pages)
    return sections, page_start, page_end
