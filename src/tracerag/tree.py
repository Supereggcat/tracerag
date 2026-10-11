"""Build the hierarchical index that the retriever walks.

The tree has three jobs: keep every section attached to its ancestors so a
model can see context while navigating, keep the leaf granularity small
enough to be useful as evidence, and stay deterministic so an index built
twice from the same file is byte-identical.
"""

from __future__ import annotations

from tracerag.models import Block, Node
from tracerag.text import estimate_tokens, truncate

DEFAULT_SUMMARY_CHARS = 260
DEFAULT_MAX_LEAF_CHARS = 4000
DEFAULT_OVERLAP_CHARS = 320


def build_tree(
    blocks: list[Block],
    *,
    title: str | None = None,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
    max_leaf_chars: int = DEFAULT_MAX_LEAF_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> Node:
    """Turn ingested blocks into a section tree."""

    root = Node(
        id="n0",
        title=title or _guess_title(blocks),
        level=0,
        page_start=blocks[0].page if blocks else 1,
        page_end=blocks[-1].page if blocks else 1,
    )
    stack: list[Node] = []

    for block in blocks:
        path = block.heading_path
        shared = 0
        while shared < len(path) and shared < len(stack) and stack[shared].title == path[shared]:
            shared += 1
        del stack[shared:]

        for depth in range(shared, len(path)):
            parent = stack[-1] if stack else root
            node = Node(
                id="",
                title=path[depth],
                level=depth + 1,
                page_start=block.page,
                page_end=block.page,
                parent_id=parent.id,
            )
            parent.children.append(node)
            stack.append(node)

        target = stack[-1] if stack else root
        target.text = f"{target.text}\n\n{block.text}" if target.text else block.text
        target.page_end = max(target.page_end, block.page)
        for ancestor in stack:
            ancestor.page_end = max(ancestor.page_end, block.page)

    _assign_identifiers(root)
    _split_long_leaves(root, max_leaf_chars=max_leaf_chars, overlap_chars=overlap_chars)
    _finalise(root, summary_chars=summary_chars)
    return root


def flatten(root: Node, *, max_level: int | None = None) -> list[Node]:
    """Depth-first listing, optionally limited to the top levels."""

    nodes = []
    for node in root.walk():
        if node.id == root.id:
            continue
        if max_level is not None and node.level > max_level:
            continue
        nodes.append(node)
    return nodes


def _guess_title(blocks: list[Block]) -> str:
    for block in blocks:
        if block.heading_path:
            return block.heading_path[0]
    return "Document"


def _assign_identifiers(root: Node) -> None:
    root.id = "n0"
    root.parent_id = None

    def visit(node: Node, path: str) -> None:
        for position, child in enumerate(node.children, start=1):
            child.id = f"{path}.{position}" if path else f"n{position}"
            child.parent_id = node.id
            child.level = node.level + 1
            visit(child, child.id)

    visit(root, "")


def _split_long_leaves(
    root: Node, *, max_leaf_chars: int, overlap_chars: int
) -> None:
    """Give oversized leaves children so retrieval stays fine grained."""

    for node in list(root.walk()):
        if node.children or len(node.text) <= max_leaf_chars:
            continue
        windows = _window(node.text, max_leaf_chars, overlap_chars)
        if len(windows) < 2:
            continue
        for position, window in enumerate(windows, start=1):
            node.children.append(
                Node(
                    id=f"{node.id}.{position}",
                    title=f"{truncate(node.title, 70)} (part {position}/{len(windows)})",
                    level=node.level + 1,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    text=window,
                    parent_id=node.id,
                )
            )


def _window(text: str, size: int, overlap: int) -> list[str]:
    window_count = 1
    start = 0
    windows: list[str] = []
    while start < len(text):
        end = start + size
        if end >= len(text):
            windows.append(text[start:])
            break
        cut = text.rfind("\n", start + size // 2, end)
        if cut == -1:
            cut = end
        windows.append(text[start:cut])
        window_count += 1
        start = max(cut - overlap, start + 1)
    return [item.strip() for item in windows if item.strip()]


def _finalise(root: Node, *, summary_chars: int) -> None:
    for node in reversed(list(root.walk())):
        node.page_start = min(
            [node.page_start, *(child.page_start for child in node.children)]
        )
        node.page_end = max(
            [node.page_end, *(child.page_end for child in node.children)]
        )
        node.token_estimate = estimate_tokens(node.text) + estimate_tokens(node.summary)

        if node.text.strip():
            node.summary = truncate(node.text, summary_chars)
        elif node.children:
            titles = "; ".join(child.title for child in node.children)
            node.summary = truncate(f"{node.title}: {titles}", summary_chars)
        else:
            node.summary = node.title

        node.token_estimate = estimate_tokens(node.text) + estimate_tokens(node.summary)
