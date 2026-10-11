"""Turn a file on disk into a flat list of headed blocks.

The ingestion layer is intentionally boring. It does not call a model and it
does not try to be clever about layout: it walks the source lines, decides
which lines are headings, and emits the text between headings together with
the outline path that was in force. Everything downstream trusts that path.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from tracerag.models import Block
from tracerag.text import normalize_whitespace

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
HTML_HEADING = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
HTML_BREAK = re.compile(r"</(?:p|div|li|tr|section|article)>", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s*[.)\u3001]?\s+(\S.*)$")
CHINESE_CHAPTER = re.compile(r"^第\s*[0-9\u4e00-\u9fff]+\s*[章篇部]")
CHINESE_SECTION = re.compile(r"^第\s*[0-9\u4e00-\u9fff]+\s*[节条]")
KEYWORD_HEADING = re.compile(
    r"^(?:abstract|introduction|background|related work|method(?:ology)?|"
    r"experiments?|results?|discussion|conclusions?|references|appendix|"
    r"summary|overview|table of contents)\b",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {".txt", ".text", ".log", ".rst"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}


def load_blocks(path: str | Path) -> list[Block]:
    """Read ``path`` and return its blocks in document order."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no such file: {source}")
    suffix = source.suffix.lower()

    if suffix in PDF_SUFFIXES:
        pages = _pdf_pages(source)
        return _blocks_from_pages(pages, markdown=False)
    if suffix in MARKDOWN_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="replace")
        return _blocks_from_pages([(1, text)], markdown=True)
    if suffix in HTML_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="replace")
        return _blocks_from_pages([(1, _html_to_text(text))], markdown=True)
    if suffix in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="replace")
        return _blocks_from_pages([(1, text)], markdown=False)

    supported = sorted(PDF_SUFFIXES | MARKDOWN_SUFFIXES | HTML_SUFFIXES | TEXT_SUFFIXES)
    raise ValueError(f"unsupported file type {suffix!r}; supported: {', '.join(supported)}")


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError("reading PDFs requires the 'pypdf' package") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - malformed PDFs
            text = ""
        pages.append((number, text))
    return pages


def _html_to_text(raw: str) -> str:
    """Flatten HTML while keeping heading structure visible to the parser."""

    marked = HTML_HEADING.sub(lambda m: f"\n{'#' * int(m.group(1))} {_strip_tags(m.group(2))}\n", raw)
    marked = HTML_BREAK.sub("\n", marked)
    marked = re.sub(r"<br\s*/?>", "\n", marked, flags=re.IGNORECASE)
    return html.unescape(_strip_tags(marked))


def _strip_tags(raw: str) -> str:
    return normalize_whitespace(HTML_TAG.sub(" ", raw))


def _blocks_from_pages(
    pages: list[tuple[int, str]], *, markdown: bool
) -> list[Block]:
    """Group page text into blocks keyed by the outline path in force."""

    stack: dict[int, str] = {}
    buffer: list[str] = []
    buffer_page = pages[0][0] if pages else 1
    blocks: list[Block] = []
    saw_heading = False
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return
        path = tuple(stack[key] for key in sorted(stack)) if saw_heading else ()
        blocks.append(
            Block(order=len(blocks), page=buffer_page, heading_path=path, text=text)
        )

    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if markdown and FENCE.match(line):
                in_fence = not in_fence
                buffer.append(line)
                continue

            heading = None if in_fence else _detect_heading(stripped, markdown=markdown)
            if heading is None:
                if not buffer:
                    buffer_page = page_number
                buffer.append(line)
                continue

            level, title = heading
            flush()
            saw_heading = True
            for deeper in [key for key in stack if key >= level]:
                del stack[deeper]
            stack[level] = title
            buffer_page = page_number

    flush()
    return blocks


def _detect_heading(line: str, *, markdown: bool) -> tuple[int, str] | None:
    """Return ``(level, title)`` when ``line`` looks like a heading."""

    if not line or len(line) > 120:
        return None

    if markdown:
        match = MARKDOWN_HEADING.match(line)
        if match:
            title = match.group(2).strip()
            if title:
                return min(len(match.group(1)), 4), title

    if CHINESE_CHAPTER.match(line):
        return 1, line
    if CHINESE_SECTION.match(line):
        return 2, line

    numbered = NUMBERED_HEADING.match(line)
    if numbered and not line.endswith((".", "\u3002", ",", "\uff0c", ";")):
        depth = numbered.group(1).count(".") + 1
        if len(numbered.group(2)) <= 90:
            return min(depth, 4), line

    if KEYWORD_HEADING.match(line) and len(line) <= 60:
        return 1, line

    # A short, unpunctuated, title-ish line is good evidence in prose sources
    # such as extracted PDF text, where heading markup is lost.
    if (
        not markdown
        and 3 <= len(line) <= 60
        and not line.endswith((".", "\u3002", ":", "\uff1a", ";", "\uff1b", ",", "\uff0c"))
        and not line.startswith(("-", "*", "\u2022", ">"))
        and line.count(" ") <= 9
        and (line[0].isupper() or CHINESE_CHAPTER.match(line))
    ):
        return 2, line

    return None
