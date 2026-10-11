"""Language-aware text helpers.

The tokeniser is deliberately dependency free and handles mixed Chinese and
English text, because the corpora this project is aimed at are usually a mix
of both. CJK runs are indexed as unigrams plus bigrams, which gives BM25 a
usable signal without a segmentation dictionary.
"""

from __future__ import annotations

import re

LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")
CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
WHITESPACE = re.compile(r"[ \t\u00a0\u3000]+")
TOKEN_ESTIMATE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

SENTENCE_SPLIT = re.compile(
    r"(?<=[\u3002\uff01\uff1f!?])\s*"
    r"|\n{2,}"
    r"|(?<=\.)\s+(?=[A-Z0-9\u201c\"'(\[])"
)

STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have how i if in into is it
    its of on or that the their then there these they this to was were what when
    where which who why will with would you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Split text into search tokens suitable for BM25."""

    if not text:
        return []
    lowered = text.lower()
    tokens = LATIN_TOKEN.findall(lowered)
    for run in CJK_RUN.findall(lowered):
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def content_tokens(text: str) -> list[str]:
    """Tokens with English stopwords removed, used for support checking."""

    return [token for token in tokenize(text) if token not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, merging fragments that are too short."""

    if not text or not text.strip():
        return []
    raw = [part.strip() for part in SENTENCE_SPLIT.split(text) if part and part.strip()]
    merged: list[str] = []
    for part in raw:
        if merged and len(part) <= 3:
            merged[-1] = f"{merged[-1]}{part}"
        else:
            merged.append(part)
    return merged


def estimate_tokens(text: str) -> int:
    """Cheap token estimate that behaves sensibly for CJK and Latin text."""

    if not text:
        return 0
    cjk = len(TOKEN_ESTIMATE_CJK.findall(text))
    other = max(len(text) - cjk, 0)
    return int(cjk * 1.05 + other / 3.6) + 1


def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def truncate(text: str, limit: int = 240) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "\u2026"


def overlap_ratio(query_tokens: list[str], document_tokens: list[str]) -> float:
    """Fraction of distinct query tokens that appear in the document."""

    unique = {token for token in query_tokens if len(token) > 1 or token.isalnum()}
    if not unique:
        return 0.0
    haystack = set(document_tokens)
    hits = sum(1 for token in unique if token in haystack)
    return hits / len(unique)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "document"
