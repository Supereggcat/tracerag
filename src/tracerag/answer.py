"""Turn retrieved sections into an answer with a ledger that audits it.

Two things happen here. First a synthesiser writes the answer, either by
asking a model for prose with inline ``[n]`` markers or, with no key
configured, by extracting the best sentences from the evidence. Then the
ledger re-reads that answer sentence by sentence and checks each claim
against the sections it cites, flagging anything it cannot substantiate.
"""

from __future__ import annotations

import re
from collections import Counter

from tracerag.lexical import BM25
from tracerag.llm import LLMClient, LLMError
from tracerag.models import Citation, LedgerEntry, Node, Usage
from tracerag.text import (
    content_tokens,
    overlap_ratio,
    split_sentences,
    truncate,
)

ANSWER_SYSTEM = (
    "You answer questions using only the supplied context blocks. Every "
    "sentence must end with one or more citation markers of the form [1] that "
    "point at the blocks that support it. Never cite a block you did not use. "
    "If the context does not contain the answer, reply exactly: I could not "
    "find that in this document."
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
CITATION_PREFIX = re.compile(r"^\s*((?:\[\d+\]\s*)+)")
SUPPORT_THRESHOLD = 0.45
EXTRACTIVE_SENTENCES = 4


def synthesize(
    question: str,
    nodes: list[Node],
    *,
    llm: LLMClient,
    usage: Usage,
    mode: str,
) -> tuple[str, list[Citation]]:
    """Produce the answer text and the citations it relies on."""

    if not nodes:
        return "I could not find that in this document.", []

    citations = [
        Citation(
            ref=position,
            node_id=node.id,
            title=node.title,
            page_start=node.page_start,
            page_end=node.page_end,
            score=0.0,
            snippet=truncate(node.text or node.summary, 300),
        )
        for position, node in enumerate(nodes, start=1)
    ]

    if llm.enabled and mode != "lexical":
        text = _generate(question, nodes, llm=llm, usage=usage)
        if text:
            return _renumber(text, citations), citations
    return _extract(question, nodes), citations


def build_ledger(
    answer_text: str,
    citations: list[Citation],
    nodes: list[Node],
    *,
    threshold: float = SUPPORT_THRESHOLD,
) -> list[LedgerEntry]:
    """Check every sentence of the answer against the sections it cites."""

    by_ref = {citation.ref: citation for citation in citations}
    by_node = {node.id: node for node in nodes}
    ledger: list[LedgerEntry] = []

    for sentence in _sentences_with_citations(answer_text):
        refs = sorted({int(match) for match in CITATION_PATTERN.findall(sentence)})
        refs = [ref for ref in refs if ref in by_ref]
        body = CITATION_PATTERN.sub("", sentence).strip()

        if not body:
            continue
        if not refs:
            ledger.append(
                LedgerEntry(
                    sentence=sentence,
                    citation_refs=[],
                    support=0.0,
                    supported=False,
                    note="no citation marker",
                )
            )
            continue

        sentence_tokens = content_tokens(body)
        support = 0.0
        for ref in refs:
            node = by_node.get(by_ref[ref].node_id)
            if node is None:
                continue
            support = max(
                support,
                overlap_ratio(sentence_tokens, content_tokens(node.text or node.summary)),
            )
        ledger.append(
            LedgerEntry(
                sentence=sentence,
                citation_refs=refs,
                support=support,
                supported=support >= threshold,
                note="" if support >= threshold else "weak lexical overlap with cited section",
            )
        )
    return ledger


def _sentences_with_citations(text: str) -> list[str]:
    """Split an answer without letting a marker drift onto the next sentence.

    Answers are written as ``claim. [1] next claim. [2]``. A naive split on
    the full stop hands the ``[1]`` to the following sentence, which would
    make the ledger audit the wrong claim, so any marker that opens a
    sentence is re-attached to the sentence before it.
    """

    sentences: list[str] = []
    for part in split_sentences(text):
        match = CITATION_PREFIX.match(part)
        if match and sentences:
            sentences[-1] = f"{sentences[-1].rstrip()} {match.group(1).strip()}"
            part = part[match.end() :].strip()
            if not part:
                continue
        sentences.append(part)
    return sentences


def _generate(
    question: str, nodes: list[Node], *, llm: LLMClient, usage: Usage
) -> str:
    context = "\n\n".join(
        f"[{position}] {node.title} (pages {node.page_start}-{node.page_end})\n"
        f"{truncate(node.text or node.summary, 2600)}"
        for position, node in enumerate(nodes, start=1)
    )
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nContext blocks:\n{context}"},
    ]
    try:
        text, tokens = llm.chat(messages, max_tokens=700)
    except LLMError:
        return ""
    usage.add(**tokens)
    return text.strip()


def _renumber(text: str, citations: list[Citation]) -> str:
    """Drop citation markers that point outside the supplied blocks."""

    highest = len(citations)

    def replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return match.group(0) if 1 <= value <= highest else ""

    return re.sub(r"\s+([.,;:])", r"\1", CITATION_PATTERN.sub(replace, text)).strip()


def _extract(question: str, nodes: list[Node]) -> str:
    """Offline fallback: quote the sentences that best match the question.

    The node title takes part in the match. A heading such as "6.1 Retention"
    is often the only place the user's vocabulary appears, and dropping it
    here would make the offline path noticeably worse than the online one.
    """

    candidates: list[tuple[str, int, str]] = []
    for rank, node in enumerate(nodes, start=1):
        for sentence in split_sentences(node.text or node.summary):
            cleaned = sentence.strip()
            if len(cleaned) >= 12:
                candidates.append((cleaned, rank, f"{node.title} {cleaned}"))
    if not candidates:
        return "I could not find that in this document."

    scorer = BM25([haystack for _, _, haystack in candidates])
    scored: list[tuple[float, int, str, int]] = []
    best_raw = 0.0
    for index, (sentence, rank, _) in enumerate(candidates):
        raw = scorer.score(question, index)
        best_raw = max(best_raw, raw)
        scored.append((raw * (1.0 + 0.5 / rank), index, sentence, rank))
    if best_raw <= 0.0:
        return "I could not find that in this document."
    scored.sort(key=lambda item: item[0], reverse=True)

    chosen: list[tuple[int, str, int]] = []
    per_node: Counter[int] = Counter()
    for score, index, sentence, rank in scored:
        if score <= 0 and chosen:
            break
        if per_node[rank] >= 2:
            continue
        per_node[rank] += 1
        chosen.append((index, sentence, rank))
        if len(chosen) >= EXTRACTIVE_SENTENCES:
            break
    chosen.sort(key=lambda item: item[0])
    return " ".join(f"{sentence} [{rank}]" for _, sentence, rank in chosen)
