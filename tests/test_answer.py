from __future__ import annotations

from tracerag.answer import build_ledger, synthesize
from tracerag.llm import LLMClient, LLMConfig
from tracerag.models import Citation, Node, Usage


def _offline() -> LLMClient:
    return LLMClient(LLMConfig(api_key=""))


def test_citation_markers_stay_with_the_sentence_they_support() -> None:
    citations = [
        Citation(1, "n1", "A", 1, 1, 0.0, ""),
        Citation(2, "n2", "B", 2, 2, 0.0, ""),
    ]
    nodes = [
        Node(id="n1", title="A", level=1, page_start=1, page_end=1, text="alpha beta gamma"),
        Node(id="n2", title="B", level=1, page_start=2, page_end=2, text="delta epsilon zeta"),
    ]
    text = "Alpha beta gamma. [1] Delta epsilon zeta. [2]"
    ledger = build_ledger(text, citations, nodes)
    assert [entry.citation_refs for entry in ledger] == [[1], [2]]
    assert all(entry.supported for entry in ledger)


def test_ledger_flags_sentences_without_a_citation() -> None:
    nodes = [Node(id="n1", title="A", level=1, page_start=1, page_end=1, text="alpha beta")]
    citations = [Citation(1, "n1", "A", 1, 1, 0.0, "")]
    ledger = build_ledger("Alpha beta is stated here. Something else entirely. [1]", citations, nodes)
    assert ledger[0].supported is False
    assert ledger[0].note == "no citation marker"


def test_ledger_flags_a_claim_the_cited_section_does_not_support() -> None:
    nodes = [Node(id="n1", title="A", level=1, page_start=1, page_end=1, text="alpha beta gamma")]
    citations = [Citation(1, "n1", "A", 1, 1, 0.0, "")]
    ledger = build_ledger("Quantum tunnelling reverses entropy. [1]", citations, nodes)
    assert ledger[0].supported is False
    assert ledger[0].support < 0.45


def test_synthesize_without_a_key_returns_verbatim_evidence() -> None:
    nodes = [
        Node(
            id="n1",
            title="Refund timing",
            level=1,
            page_start=1,
            page_end=1,
            text="Refunds take three to five business days.",
        )
    ]
    usage = Usage()
    text, citations = synthesize("How long do refunds take?", nodes, llm=_offline(), usage=usage, mode="hybrid")
    assert "three to five business days" in text
    assert "[1]" in text
    assert citations[0].node_id == "n1"
    assert usage.calls == 0


def test_synthesize_with_no_evidence_says_so() -> None:
    text, citations = synthesize("anything", [], llm=_offline(), usage=Usage(), mode="lexical")
    assert citations == []
    assert "could not find" in text.lower()
