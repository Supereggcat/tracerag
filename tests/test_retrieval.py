from __future__ import annotations

import pytest

from tracerag.llm import LLMClient, LLMConfig
from tracerag.retrieve import MODES, RetrievalConfig, TreeWalker
from tracerag.store import Corpus


def test_mode_is_validated() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(mode="telepathy")
    with pytest.raises(ValueError):
        RetrievalConfig(beam=0)


@pytest.mark.parametrize("mode", MODES)
def test_walker_finds_the_section_that_holds_the_answer(
    indexed: Corpus, doc_ids: dict[str, str], mode: str
) -> None:
    tree = indexed.load_tree(doc_ids["atlas-handbook.md"])
    walker = TreeWalker(
        RetrievalConfig(mode=mode, max_selected=5),
        LLMClient(LLMConfig(api_key="", base_url="http://127.0.0.1:1")),
    )
    result = walker.run("退款到账要多久?", tree)
    titles = [node.title for node in result.selected]
    assert any("财务侧审批" in title for title in titles)
    assert result.trace, "every query must leave an audit trail"
    assert result.usage.calls == 0, "no key configured means no model calls"


def test_pruned_candidates_are_recorded_in_the_trace(
    indexed: Corpus, doc_ids: dict[str, str]
) -> None:
    tree = indexed.load_tree(doc_ids["atlas-handbook.md"])
    result = TreeWalker(RetrievalConfig(mode="lexical")).run("故障响应流程", tree)
    decisions = {step.decision for step in result.trace}
    assert "prune" in decisions
    assert decisions <= {"prune", "descend", "select"}
    assert all(step.rationale for step in result.trace)


def test_router_degrades_gracefully_when_the_model_is_unreachable(
    indexed: Corpus, doc_ids: dict[str, str]
) -> None:
    tree = indexed.load_tree(doc_ids["orbit-api-notes.md"])
    broken = LLMClient(LLMConfig(api_key="test-key", base_url="http://127.0.0.1:1/"))
    walker = TreeWalker(RetrievalConfig(mode="hybrid"), broken)
    result = walker.run("What happens if a delivery attempt fails?", tree)
    assert result.selected
    assert result.router_failures >= 1


def test_context_budget_is_respected(indexed: Corpus, doc_ids: dict[str, str]) -> None:
    tree = indexed.load_tree(doc_ids["helios-annual-report.md"])
    config = RetrievalConfig(mode="lexical", context_tokens=120, max_selected=8)
    result = TreeWalker(config).run("收入结构", tree)
    assert sum(node.token_estimate for node in result.selected) <= 120 + result.selected[0].token_estimate
