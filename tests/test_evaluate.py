from __future__ import annotations

from pathlib import Path

from tracerag.evaluate import format_report, load_cases, run_eval
from tracerag.llm import LLMClient, LLMConfig
from tracerag.pipeline import index_path
from tracerag.store import Corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eval_runs_offline_and_beats_the_flat_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "index"
    index_path(REPO_ROOT / "data" / "corpus", data_dir=data_dir)
    cases = load_cases(REPO_ROOT / "eval" / "questions.jsonl")
    report = run_eval(
        cases,
        data_dir=data_dir,
        mode="lexical",
        top_k=5,
        llm=LLMClient(LLMConfig(api_key="")),
    )

    assert len(report.cases) == len(cases)
    assert report.tree_recall > report.baseline_recall
    assert report.tree_recall_multihop > report.baseline_recall_multihop
    assert report.tree_tokens < report.baseline_tokens
    assert report.support > 0.9

    rendered = format_report(report)
    assert "tree walker" in rendered
    assert "naive chunking" in rendered


def test_eval_case_parsing_ignores_comments(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '# a comment\n{"doc":"a.md","question":"q","expect_any":["A"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].expect_any == ["A"]
