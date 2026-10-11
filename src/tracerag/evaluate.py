"""Measure the tree walker against naive fixed-window chunking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tracerag.baseline import naive_chunks, naive_retrieve
from tracerag.llm import LLMClient
from tracerag.pipeline import ask
from tracerag.retrieve import RetrievalConfig
from tracerag.store import Corpus


@dataclass
class EvalCase:
    question: str
    doc: str
    expect_any: list[str] = field(default_factory=list)
    expect_all: list[str] = field(default_factory=list)
    difficulty: str = "single"


@dataclass
class EvalOutcome:
    question: str
    doc: str
    difficulty: str
    tree_hit: bool
    baseline_hit: bool
    tree_sections: list[str]
    baseline_sections: list[str]
    supported_ratio: float
    tree_tokens: int
    baseline_tokens: int
    nodes_opened: int
    latency_ms: int


@dataclass
class EvalReport:
    mode: str
    top_k: int
    cases: list[EvalOutcome] = field(default_factory=list)

    @property
    def tree_recall(self) -> float:
        return _mean([case.tree_hit for case in self.cases])

    @property
    def baseline_recall(self) -> float:
        return _mean([case.baseline_hit for case in self.cases])

    @property
    def support(self) -> float:
        return _mean([case.supported_ratio for case in self.cases])

    @property
    def multihop_cases(self) -> list[EvalOutcome]:
        return [case for case in self.cases if case.difficulty == "multi"]

    @property
    def tree_recall_multihop(self) -> float:
        return _mean([case.tree_hit for case in self.multihop_cases])

    @property
    def baseline_recall_multihop(self) -> float:
        return _mean([case.baseline_hit for case in self.multihop_cases])

    @property
    def tree_tokens(self) -> float:
        return _mean([case.tree_tokens for case in self.cases])

    @property
    def baseline_tokens(self) -> float:
        return _mean([case.baseline_tokens for case in self.cases])

    @property
    def mean_nodes(self) -> float:
        return _mean([case.nodes_opened for case in self.cases])

    @property
    def mean_latency_ms(self) -> float:
        return _mean([case.latency_ms for case in self.cases])

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "summary": {
                "tree_recall_at_k": round(self.tree_recall, 4),
                "baseline_recall_at_k": round(self.baseline_recall, 4),
                "tree_recall_at_k_multihop": round(self.tree_recall_multihop, 4),
                "baseline_recall_at_k_multihop": round(self.baseline_recall_multihop, 4),
                "citation_support": round(self.support, 4),
                "mean_evidence_tokens": round(self.tree_tokens, 1),
                "mean_baseline_tokens": round(self.baseline_tokens, 1),
                "mean_nodes_opened": round(self.mean_nodes, 2),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
                "cases": len(self.cases),
            },
            "cases": [asdict(case) for case in self.cases],
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                question=payload["question"],
                doc=payload["doc"],
                expect_any=list(payload.get("expect_any", [])),
                expect_all=list(payload.get("expect_all", [])),
                difficulty=payload.get("difficulty", "single"),
            )
        )
    return cases


def run_eval(
    cases: list[EvalCase],
    *,
    data_dir: str | Path = "data/index",
    mode: str = "lexical",
    top_k: int = 5,
    llm: LLMClient | None = None,
) -> EvalReport:
    corpus = Corpus(data_dir)
    by_source = {meta.source_name: meta for meta in corpus.list_docs()}
    report = EvalReport(mode=mode, top_k=top_k)

    for case in cases:
        meta = by_source.get(case.doc)
        if meta is None:
            continue
        tree = corpus.load_tree(meta.doc_id)
        config = RetrievalConfig(mode=mode, max_selected=top_k)
        answer = ask(
            case.question,
            meta.doc_id,
            data_dir=data_dir,
            config=config,
            llm=llm,
            corpus=corpus,
        )

        selected = [tree.find(node_id) for node_id in answer.selected]
        tree_sections = [node.title for node in selected if node is not None]
        tree_tokens = sum(
            (node.token_estimate or 0) for node in selected if node is not None
        )
        chunks = naive_chunks(tree)
        baseline = naive_retrieve(case.question, chunks, top_k=top_k)
        baseline_sections = sorted({title for chunk in baseline for title in chunk.sections})
        baseline_tokens = sum(chunk.tokens for chunk in baseline)

        report.cases.append(
            EvalOutcome(
                question=case.question,
                doc=case.doc,
                difficulty=case.difficulty,
                tree_hit=_matches(case, tree_sections),
                baseline_hit=_matches(case, baseline_sections),
                tree_sections=tree_sections,
                baseline_sections=baseline_sections,
                supported_ratio=answer.supported_ratio,
                tree_tokens=tree_tokens,
                baseline_tokens=baseline_tokens,
                nodes_opened=len(answer.trace),
                latency_ms=answer.usage.latency_ms,
            )
        )
    return report


def format_report(report: EvalReport) -> str:
    lines = [
        "| metric | tree walker | naive chunking |",
        "| --- | ---: | ---: |",
        f"| recall@{report.top_k} (all {len(report.cases)} cases) | {report.tree_recall:.1%} | {report.baseline_recall:.1%} |",
        f"| recall@{report.top_k} (multi-hop only) | {report.tree_recall_multihop:.1%} | {report.baseline_recall_multihop:.1%} |",
        f"| evidence tokens sent to the answerer | {report.tree_tokens:.0f} | {report.baseline_tokens:.0f} |",
        f"| citation support | {report.support:.1%} | n/a |",
        f"| mean nodes visited | {report.mean_nodes:.1f} | n/a |",
        f"| mean latency | {report.mean_latency_ms:.0f} ms | n/a |",
        "",
        f"_mode: {report.mode}, cases: {len(report.cases)}_",
    ]
    missed = [case for case in report.cases if not case.tree_hit]
    if missed:
        lines.append("")
        lines.append("Unresolved questions:")
        lines.extend(f"- {case.question}" for case in missed)
    return "\n".join(lines)


def _matches(case: EvalCase, sections: list[str]) -> bool:
    haystack = " \n ".join(sections)
    if case.expect_all:
        return all(item in haystack for item in case.expect_all)
    if case.expect_any:
        return any(item in haystack for item in case.expect_any)
    return False


def _mean(values: list) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
