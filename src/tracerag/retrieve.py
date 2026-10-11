"""Walk the document tree with reasoning instead of similarity.

Retrieval happens one level at a time. Every level is narrowed by BM25 first,
which is cheap and deterministic, and then optionally refined by a model that
sees nothing but the question and a short list of candidate sections. The
point of the two-stage design is that the model never gets asked to rank a
corpus it cannot fit in a prompt, and that every decision it does make is
written down as a ``TraceStep``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tracerag.lexical import BM25
from tracerag.llm import LLMClient, LLMError, parse_json_block
from tracerag.models import Node, TraceStep, Usage
from tracerag.text import estimate_tokens, truncate

ROUTER_SYSTEM = (
    "You route a retrieval query through a document tree. You are given the "
    "question and a numbered list of candidate sections, each with a short "
    "summary. Choose only the sections that could plausibly contain the answer. "
    "Prefer keeping a section you are unsure about over pruning one that might "
    'be relevant. Reply with JSON only: {"keep": [1, 3], "reason": "..."}'
)

MODE_LEXICAL = "lexical"
MODE_HYBRID = "hybrid"
MODE_REASONING = "reasoning"
MODES = (MODE_LEXICAL, MODE_HYBRID, MODE_REASONING)


@dataclass
class RetrievalConfig:
    """Tunables for one retrieval pass."""

    mode: str = MODE_HYBRID
    max_depth: int = 5
    beam: int = 4
    max_nodes: int = 40
    max_selected: int = 5
    context_tokens: int = 3600
    ancestor_weight: float = 0.35
    max_trace: int = 80

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.beam < 1:
            raise ValueError("beam must be at least 1")


@dataclass
class RetrievalResult:
    """What the walk produced: evidence, the audit trail and the cost."""

    selected: list[Node] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    nodes_opened: int = 0
    router_failures: int = 0


class TreeWalker:
    """Reasoning-first retriever over a single document tree."""

    def __init__(
        self, config: RetrievalConfig | None = None, llm: LLMClient | None = None
    ) -> None:
        self.config = config or RetrievalConfig()
        self.llm = llm if llm is not None else LLMClient()

    def run(self, question: str, root: Node) -> RetrievalResult:
        config = self.config
        result = RetrievalResult(usage=Usage(model=self.llm.config.model))
        result.nodes_opened = 1
        if not root.children:
            result.selected = [root]
            return result

        frontier: list[tuple[Node, float]] = [(root, 0.0)]
        reached_leaves: list[tuple[Node, float]] = []
        step_index = 0

        for _ in range(config.max_depth):
            candidates: list[tuple[Node, float, float]] = []
            for parent, parent_score in frontier:
                for child in parent.children:
                    candidates.append((child, 0.0, parent_score))
            if not candidates:
                break

            scorer = BM25([f"{node.title}\n{node.summary}" for node, _, _ in candidates])
            raw_scores = scorer.rank(question)
            scored = [
                (
                    node,
                    raw_score,
                    raw_score + config.ancestor_weight * parent_score,
                )
                for (node, _, parent_score), raw_score in zip(candidates, raw_scores)
            ]
            scored.sort(key=lambda item: item[2], reverse=True)

            limit = len(scored) if config.mode == MODE_REASONING else config.beam
            shortlist = scored[:limit]
            kept_ids, llm_reason = self._route(question, shortlist, result)

            kept: list[tuple[Node, float]] = []
            for position, (node, own_score, combined) in enumerate(scored):
                on_shortlist = position < limit
                if not on_shortlist:
                    decision, rationale = "prune", "ranked below the beam cut-off"
                elif kept_ids is not None:
                    if node.id in kept_ids:
                        decision = "select" if node.is_leaf() else "descend"
                        rationale = llm_reason or "kept by the router"
                    else:
                        decision, rationale = "prune", "dropped by the router"
                else:
                    decision = "select" if node.is_leaf() else "descend"
                    rationale = "within the beam cut-off"

                if len(result.trace) < config.max_trace:
                    step_index += 1
                    result.trace.append(
                        TraceStep(
                            index=step_index,
                            node_id=node.id,
                            title=node.title,
                            level=node.level,
                            score=combined,
                            scorer="bm25+llm" if kept_ids is not None else "bm25",
                            decision=decision,
                            rationale=rationale,
                        )
                    )
                if decision in ("descend", "select"):
                    kept.append((node, combined))

            if not kept:
                kept = [(shortlist[0][0], shortlist[0][2])] if shortlist else []
                if kept and len(result.trace) < config.max_trace:
                    step_index += 1
                    node, combined = kept[0]
                    result.trace.append(
                        TraceStep(
                            index=step_index,
                            node_id=node.id,
                            title=node.title,
                            level=node.level,
                            score=combined,
                            scorer="bm25",
                            decision="descend",
                            rationale="best remaining candidate after an empty router reply",
                        )
                    )

            frontier = []
            for node, combined in kept:
                result.nodes_opened += 1
                if node.is_leaf():
                    reached_leaves.append((node, combined))
                else:
                    frontier.append((node, combined))
            if not frontier:
                break
            if result.nodes_opened >= config.max_nodes:
                reached_leaves.extend((node, score) for node, score in frontier)
                break

        if not reached_leaves:
            reached_leaves = list(frontier)

        result.scores = {node.id: score for node, score in reached_leaves}
        result.selected = _fit_budget(reached_leaves, config)
        return result

    def _route(
        self,
        question: str,
        shortlist: list[tuple[Node, float, float]],
        result: RetrievalResult,
    ) -> tuple[set[str] | None, str]:
        """Ask the model to keep or drop sections. ``None`` means BM25 only."""

        if not self.llm.enabled or self.config.mode == MODE_LEXICAL:
            return None, ""
        if not shortlist:
            return None, ""

        listing = "\n".join(
            f"[{position}] {node.title} :: {truncate(node.summary, 160)}"
            for position, (node, _, _) in enumerate(shortlist, start=1)
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\n\nCandidates:\n{listing}",
            },
        ]
        try:
            text, usage = self.llm.chat(messages, json_mode=True, max_tokens=400)
        except LLMError:
            result.router_failures += 1
            return None, ""

        result.usage.add(**usage)
        try:
            payload = parse_json_block(text)
        except LLMError:
            result.router_failures += 1
            return None, ""

        keep = payload.get("keep")
        if not isinstance(keep, list):
            result.router_failures += 1
            return None, ""
        kept = {
            shortlist[position - 1][0].id
            for position in keep
            if isinstance(position, int) and 1 <= position <= len(shortlist)
        }
        reason = str(payload.get("reason") or "").strip()
        return kept, truncate(reason, 200) if reason else ""


def _fit_budget(
    candidates: list[tuple[Node, float]], config: RetrievalConfig
) -> list[Node]:
    """Take the best leaves that fit in the answerer's context budget."""

    ordered = sorted(
        {node.id: (node, score) for node, score in candidates}.values(),
        key=lambda item: item[1],
        reverse=True,
    )
    selected: list[Node] = []
    budget = config.context_tokens
    for node, _ in ordered:
        cost = node.token_estimate or estimate_tokens(node.text)
        if selected and cost > budget:
            continue
        selected.append(node)
        budget -= cost
        if len(selected) >= config.max_selected:
            break
    return selected
