"""Command line entry point for TraceRAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tracerag.evaluate import format_report, load_cases, run_eval
from tracerag.llm import LLMClient, LLMConfig
from tracerag.models import Answer
from tracerag.pipeline import ask as pipeline_ask
from tracerag.pipeline import index_path
from tracerag.retrieve import MODES, RetrievalConfig
from tracerag.store import Corpus

DEFAULT_DATA_DIR = "data/index"
DEFAULT_CASES = "eval/questions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracerag",
        description="Vectorless, reasoning-driven retrieval with an auditable trace.",
    )
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="index directory")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="ingest files or directories")
    index.add_argument("paths", nargs="+")
    index.add_argument("--json", action="store_true")

    docs = sub.add_parser("docs", help="list indexed documents")
    docs.add_argument("--json", action="store_true")

    remove = sub.add_parser("rm", help="drop an indexed document")
    remove.add_argument("doc_id")

    ask = sub.add_parser("ask", help="ask a question about one document")
    ask.add_argument("question")
    ask.add_argument("--doc", required=True, help="document id")
    ask.add_argument("--mode", choices=MODES, default="hybrid")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--trace", action="store_true", help="print the retrieval trace")
    ask.add_argument("--json", action="store_true")

    evaluate = sub.add_parser("eval", help="score against naive chunking")
    evaluate.add_argument("--cases", default=DEFAULT_CASES)
    evaluate.add_argument("--mode", choices=MODES, default="lexical")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)

    if args.command == "index":
        metas = []
        for target in args.paths:
            metas.extend(index_path(target, data_dir=data_dir))
        if args.json:
            print(json.dumps([meta.to_dict() for meta in metas], ensure_ascii=False, indent=2))
        else:
            for meta in metas:
                print(
                    f"indexed {meta.doc_id}  nodes={meta.nodes} leaves={meta.leaves} "
                    f"depth={meta.max_depth} pages={meta.pages} tokens~{meta.estimated_tokens}"
                )
        return 0

    if args.command == "docs":
        metas = Corpus(data_dir).list_docs()
        if args.json:
            print(json.dumps([meta.to_dict() for meta in metas], ensure_ascii=False, indent=2))
        elif not metas:
            print("no documents indexed yet")
        else:
            for meta in metas:
                print(
                    f"{meta.doc_id}  {meta.source_name}  nodes={meta.nodes} "
                    f"tokens~{meta.estimated_tokens}  {meta.created_at}"
                )
        return 0

    if args.command == "rm":
        removed = Corpus(data_dir).delete(args.doc_id)
        print("removed" if removed else "not found")
        return 0 if removed else 1

    if args.command == "ask":
        config = RetrievalConfig(mode=args.mode, max_selected=args.top_k)
        answer = pipeline_ask(
            args.question,
            args.doc,
            data_dir=data_dir,
            config=config,
            llm=LLMClient(LLMConfig.from_env()),
        )
        if args.json:
            print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_answer(answer, show_trace=args.trace)
        return 0

    if args.command == "eval":
        cases = load_cases(args.cases)
        report = run_eval(
            cases,
            data_dir=data_dir,
            mode=args.mode,
            top_k=args.top_k,
            llm=LLMClient(LLMConfig.from_env()),
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_report(report))
        return 0

    if args.command == "serve":
        from tracerag.webapp import create_app

        import uvicorn

        uvicorn.run(
            create_app(data_dir, llm=LLMClient(LLMConfig.from_env())),
            host=args.host,
            port=args.port,
        )
        return 0

    return 1


def _force_utf8_stdout() -> None:
    """Keep CJK output readable in Windows consoles that default to a legacy code page."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def print_answer(answer: Answer, *, show_trace: bool = False) -> None:
    print(answer.text)
    print()
    print("Citations")
    print("---------")
    if not answer.citations:
        print("(none)")
    for citation in answer.citations:
        print(
            f"[{citation.ref}] {citation.node_id}  {citation.title}  "
            f"p.{citation.page_start}-{citation.page_end}  score={citation.score:.3f}"
        )
    print()
    print(f"Evidence ledger (support {answer.supported_ratio:.0%})")
    print("---------------")
    for entry in answer.ledger:
        mark = "[ok]" if entry.supported else "[!!]"
        refs = ",".join(str(ref) for ref in entry.citation_refs) or "-"
        print(f"{mark} refs={refs} support={entry.support:.2f}  {entry.sentence}")
    print()
    usage = answer.usage
    print(
        f"mode={answer.mode} model={usage.model or 'offline'} calls={usage.calls} "
        f"tokens={usage.total_tokens} cost=${usage.cost_usd:.6f} {usage.latency_ms}ms"
    )
    if show_trace:
        print()
        print("Retrieval trace")
        print("---------------")
        for step in answer.trace:
            print(
                f"{step.index:>3}. {step.decision:<8} {step.node_id:<8} "
                f"score={step.score:>7.3f} {step.scorer:<9} {step.title}"
            )
            if step.rationale:
                print(f"       -> {step.rationale}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
