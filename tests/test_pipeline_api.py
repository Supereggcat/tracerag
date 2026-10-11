from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tracerag.llm import LLMClient, LLMConfig
from tracerag.pipeline import ask, index_file, index_path
from tracerag.retrieve import RetrievalConfig
from tracerag.store import Corpus
from tracerag.webapp import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


def _fresh(tmp_path: Path) -> Corpus:
    corpus = Corpus(tmp_path / "index")
    index_path(CORPUS_DIR, corpus=corpus, data_dir=tmp_path / "index")
    return corpus


def test_index_reports_tree_shape(tmp_path: Path) -> None:
    meta = index_file(CORPUS_DIR / "atlas-handbook.md", data_dir=tmp_path / "index")
    assert meta.nodes > 10
    assert meta.leaves >= 1
    assert meta.max_depth == 3
    assert meta.outline and meta.outline[0]["level"] == 1


def test_ask_returns_citations_trace_and_usage(tmp_path: Path) -> None:
    corpus = _fresh(tmp_path)
    doc_id = next(m.doc_id for m in corpus.list_docs() if m.source_name == "atlas-handbook.md")
    answer = ask(
        "退款到账要多久?",
        doc_id,
        data_dir=tmp_path / "index",
        config=RetrievalConfig(mode="lexical"),
        llm=LLMClient(LLMConfig(api_key="")),
        corpus=corpus,
    )
    assert answer.citations
    assert answer.trace
    assert answer.selected
    assert answer.supported_ratio > 0
    assert answer.usage.calls == 0
    assert answer.to_dict()["doc_id"] == doc_id


def test_unknown_document_raises(tmp_path: Path) -> None:
    corpus = _fresh(tmp_path)
    try:
        ask("hello", "missing-doc", data_dir=tmp_path / "index", corpus=corpus)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_corpus_round_trip(tmp_path: Path) -> None:
    corpus = _fresh(tmp_path)
    metas = corpus.list_docs()
    assert len(metas) == 3
    doc_id = metas[0].doc_id
    assert corpus.exists(doc_id)
    assert corpus.load_meta(doc_id).doc_id == doc_id
    assert corpus.load_tree(doc_id).children
    assert corpus.delete(doc_id) is True
    assert corpus.exists(doc_id) is False


def test_api_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "index"
    client = TestClient(
        create_app(data_dir, llm=LLMClient(LLMConfig(api_key="")))
    )

    payload = (CORPUS_DIR / "orbit-api-notes.md").read_bytes()
    upload = client.post(
        "/api/docs",
        files={"file": ("orbit-api-notes.md", payload, "text/markdown")},
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["doc_id"]

    health = client.get("/api/health").json()
    assert health["llm_enabled"] is False
    assert health["documents"] == 1

    tree = client.get(f"/api/docs/{doc_id}/tree").json()
    assert tree["title"]
    assert tree["children"]

    response = client.post(
        "/api/query",
        json={
            "question": "How long until a cursor stops working?",
            "doc_id": doc_id,
            "mode": "lexical",
            "top_k": 4,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["citations"]
    assert any("Pagination" in item["title"] for item in body["citations"])
    assert body["ledger"]

    assert client.post(
        "/api/query",
        json={"question": "x", "doc_id": doc_id, "mode": "nonsense"},
    ).status_code == 400
    assert client.get("/api/docs/missing/tree").status_code == 404
    assert client.delete(f"/api/docs/{doc_id}").json()["removed"] is True


def test_upload_rejects_unsupported_files(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "index", llm=LLMClient(LLMConfig(api_key=""))))
    response = client.post(
        "/api/docs", files={"file": ("data.bin", b"\x00\x01\x02", "application/octet-stream")}
    )
    assert response.status_code == 400
