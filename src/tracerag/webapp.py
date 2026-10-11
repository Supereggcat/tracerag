"""FastAPI application backing the TraceRAG web UI."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tracerag.llm import LLMClient, LLMConfig
from tracerag.pipeline import ask as pipeline_ask
from tracerag.pipeline import index_file
from tracerag.retrieve import MODES, RetrievalConfig
from tracerag.store import Corpus

WEB_DIR = Path(
    os.getenv("TRACERAG_WEB_DIR")
    or Path(__file__).resolve().parents[2] / "web"
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_id: str
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=12)


def create_app(
    data_dir: str | Path = "data/index", *, llm: LLMClient | None = None
) -> FastAPI:
    app = FastAPI(title="TraceRAG", version="0.1.0")
    corpus = Corpus(data_dir)
    client = llm if llm is not None else LLMClient(LLMConfig.from_env())
    app.state.corpus = corpus
    app.state.llm = client

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": client.config.model,
            "llm_enabled": client.enabled,
            "documents": len(corpus.list_docs()),
        }

    @app.get("/api/config")
    def config() -> dict:
        return {
            "llm_enabled": client.enabled,
            "model": client.config.model,
            "base_url": client.config.base_url,
            "modes": list(MODES),
        }

    @app.get("/api/docs")
    def list_docs() -> list[dict]:
        return [meta.to_dict() for meta in corpus.list_docs()]

    @app.get("/api/docs/{doc_id}/tree")
    def document_tree(doc_id: str) -> dict:
        if not corpus.exists(doc_id):
            raise HTTPException(status_code=404, detail="unknown document")
        return corpus.load_tree(doc_id).to_dict()

    @app.get("/api/docs/{doc_id}")
    def document_meta(doc_id: str) -> dict:
        if not corpus.exists(doc_id):
            raise HTTPException(status_code=404, detail="unknown document")
        return corpus.load_meta(doc_id).to_dict()

    @app.delete("/api/docs/{doc_id}")
    def delete_document(doc_id: str) -> dict:
        return {"removed": corpus.delete(doc_id)}

    @app.post("/api/docs")
    async def upload_document(file: UploadFile = File(...)) -> dict:
        original = Path(file.filename or "upload.txt").name
        staging = Path(tempfile.mkdtemp(prefix="tracerag-"))
        temporary = staging / original
        temporary.write_bytes(await file.read())
        try:
            meta = index_file(temporary, corpus=corpus, data_dir=data_dir)
        except Exception as exc:  # surfaced to the UI as a 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return meta.to_dict()

    @app.post("/api/query")
    def query(request: QueryRequest) -> dict:
        if request.mode not in MODES:
            raise HTTPException(status_code=400, detail=f"mode must be one of {list(MODES)}")
        config = RetrievalConfig(mode=request.mode, max_selected=request.top_k)
        try:
            answer = pipeline_ask(
                request.question,
                request.doc_id,
                config=config,
                llm=client,
                corpus=corpus,
                data_dir=data_dir,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return answer.to_dict()

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app
