from __future__ import annotations

from pathlib import Path

import pytest

from tracerag.pipeline import index_path
from tracerag.store import Corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("index")


@pytest.fixture(scope="session")
def indexed(corpus_root: Path) -> Corpus:
    corpus = Corpus(corpus_root)
    index_path(CORPUS_DIR, corpus=corpus, data_dir=corpus_root)
    return corpus


@pytest.fixture(scope="session")
def doc_ids(indexed: Corpus) -> dict[str, str]:
    return {meta.source_name: meta.doc_id for meta in indexed.list_docs()}
