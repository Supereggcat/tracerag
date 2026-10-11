"""On-disk corpus storage: one directory per indexed document."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracerag.models import Node
from tracerag.text import slugify

INDEX_VERSION = 1


@dataclass
class DocMeta:
    doc_id: str
    title: str
    source_name: str
    suffix: str
    created_at: str
    sha256: str
    pages: int = 1
    nodes: int = 0
    leaves: int = 0
    blocks: int = 0
    characters: int = 0
    estimated_tokens: int = 0
    max_depth: int = 0
    outline: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocMeta":
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in known})


class Corpus:
    """A directory of indexed documents."""

    def __init__(self, root: str | Path = "data/index") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def doc_dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def exists(self, doc_id: str) -> bool:
        return (self.doc_dir(doc_id) / "index.json").exists()

    def make_doc_id(self, name: str, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        return f"{slugify(Path(name).stem)[:48]}-{digest[:8]}"

    def save(
        self,
        *,
        doc_id: str,
        meta: DocMeta,
        tree: Node,
        source_bytes: bytes,
        suffix: str,
    ) -> None:
        directory = self.doc_dir(doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"source{suffix}").write_bytes(source_bytes)
        (directory / "meta.json").write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (directory / "index.json").write_text(
            json.dumps(
                {"version": INDEX_VERSION, "root": tree.to_dict()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_tree(self, doc_id: str) -> Node:
        payload = self._read_json(self.doc_dir(doc_id) / "index.json")
        return Node.from_dict(payload["root"])

    def load_meta(self, doc_id: str) -> DocMeta:
        payload = self._read_json(self.doc_dir(doc_id) / "meta.json")
        return DocMeta.from_dict(payload)

    def list_docs(self) -> list[DocMeta]:
        documents: list[DocMeta] = []
        for directory in sorted(self.root.iterdir() if self.root.exists() else []):
            meta_path = directory / "meta.json"
            if not directory.is_dir() or not meta_path.exists():
                continue
            try:
                documents.append(DocMeta.from_dict(self._read_json(meta_path)))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        documents.sort(key=lambda item: item.created_at, reverse=True)
        return documents

    def delete(self, doc_id: str) -> bool:
        directory = self.doc_dir(doc_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def build_meta(
    *,
    doc_id: str,
    title: str,
    source_name: str,
    suffix: str,
    payload: bytes,
    tree: Node,
    block_count: int,
    pages: int,
) -> DocMeta:
    nodes = list(tree.walk())
    leaves = [node for node in nodes if node.is_leaf()]
    return DocMeta(
        doc_id=doc_id,
        title=title,
        source_name=source_name,
        suffix=suffix,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sha256=hashlib.sha256(payload).hexdigest(),
        pages=pages,
        nodes=len(nodes),
        leaves=len(leaves),
        blocks=block_count,
        characters=sum(len(node.text) for node in nodes),
        estimated_tokens=sum(node.token_estimate for node in nodes),
        max_depth=max((node.level for node in nodes), default=0),
        outline=_outline(tree),
    )


def _outline(tree: Node, limit: int = 200) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for node in tree.walk():
        if node.id == tree.id:
            continue
        entries.append(
            {
                "id": node.id,
                "title": node.title,
                "level": node.level,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "tokens": node.token_estimate,
                "leaf": node.is_leaf(),
            }
        )
        if len(entries) >= limit:
            break
    return entries
