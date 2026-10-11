from __future__ import annotations

from pathlib import Path

import pytest

from tracerag.ingest import load_blocks
from tracerag.tree import build_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_markdown_headings_become_nested_nodes() -> None:
    blocks = load_blocks(REPO_ROOT / "data" / "corpus" / "atlas-handbook.md")
    tree = build_tree(blocks)
    titles = [node.title for node in tree.walk()]
    assert "4. 退款流程" in titles
    assert "4.2 财务侧审批" in titles

    section = next(node for node in tree.walk() if node.title == "4.2 财务侧审批")
    assert section.level == 3
    assert "五个工作日" in section.text
    assert section.parent_id is not None


def test_ids_are_hierarchical_and_stable() -> None:
    blocks = load_blocks(REPO_ROOT / "data" / "corpus" / "orbit-api-notes.md")
    first = build_tree(blocks)
    second = build_tree(blocks)
    assert [node.id for node in first.walk()] == [node.id for node in second.walk()]
    assert first.children[0].id == "n1"
    assert first.children[0].children[0].id == "n1.1"


def test_plain_text_attaches_to_root_when_no_headings_exist(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("just a paragraph of prose without any structure at all.", encoding="utf-8")
    blocks = load_blocks(source)
    tree = build_tree(blocks)
    assert blocks[0].heading_path == ()
    assert tree.children == []
    assert "just a paragraph" in tree.text


def test_oversized_leaf_is_split_into_windows() -> None:
    from tracerag.models import Block

    block = Block(order=0, page=1, heading_path=("Section",), text="sentence. " * 1200)
    tree = build_tree([block], max_leaf_chars=1200, overlap_chars=100)
    section = tree.children[0]
    assert section.children, "a long section should gain child windows"
    assert all("*" not in child.title for child in section.children)
    assert all(child.parent_id == section.id for child in section.children)


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "data.xyz"
    source.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_blocks(source)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_blocks(tmp_path / "nope.md")
