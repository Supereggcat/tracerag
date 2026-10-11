from __future__ import annotations

from tracerag.text import (
    content_tokens,
    estimate_tokens,
    overlap_ratio,
    slugify,
    split_sentences,
    tokenize,
    truncate,
)


def test_tokenize_keeps_latin_words_and_cjk_bigrams() -> None:
    tokens = tokenize("BM25 检索 rate-limit")
    assert "bm25" in tokens
    assert "rate-limit" in tokens
    assert "检索" in tokens
    assert "索" in tokens


def test_tokenize_does_not_join_cjk_across_runs() -> None:
    tokens = tokenize("退款 流程")
    assert "款流" not in tokens


def test_split_sentences_handles_both_scripts() -> None:
    text = "第一句。第二句! Third one. Fourth one."
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "第一句。"
    assert sentences[1] == "第二句!"
    assert sentences[2] == "Third one."


def test_stopwords_are_removed_from_content_tokens() -> None:
    assert "the" not in content_tokens("the retention period")
    assert "retention" in content_tokens("the retention period")


def test_overlap_ratio_is_bounded() -> None:
    assert overlap_ratio(["a", "b"], ["a", "b"]) == 1.0
    assert overlap_ratio(["a", "b"], ["c"]) == 0.0
    assert overlap_ratio([], ["c"]) == 0.0


def test_estimate_tokens_grows_with_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") < estimate_tokens("hello world " * 20)
    assert estimate_tokens("这是一段中文") > 0


def test_truncate_and_slugify() -> None:
    assert truncate("abc", 10) == "abc"
    assert truncate("abcdefghij", 5).endswith("\u2026")
    assert slugify("Atlas Handbook.md") == "atlas-handbook-md"
    assert slugify("手册") == "document"
