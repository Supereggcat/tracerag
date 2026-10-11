"""A dependency-free BM25 scorer.

This is the router that narrows each level of the tree before any model is
consulted. Keeping it pure Python matters: it is what lets TraceRAG answer
questions with no API key and no network at all.
"""

from __future__ import annotations

import math
from collections import Counter

from tracerag.text import tokenize


class BM25:
    """Okapi BM25 over an in-memory list of documents."""

    def __init__(
        self, documents: list[str], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.corpus]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.frequencies = [Counter(tokens) for tokens in self.corpus]
        self.document_frequency: Counter[str] = Counter()
        for frequency in self.frequencies:
            self.document_frequency.update(frequency.keys())
        self.total_documents = len(self.corpus)

    def _idf(self, term: str) -> float:
        frequency = self.document_frequency.get(term, 0)
        return math.log(
            1.0 + (self.total_documents - frequency + 0.5) / (frequency + 0.5)
        )

    def score(self, query: str, document_index: int) -> float:
        if not self.corpus or document_index >= self.total_documents:
            return 0.0
        terms = tokenize(query)
        if not terms:
            return 0.0
        frequency = self.frequencies[document_index]
        length = self.lengths[document_index] or 1
        total = 0.0
        for term in set(terms):
            appearances = frequency.get(term, 0)
            if not appearances:
                continue
            denominator = appearances + self.k1 * (
                1 - self.b + self.b * length / (self.average_length or 1)
            )
            total += self._idf(term) * appearances * (self.k1 + 1) / denominator
        return total

    def rank(self, query: str) -> list[float]:
        return [self.score(query, index) for index in range(self.total_documents)]
