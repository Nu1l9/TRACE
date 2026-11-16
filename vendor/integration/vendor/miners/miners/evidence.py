from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class EvidenceChunk:
    section: str
    text: str
    score: float


def _chunk_text(text: str, chunk_chars: int = 1800, overlap_chars: int = 250) -> List[str]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n{paragraph}".strip()
        else:
            current = f"{current}\n{paragraph}".strip()
        while len(current) > chunk_chars * 2:
            chunks.append(current[:chunk_chars])
            current = current[chunk_chars - overlap_chars :]
    if current:
        chunks.append(current)
    return chunks


class FullTextEvidenceRetriever:
    """Per-paper lexical RAG over available full-text sections."""

    def __init__(self, chunk_chars: int = 1800, overlap_chars: int = 250):
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def retrieve(
        self,
        sections: Dict[str, str],
        query: str,
        top_k: int = 6,
        max_total_chars: int = 12_000,
    ) -> List[EvidenceChunk]:
        labels: List[str] = []
        texts: List[str] = []
        for section, value in sections.items():
            for chunk in _chunk_text(value, self.chunk_chars, self.overlap_chars):
                labels.append(section)
                texts.append(chunk)
        if not texts:
            return []
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[\w-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts + [query])
        scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        order = np.argsort(-scores)
        selected: List[EvidenceChunk] = []
        used_chars = 0
        for index in order:
            text = texts[int(index)]
            if used_chars + len(text) > max_total_chars and selected:
                continue
            selected.append(
                EvidenceChunk(labels[int(index)], text, float(scores[int(index)]))
            )
            used_chars += len(text)
            if len(selected) >= top_k or used_chars >= max_total_chars:
                break
        return selected
