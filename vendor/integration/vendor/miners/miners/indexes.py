from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .dataset import DocumentStore


def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(int(top_k), scores.size)
    if top_k <= 0:
        return np.array([], dtype=np.int64)
    if top_k == scores.size:
        return np.argsort(-scores)
    selected = np.argpartition(scores, -top_k)[-top_k:]
    return selected[np.argsort(-scores[selected])]


class BM25Index:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.vectorizer: CountVectorizer = joblib.load(self.index_dir / "vectorizer.joblib")
        self.matrix = sparse.load_npz(self.index_dir / "bm25_matrix.npz").tocsr()
        self.ids = np.load(self.index_dir / "doc_ids.npy", mmap_mode="r")
        self.years = np.load(self.index_dir / "doc_years.npy", mmap_mode="r")

    @classmethod
    def build(
        cls,
        store: DocumentStore,
        index_dir: str | Path,
        max_features: int = 300_000,
        min_df: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
        rebuild: bool = False,
    ) -> "BM25Index":
        index_dir = Path(index_dir)
        if (index_dir / "bm25_matrix.npz").exists() and not rebuild:
            return cls(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        ids: List[int] = []
        years: List[int] = []

        def texts():
            for corpus_id, year, text in store.iter_index_rows():
                ids.append(corpus_id)
                years.append(year)
                yield text

        vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[\w-]{2,}\b",
            ngram_range=(1, 1),
            min_df=min_df,
            max_features=max_features,
            dtype=np.float32,
        )
        counts = vectorizer.fit_transform(texts()).tocsr().astype(np.float32)
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = float(document_lengths.mean()) or 1.0
        document_frequency = np.diff(counts.tocsc().indptr).astype(np.float32)
        idf = np.log(
            1.0 + (counts.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)

        for row_index in range(counts.shape[0]):
            start, end = counts.indptr[row_index : row_index + 2]
            if start == end:
                continue
            tf = counts.data[start:end]
            norm = k1 * (1.0 - b + b * document_lengths[row_index] / average_length)
            counts.data[start:end] = tf * (k1 + 1.0) / (tf + norm)
        counts.data *= idf[counts.indices]

        sparse.save_npz(index_dir / "bm25_matrix.npz", counts, compressed=True)
        joblib.dump(vectorizer, index_dir / "vectorizer.joblib")
        np.save(index_dir / "doc_ids.npy", np.asarray(ids, dtype=np.int64))
        np.save(index_dir / "doc_years.npy", np.asarray(years, dtype=np.int16))
        (index_dir / "config.json").write_text(
            json.dumps(
                {
                    "type": "bm25",
                    "documents": len(ids),
                    "features": len(vectorizer.vocabulary_),
                    "k1": k1,
                    "b": b,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(index_dir)

    def search(
        self, query: str, top_k: int = 1000, max_year: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []
        scores = np.asarray(self.matrix[:, query_vector.indices].sum(axis=1)).ravel()
        if max_year is not None:
            scores = scores.copy()
            scores[(self.years > int(max_year)) & (self.years >= 0)] = -np.inf
        indices = _top_indices(scores, top_k)
        return [
            (int(self.ids[index]), float(scores[index]))
            for index in indices
            if np.isfinite(scores[index]) and scores[index] > 0
        ]


class DenseIndex:
    """Sentence-transformer embeddings with exact chunked cosine search."""

    REQUIRED_FILES = ("config.json", "doc_ids.npy", "doc_years.npy", "embeddings.npy")

    def __init__(self, index_dir: str | Path, model_name: Optional[str] = None):
        self.index_dir = Path(index_dir)
        missing = [
            name for name in self.REQUIRED_FILES if not (self.index_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(
                "Dense index is incomplete at "
                f"{self.index_dir}. Missing: {', '.join(missing)}. "
                "Run `miners build-dense --rebuild` "
                "before using --mode dense or --mode hybrid."
            )
        config = json.loads((self.index_dir / "config.json").read_text(encoding="utf-8"))
        self.model_name = model_name or config["model_name"]
        self.ids = np.load(self.index_dir / "doc_ids.npy", mmap_mode="r")
        self.years = np.load(self.index_dir / "doc_years.npy", mmap_mode="r")
        self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self._model = None

    @staticmethod
    def _sentence_transformer(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers. "
                "Install MINERS with the dense optional dependencies."
            ) from exc
        return SentenceTransformer(model_name)

    @property
    def model(self):
        if self._model is None:
            self._model = self._sentence_transformer(self.model_name)
        return self._model

    @classmethod
    def build(
        cls,
        store: DocumentStore,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 64,
        rebuild: bool = False,
    ) -> "DenseIndex":
        index_dir = Path(index_dir)
        complete = all((index_dir / name).exists() for name in cls.REQUIRED_FILES)
        if complete and not rebuild:
            return cls(index_dir, model_name=model_name)
        index_dir.mkdir(parents=True, exist_ok=True)
        model = cls._sentence_transformer(model_name)
        count = store.count()
        probe = model.encode(["dimension probe"], normalize_embeddings=True)
        dimension = int(probe.shape[1])
        embeddings = np.lib.format.open_memmap(
            index_dir / "embeddings.npy",
            mode="w+",
            dtype=np.float16,
            shape=(count, dimension),
        )
        ids = np.empty(count, dtype=np.int64)
        years = np.empty(count, dtype=np.int16)
        batch_texts: List[str] = []
        batch_meta: List[Tuple[int, int]] = []
        cursor = 0
        progress_interval = 50
        next_progress = progress_interval

        def flush() -> None:
            nonlocal cursor, next_progress
            if not batch_texts:
                return
            encoded = model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            end = cursor + len(batch_texts)
            embeddings[cursor:end] = encoded.astype(np.float16)
            ids[cursor:end] = [item[0] for item in batch_meta]
            years[cursor:end] = [item[1] for item in batch_meta]
            cursor = end
            batch_texts.clear()
            batch_meta.clear()
            if cursor >= next_progress or cursor == count:
                print(f"Encoded {cursor:,}/{count:,} documents", flush=True)
                next_progress += progress_interval

        for corpus_id, year, text in store.iter_index_rows():
            batch_texts.append(text)
            batch_meta.append((corpus_id, year))
            if len(batch_texts) >= batch_size:
                flush()
        flush()
        embeddings.flush()
        np.save(index_dir / "doc_ids.npy", ids)
        np.save(index_dir / "doc_years.npy", years)
        (index_dir / "config.json").write_text(
            json.dumps(
                {
                    "type": "dense_exact",
                    "model_name": model_name,
                    "documents": count,
                    "dimension": dimension,
                    "dtype": "float16",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(index_dir, model_name=model_name)

    def search(
        self,
        query: str,
        top_k: int = 1000,
        max_year: Optional[int] = None,
        chunk_size: int = 8192,
    ) -> List[Tuple[int, float]]:
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)
        best_ids = np.array([], dtype=np.int64)
        best_scores = np.array([], dtype=np.float32)
        for start in range(0, len(self.ids), chunk_size):
            end = min(start + chunk_size, len(self.ids))
            block = np.asarray(self.embeddings[start:end], dtype=np.float32)
            scores = block @ query_vector
            if max_year is not None:
                invalid = (self.years[start:end] > int(max_year)) & (
                    self.years[start:end] >= 0
                )
                scores[invalid] = -np.inf
            local = _top_indices(scores, min(top_k, len(scores)))
            best_ids = np.concatenate((best_ids, np.arange(start, end)[local]))
            best_scores = np.concatenate((best_scores, scores[local]))
            keep = _top_indices(best_scores, min(top_k, len(best_scores)))
            best_ids, best_scores = best_ids[keep], best_scores[keep]
        order = np.argsort(-best_scores)
        return [
            (int(self.ids[best_ids[index]]), float(best_scores[index]))
            for index in order
            if np.isfinite(best_scores[index])
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Cross-encoder reranking requires sentence-transformers") from exc
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: Sequence[Dict[str, object]],
        top_k: int = 200,
        batch_size: int = 32,
    ) -> List[Tuple[int, float]]:
        pairs = [
            (query, f"{document.get('title', '')}\n{document.get('abstract', '')}")
            for document in documents
        ]
        scores = np.asarray(self.model.predict(pairs, batch_size=batch_size)).ravel()
        order = _top_indices(scores, top_k)
        return [
            (int(documents[index]["corpus_id"]), float(scores[index]))
            for index in order
        ]
