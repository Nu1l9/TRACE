from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def _normalise_year(value: Any) -> Optional[int]:
    if value in (None, "", "NR"):
        return None
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def _decode_json_array_line(raw: bytes) -> Optional[Dict[str, Any]]:
    """Decode a one-object-per-line JSON array without loading the full corpus."""
    value = raw.strip()
    if value in (b"", b"[", b"]"):
        return None
    if value.endswith(b","):
        value = value[:-1]
    return json.loads(value)


class ReviewDataset:
    """Small task-side files (papers.json and test_ids.json)."""

    def __init__(self, papers_path: str | Path, test_ids_path: str | Path):
        self.papers_path = Path(papers_path)
        self.test_ids_path = Path(test_ids_path)
        self.papers: List[Dict[str, Any]] = json.loads(
            self.papers_path.read_text(encoding="utf-8")
        )
        self.by_id = {int(item["ID"]): item for item in self.papers}
        self.test_ids = {
            int(value)
            for value in json.loads(self.test_ids_path.read_text(encoding="utf-8"))
        }

    def get(self, paper_id: int) -> Dict[str, Any]:
        try:
            return self.by_id[int(paper_id)]
        except KeyError as exc:
            raise KeyError(f"Review task ID {paper_id} does not exist") from exc

    def iter_split(self, split: str = "test") -> Iterator[Dict[str, Any]]:
        if split == "all":
            yield from self.papers
            return
        use_test = split == "test"
        for item in self.papers:
            if (int(item["ID"]) in self.test_ids) == use_test:
                yield item


class DocumentStore:
    """SQLite metadata store plus byte offsets into the original corpus file."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def build(
        cls,
        corpus_path: str | Path,
        db_path: str | Path,
        limit: Optional[int] = None,
        rebuild: bool = False,
    ) -> "DocumentStore":
        corpus_path = Path(corpus_path).resolve()
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists() and not rebuild:
            return cls(db_path)
        if db_path.exists():
            db_path.unlink()

        connection = sqlite3.connect(db_path)
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE documents (
                corpus_id INTEGER PRIMARY KEY,
                pmid TEXT,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                year INTEGER,
                journal TEXT,
                doi TEXT,
                pmc_id TEXT,
                fulltext_source TEXT,
                has_sections INTEGER NOT NULL,
                corpus_offset INTEGER NOT NULL,
                corpus_length INTEGER NOT NULL
            );
            CREATE INDEX idx_documents_year ON documents(year);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("corpus_path", str(corpus_path)),
        )

        rows = []
        count = 0
        with corpus_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                record = _decode_json_array_line(raw)
                if record is None:
                    continue
                sections = record.get("sections") or {}
                rows.append(
                    (
                        int(record["ID"]),
                        str(record.get("pmid") or ""),
                        str(record.get("title") or ""),
                        str(record.get("abstract") or ""),
                        _normalise_year(record.get("year")),
                        str(record.get("journal") or ""),
                        str(record.get("doi") or ""),
                        str(record.get("pmc_id") or ""),
                        str(record.get("fulltext_source") or ""),
                        int(bool(sections)),
                        offset,
                        len(raw),
                    )
                )
                count += 1
                if len(rows) >= 1000:
                    connection.executemany(
                        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    connection.commit()
                    rows.clear()
                if limit is not None and count >= limit:
                    break
        if rows:
            connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("document_count", str(count)),
        )
        connection.commit()
        connection.close()
        return cls(db_path)

    def count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def iter_documents(self) -> Iterator[Dict[str, Any]]:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "SELECT * FROM documents ORDER BY corpus_id"
            )
            for row in cursor:
                yield dict(row)

    def iter_index_rows(self) -> Iterator[tuple[int, int, str]]:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "SELECT corpus_id, COALESCE(year, -1), title, abstract "
                "FROM documents ORDER BY corpus_id"
            )
            for row in cursor:
                text = f"{row['title']}\n{row['abstract']}"
                yield int(row["corpus_id"]), int(row[1]), text

    def get_many(self, corpus_ids: Iterable[int]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(int(value) for value in corpus_ids))
        if not ids:
            return []
        found: Dict[int, Dict[str, Any]] = {}
        with closing(self.connect()) as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT * FROM documents WHERE corpus_id IN ({placeholders})"
                for row in connection.execute(query, chunk):
                    found[int(row["corpus_id"])] = dict(row)
        return [found[value] for value in ids if value in found]

    def get(self, corpus_id: int) -> Optional[Dict[str, Any]]:
        rows = self.get_many([corpus_id])
        return rows[0] if rows else None

    def _corpus_path(self) -> Path:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='corpus_path'"
            ).fetchone()
        if row is None:
            raise RuntimeError("Document store has no corpus_path metadata")
        return Path(row[0])

    def get_source_record(self, corpus_id: int) -> Optional[Dict[str, Any]]:
        metadata = self.get(corpus_id)
        if metadata is None:
            return None
        with self._corpus_path().open("rb") as stream:
            stream.seek(int(metadata["corpus_offset"]))
            raw = stream.read(int(metadata["corpus_length"]))
        return _decode_json_array_line(raw)

    def get_fulltext_sections(self, corpus_id: int) -> Dict[str, str]:
        record = self.get_source_record(corpus_id) or {}
        sections: Dict[str, str] = {}
        for key in ("introduction", "methods", "results", "discussion", "conclusion"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                sections[key] = value.strip()
        nested = record.get("sections") or {}
        if isinstance(nested, dict):
            for key, value in nested.items():
                if isinstance(value, str) and value.strip():
                    sections[str(key)] = value.strip()
        return sections
