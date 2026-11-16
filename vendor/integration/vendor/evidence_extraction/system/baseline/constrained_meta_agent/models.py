from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentInput:
    article_id: str
    document_id: str
    representations: dict[str, Path]


@dataclass(frozen=True)
class LoadedTask:
    case_id: str
    task_id: str
    topic: str
    pico: dict[str, Any]
    target_exposure: str
    allowed_effect_types: list[str]
    estimands: list[str]
    model_selection_policy: str
    exclusion_concepts: str
    allowed_representations: list[str]
    included_article_ids: list[str]
    execution_policy: dict[str, Any]
    documents: dict[str, DocumentInput]
    raw: dict[str, Any]
