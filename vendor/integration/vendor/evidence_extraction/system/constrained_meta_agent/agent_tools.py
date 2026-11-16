from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .models import DocumentInput, LoadedTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def inspect_document(document: DocumentInput) -> dict[str, Any]:
    """Return local, auditable facts without making an LLM call."""
    details: dict[str, Any] = {
        "article_id": document.article_id,
        "document_id": document.document_id,
        "representations": {},
    }
    for name, path in document.representations.items():
        item: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
        if name == "markdown":
            text = path.read_text(encoding="utf-8")
            item["characters"] = len(text)
            item["heading_count"] = sum(1 for line in text.splitlines() if line.startswith("#"))
        elif name == "content_list":
            import json

            item["item_count"] = len(json.loads(path.read_text(encoding="utf-8")))
        details["representations"][name] = item
    return details


def legacy_graph_extract(document: DocumentInput, task: LoadedTask) -> dict[str, Any]:
    """Use the existing graph unchanged as an isolated compatibility tool."""
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
    from graph import app  # Imported lazily so dry-run never loads the legacy graph.

    markdown = document.representations["markdown"].read_text(encoding="utf-8")
    pico_context = (
        f"{task.topic}\n"
        f"Population: {task.pico['population']}\n"
        f"Exposure: {task.pico['exposure']}\n"
        f"Comparator: {task.pico['comparator']}\n"
        f"Outcomes: {', '.join(task.pico['outcomes'])}"
    )
    result = app.invoke(
        {
            "topic": pico_context,
            "config": {
                "pico": task.pico,
                "target_exposure_concept": task.target_exposure,
                "target_population": task.pico["population"],
                "target_comparator": task.pico["comparator"],
                "target_outcomes": task.pico["outcomes"],
                "allowed_effect_types": task.allowed_effect_types,
                "estimands": task.estimands,
                "model_selection_policy": task.model_selection_policy,
                "exclusion_concepts": task.exclusion_concepts,
            },
            "papers": [{"id": document.article_id, "content": markdown, "filename": document.representations["markdown"].name}],
            "structured_data": [],
            "baseline_data": [],
            "meta_stats": {},
            "next_action": "",
        },
        config={"recursion_limit": task.execution_policy["legacy_graph_recursion_limit"]},
    )
    serialized = []
    for item in result.get("structured_data", []):
        serialized.append(item.model_dump() if hasattr(item, "model_dump") else item)
    return {"structured_data": serialized, "meta_stats": result.get("meta_stats", {})}
