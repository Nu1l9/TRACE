from __future__ import annotations

import json
from pathlib import Path

from .models import DocumentInput, LoadedTask


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_case_config(config_path: Path) -> LoadedTask:
    config_path = config_path.resolve()
    case_dir = config_path.parent
    config = _load_json(config_path)
    if config.get("workflow", {}).get("extractor") == "legacy_graph_compatibility":
        raise ValueError("legacy_graph_compatibility is not distributed; use native or typed extraction")
    input_index_path = case_dir / config["inputs"]["document_index"]
    input_index = _load_json(input_index_path)
    input_dir = input_index_path.parent
    manifest = _load_json(input_dir / input_index["manifest_file"])
    paths = _load_json(input_dir / input_index["paths_file"])

    included_ids = config["inputs"]["included_article_ids"]
    if not included_ids:
        raise ValueError("case config must contain at least one included article")
    if len(included_ids) != len(set(included_ids)):
        raise ValueError("case config contains duplicate included article IDs")

    forbidden_ids = set(config["inputs"].get("forbidden_document_ids", []))
    overlap = forbidden_ids.intersection(included_ids)
    if overlap:
        raise ValueError(f"Documents cannot be both included and forbidden: {sorted(overlap)}")

    indexed_ids = set(input_index["article_ids"])
    allowed_representations = config["inputs"]["representations"]
    loaded_documents: dict[str, DocumentInput] = {}
    for article_id in included_ids:
        if article_id not in indexed_ids:
            raise ValueError(f"{article_id} is not listed in the document index")
        entry = manifest["documents"].get(article_id)
        path_entry = paths["documents"].get(article_id)
        if not entry or not path_entry:
            raise ValueError(f"Missing manifest or path mapping for {article_id}")

        representations: dict[str, Path] = {}
        for representation in allowed_representations:
            raw_path = path_entry.get(representation)
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {representation} input: {path}")
            representations[representation] = path

        if config["workflow"]["extractor"] == "legacy_graph_compatibility" and "markdown" not in representations:
            raise ValueError("legacy_graph_compatibility requires markdown input")
        loaded_documents[article_id] = DocumentInput(
            article_id=article_id,
            document_id=entry["document_id"],
            representations=representations,
        )

    pico = config["pico"]
    extraction = config["extraction"]
    return LoadedTask(
        case_id=config["case_id"],
        task_id=config["task_id"],
        topic=config["topic"],
        pico=pico,
        target_exposure=pico["exposure"],
        allowed_effect_types=extraction["allowed_effect_types"],
        estimands=extraction["estimands"],
        model_selection_policy=extraction["model_selection_policy"],
        exclusion_concepts=extraction["exclusion_concepts"],
        allowed_representations=allowed_representations,
        included_article_ids=included_ids,
        execution_policy=config["execution_policy"],
        documents=loaded_documents,
        raw=config,
    )


def load_smoke_task(case_dir: Path, article_id: str) -> LoadedTask:
    task = load_case_config(case_dir / "case_config.json")
    if article_id not in task.included_article_ids:
        raise ValueError(f"{article_id} is not included by this case config")
    return task
