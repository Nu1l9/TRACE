"""Run the frozen Analysis GraphRAG analysis component on an upstream Evidence contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
load_dotenv(PACKAGE_ROOT / ".env")
load_dotenv(PACKAGE_ROOT.parent / ".env")

from analysis_engine.method_selector import LLMStatisticalMethodSelector
from analysis_engine.planner import LLMAnalysisPlanner
from analysis_engine.reporting_bias_selector import LLMReportingBiasSelector
from analysis_engine.retrieval import OpenAIEmbeddingBackend
from analysis_engine.retrieval_artifacts import embedding_retrieval_artifact, graph_artifact
from analysis_engine.runtime_facets import LLMRuntimeFacetAnnotator
from analysis_engine.semantic_graph import SourceSemantics
from analysis_engine.topic_reviewer_graph import GraphTopicReviewer
from analysis_engine.graph_agent import GraphAnalysisAgent
from analysis_engine.reporting import render_graph_report
from contract_adapter import load_analysis_contract


def _env(name: str, fallback: str | None = None) -> str | None:
    return os.getenv(name) or fallback


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _direction(statistics, directions: list[str]) -> str:
    declared = {value for value in directions if value}
    if len(declared) == 1:
        return next(iter(declared))
    if statistics.effect_type in {"SMD", "MD", "RD", "sensitivity", "specificity", "proportion"}:
        return "source_defined_direction"
    if statistics.ci_lower <= 1 <= statistics.ci_upper:
        return "no_clear_difference"
    return "decreased_risk" if statistics.pooled_effect < 1 else "increased_risk"


def _summary(result, candidate_directions: dict[str, str]) -> dict[str, object]:
    analyses = []
    for item in result.analyses:
        pooled = None
        if item.statistics:
            pooled = {
                "effect_type": item.statistics.effect_type,
                "pooled_effect": item.statistics.pooled_effect,
                "ci_lower": item.statistics.ci_lower,
                "ci_upper": item.statistics.ci_upper,
                "heterogeneity_i2": item.statistics.i_squared,
                "direction": _direction(
                    item.statistics,
                    [candidate_directions.get(evidence_id, "") for evidence_id in item.selected_evidence_ids],
                ),
                "member_candidate_ids": item.statistics.member_evidence_ids,
            }
        analyses.append({
            "analysis_id": item.analysis.analysis_id,
            "title": item.analysis.title,
            "kind": item.analysis.kind,
            "status": item.status,
            "selected_candidate_ids": item.selected_evidence_ids,
            "pooled_result": pooled,
            "warnings": item.warnings,
        })
    return {"status": result.status, "analyses": analyses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing the fixed analysis JSON contract.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty or new folder for run artifacts.")
    parser.add_argument("--check", action="store_true", help="Validate input schema without any API or R calls.")
    args = parser.parse_args()

    protocol, task, profile, payloads, study_by_article = load_analysis_contract(args.input_dir)
    candidate_count = sum(len(item["evidence"]) for item in payloads)
    if args.check:
        print(json.dumps({
            "status": "valid",
            "task_id": task["task_id"],
            "evidence_count": candidate_count,
            "article_count": len(payloads),
            "profile_version": profile["profile_version"],
            "gold_accessed": False,
            "remote_calls": 0,
        }, ensure_ascii=False, indent=2))
        return

    llm_key = _env("ANALYSIS_LLM_API_KEY", _env("OPENAI_API_KEY"))
    embedding_key = _env("ANALYSIS_EMBEDDING_API_KEY", _env("OPENAI_API_KEY"))
    if not llm_key:
        raise RuntimeError("Set ANALYSIS_LLM_API_KEY (or OPENAI_API_KEY) in .env.")
    if not embedding_key:
        raise RuntimeError("Set ANALYSIS_EMBEDDING_API_KEY (or OPENAI_API_KEY) in .env.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_model = os.getenv("ANALYSIS_LLM_MODEL", "gpt-5.6-luna")
    llm_base_url = os.getenv("ANALYSIS_LLM_BASE_URL", "https://api.openai.com/v1")
    embedding_model = os.getenv("ANALYSIS_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url = os.getenv("ANALYSIS_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    max_retries = int(os.getenv("ANALYSIS_API_MAX_RETRIES", "3"))

    embeddings = OpenAIEmbeddingBackend(
        model=embedding_model, api_key=embedding_key, base_url=embedding_base_url,
        cache_path=output_dir / "embedding_cache.json", max_retries=max_retries,
    )
    common = {
        "model": llm_model,
        "api_key": llm_key,
        "base_url": llm_base_url,
        "max_retries": max_retries,
    }
    planner = LLMAnalysisPlanner(reasoning_effort=os.getenv("ANALYSIS_LLM_REASONING_EFFORT", "high"), **common)
    reviewer = GraphTopicReviewer(always_review=True, hide_calculation_details=True, **common)
    facet_annotator = None
    method_selector = LLMStatisticalMethodSelector(prompt_style="neutral_neutral", **common)
    reporting_bias_selector = LLMReportingBiasSelector(**common)
    agent = GraphAnalysisAgent(
        embeddings=embeddings,
        planner=planner,
        reviewer=reviewer,
        runtime_facet_annotator=facet_annotator,
        method_selector=method_selector,
        reporting_bias_selector=reporting_bias_selector,
        pre_review_filter_mode="semantic_semantic",
        audit_dir=output_dir,
        progress_callback=lambda message: print(message, flush=True),
    )
    print(
        f"[Start] task={task['task_id']}; Evidence={candidate_count}; GraphRAG=semantic source semantics; Gold=false.",
        flush=True,
    )
    result = agent.run(
        protocol=protocol,
        task=task,
        analysis_profile=profile,
        extractor_payloads=payloads,
        study_by_article=study_by_article,
        reporting_bias_artifact_dir=output_dir / "reporting_bias",
        forest_plot_artifact_dir=output_dir / "forest_plots",
    )
    (output_dir / "analysis_result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "analysis_report.md").write_text(render_graph_report(result), encoding="utf-8")
    if agent.last_graph is not None:
        _write_json(output_dir / "graph.json", graph_artifact(agent.last_graph))
    if agent.last_candidate_sets is not None:
        _write_json(output_dir / "embedding_retrieval.json", embedding_retrieval_artifact(agent.last_candidate_sets))

    raw_candidates = json.loads(
        (args.input_dir / "analysis_evidence_candidates.json").read_text(encoding="utf-8")
        if (args.input_dir / "analysis_evidence_candidates.json").exists()
        else (args.input_dir / "inputs" / "analysis_evidence_candidates.json").read_text(encoding="utf-8")
    )["candidates"]
    direction_by_id = {
        str(item["candidate_id"]): str((item.get("effect") or {}).get("direction") or "")
        for item in raw_candidates
    }
    _write_json(output_dir / "analysis_summary.json", _summary(result, direction_by_id))
    _write_json(output_dir / "run_metadata.json", {
        "release": "evidence_analysis",
        "architecture": "semantic_source_semantic_graphrag_with_expandable_review",
        "task_id": task["task_id"],
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "transport": {
            "embedding": embeddings.diagnostics(),
            "planner_llm_calls": planner.calls_made,
            "reviewer_llm_calls": reviewer.calls_made,
            "source_semantics_llm_calls": facet_annotator.calls_made,
            "method_selector_llm_calls": method_selector.calls_made,
            "reporting_bias_selector_llm_calls": reporting_bias_selector.calls_made,
        },
        "gold_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"status": result.status, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
