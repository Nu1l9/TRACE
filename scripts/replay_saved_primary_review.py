"""Locally recalculate a primary topic from a saved Reviewer decision.

This is an auditable sensitivity preview. It never calls a model and must not
be presented as a fresh end-to-end analysis or as a replacement for review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "vendor/integration"),
    str(ROOT / "vendor/integration/vendor/evidence_analysis"),
    str(ROOT),
]

from analysis_core.plan_resolver import resolve_analysis_plan_structured, runtime_profile_for_analysis
from analysis_core.base_agent import _apply_dependence_policy
from analysis_engine.evidence_graph import load_evidence_records
from analysis_engine.pairwise_tools import MetaforPairwiseMetaTool, PairwiseEffectTool
from analysis_engine.semantic_graph import SemanticEvidenceGraph
from extensions.analysis_worker import load_analysis_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, task, profile, payloads, studies = load_analysis_contract(args.input_dir)
    plan = resolve_analysis_plan_structured(protocol, task, profile)
    definition = next(item for item in plan.analyses if item.kind == "primary")
    if definition.dependence_policy != "clustered_effects_rho_0_6":
        raise ValueError("The supplied primary topic has no clustered-effects declaration.")
    records = load_evidence_records(payloads, studies)
    graph = SemanticEvidenceGraph(records, plan=plan)
    candidates = graph.topic_candidates(definition, plan)
    allowed = {record.evidence_id: record for group in candidates.groups for record in group.records}
    previous = json.loads((args.prior_analysis / "analysis_result.json").read_text())
    prior_topic = next(item for item in previous["analyses"]
                       if item["analysis"]["analysis_id"] == definition.analysis_id)
    prior_ids = prior_topic["selected_evidence_ids"]
    retained = _apply_dependence_policy(
        definition, [allowed[eid] for eid in prior_ids if eid in allowed], []
    )
    runtime = runtime_profile_for_analysis(definition, plan)
    poolability = PairwiseEffectTool().qualify(runtime, retained)
    effects = poolability.study_effects
    if len({item.study_id for item in effects}) < 3:
        raise ValueError("Fewer than three independent studies survived the declared contrast gate.")
    calculator = MetaforPairwiseMetaTool()
    sensitivity = {}
    for marker in ("0_3", "0_6", "0_9"):
        alternate = runtime.model_copy(update={
            "pooling_scale": f"log_ratio_inverse_variance__clustered_rho_{marker}"
        })
        result = calculator.invoke(alternate, effects)
        sensitivity[marker.replace("_", ".")] = result.model_dump(mode="json")
    output = {
        "schema_version": "saved_review_clustered_replay/1.0",
        "status": "local_replay_not_fresh_reviewer",
        "case_id": protocol.get("case_id"),
        "analysis_id": definition.analysis_id,
        "source_review": str((args.prior_analysis / "analysis_result.json").resolve()),
        "original_selected_count": len(prior_ids),
        "graph_eligible_count": len(allowed),
        "retained_count": len(retained),
        "independent_study_count": len({item.study_id for item in effects}),
        "selected_evidence_ids": [item.evidence_id for item in effects],
        "prior_selections_not_retained": sorted(set(prior_ids) - {item.evidence_id for item in effects}),
        "sensitivity_by_assumed_within_study_rho": sensitivity,
        "limitations": [
            "Reviewer selections were reused; a new model review was not run.",
            "No gold labels were used to select effects.",
            "Source contrast labels may omit or misclassify valid estimates; inspect original reports before publication.",
            "HR and RR are combined on a generic log-ratio scale, not claimed to be identical measures.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "status", "original_selected_count", "retained_count", "independent_study_count"
    )}, ensure_ascii=False))
    for rho, value in sensitivity.items():
        print(f"rho={rho}: {value['pooled_effect']:.4f} "
              f"[{value['ci_lower']:.4f}, {value['ci_upper']:.4f}]")


if __name__ == "__main__":
    main()
