"""Semantic: source-predicate graph retrieval with one auditable full-graph retry."""

import json
from pathlib import Path

from analysis_core.plan_resolver import runtime_profile_for_analysis
from analysis_core.base_agent import _apply_dependence_policy
from analysis_core.models import ToolTraceEntry

from .retrieval_artifacts import graph_artifact, embedding_retrieval_artifact
from .semantic_graph import SemanticEvidenceGraph, SemanticRetriever
from .pipeline_agent import PipelineAnalysisAgent, _apply_categorical_high_low_rule
from .eligibility import is_categorical_high_low


class GraphAnalysisAgent(PipelineAnalysisAgent):
    def __init__(self, *, audit_dir: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        facet_path = getattr(self.runtime_facet_annotator, "artifact_path", None)
        self.audit_dir = audit_dir or (facet_path.parent if facet_path else None)
        self.retrieval_audit = {}

    def run(self, **kwargs):
        result = super().run(**kwargs)
        for topic in result.analyses:
            candidates = (self.last_candidate_sets or {}).get(topic.analysis.analysis_id)
            if candidates is None:
                continue
            counts = {
                "candidate_record_count": sum(len(group.records) for group in candidates.groups),
                "candidate_context_count": len(candidates.groups),
                "candidate_study_count": len({group.study_id for group in candidates.groups}),
            }
            topic.candidate_inventory = topic.candidate_inventory.model_copy(update=counts)
            memory = result.memory.topics[topic.analysis.analysis_id]
            for key, value in counts.items():
                setattr(memory, key, value)
        for topic_id, audit in self.retrieval_audit.items():
            result.trace.append(ToolTraceEntry(
                tool_name="review_with_graph_expansion_semantic", status="completed",
                summary=f"{topic_id}: {len(audit['rounds'])} review round(s); {audit['stop_reason']}.",
            ))
        return result

    def _save_audit(self):
        if self.audit_dir is None:
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": "semantic_source_semantics_graph_with_expansion",
            "graph_inventory": self.last_graph.topic_audit if self.last_graph else {},
            "topic_reviews": self.retrieval_audit,
        }
        temporary = self.audit_dir / "retrieval_expansion_audit.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.audit_dir / "retrieval_expansion_audit.json")

    def _build_graph(self, records, plan, runtime_facets):
        self.retrieval_audit = {}
        graph = SemanticEvidenceGraph(records, plan=plan, runtime_facets=runtime_facets)
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            (self.audit_dir / "graph.json").write_text(
                json.dumps(graph_artifact(graph), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        return graph

    def _make_retriever(self, graph):
        return SemanticRetriever(graph, self.embeddings, audit_callback=self._save_audit)

    def _review_candidates(self, definition, candidates, plan, retriever):
        rounds = []
        self.retrieval_audit[definition.analysis_id] = {"rounds": rounds, "status": "running"}
        for attempt in range(2):
            if "Analysis eligibility: categorical_highest_vs_lowest" in definition.selection_rules:
                original_count = sum(len(group.records) for group in candidates.groups)
                groups = []
                for group in candidates.groups:
                    records = [record for record in group.records if is_categorical_high_low(record)]
                    if records:
                        groups.append(group.model_copy(update={
                            "records": records,
                            "evidence_ids": [record.evidence_id for record in records],
                            "article_ids": sorted({record.article_id for record in records}),
                        }))
                candidates = candidates.model_copy(update={
                    "groups": groups,
                    "retrieval_trace": [*candidates.retrieval_trace,
                        f"Prespecified categorical highest-versus-lowest gate retained "
                        f"{sum(len(group.records) for group in groups)}/{original_count} Evidence."],
                })
            snapshot = embedding_retrieval_artifact({definition.analysis_id: candidates})["topics"][0]
            round_audit = {"round": attempt + 1, "retrieval": snapshot, "status": "reviewing"}
            rounds.append(round_audit)
            self._save_audit()
            # Quantitative feasibility is a tool result, not an LLM guess about
            # missing reported CIs or zero cells. This does not select evidence.
            if hasattr(self.reviewer, "set_tool_feedback"):
                profile = runtime_profile_for_analysis(definition, plan)
                records = {r.evidence_id: r for g in candidates.groups for r in g.records}
                feedback = []
                for record in records.values():
                    try:
                        checked = self.effect_tool.qualify(profile, [record])
                        feedback.extend(d.model_dump() for d in checked.decisions)
                    except (ValueError, TypeError, ZeroDivisionError, NotImplementedError) as exc:
                        feedback.append({"evidence_id": record.evidence_id, "status": "validation_failed",
                                         "reasons": [str(exc)]})
                round_audit["quantitative_preflight"] = feedback
                self.reviewer.set_tool_feedback(definition.analysis_id, feedback)
                self._save_audit()
            review = self.reviewer.review(definition, candidates)
            selected_ids = set(review.selected_evidence_ids)
            records = {
                record.evidence_id: record for group in candidates.groups for record in group.records
                if record.evidence_id in selected_ids
            }
            selected = _apply_dependence_policy(definition, list(records.values()), review.assignments)
            selected = _apply_categorical_high_low_rule(definition, selected, review.assignments)
            poolability = self.effect_tool.qualify(runtime_profile_for_analysis(definition, plan), selected)
            studies = len({effect.study_id for effect in poolability.study_effects})
            requested = getattr(review, "request_expansion", False)
            needs_more = studies < definition.minimum_studies or requested
            round_audit.update(
                status="reviewed", review=review.model_dump(),
                poolable_study_count=studies,
                trigger=("insufficient_poolable_studies" if studies < definition.minimum_studies
                         else "reviewer_requested" if requested else "none"),
            )
            if not needs_more:
                stop = "sufficient_local_review"
                break
            seen = {record.evidence_id for group in candidates.groups for record in group.records}
            if seen == {record.evidence_id for record in retriever.graph.records}:
                stop = "complete_source_graph_reviewed"
                break
            if attempt == 1:
                stop = "expansion_limit_reached"
                break
            self._progress(
                f"[expand_retrieval] {definition.analysis_id}: {round_audit['trigger']}; "
                "restoring full source graph and re-reviewing all candidates."
            )
            round_audit["status"] = "expanding"
            self._save_audit()
            candidates = retriever.expand(definition, plan)
            round_audit["status"] = "reviewed"
        self.retrieval_audit[definition.analysis_id].update(status="completed", stop_reason=stop)
        self._save_audit()
        # Return the last adjudication, not a union of decisions from different
        # scopes: the expanded reviewer may replace an earlier competing model.
        return review, candidates
