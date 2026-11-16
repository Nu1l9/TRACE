"""Serializable inspection artefacts for the concrete GraphRAG retrieval run."""

from __future__ import annotations

from typing import Any


def graph_artifact(graph) -> dict[str, Any]:
    """Export the exact typed NetworkX graph without relying on a viewer format."""
    return {
        "graph_type": type(graph.graph).__name__,
        "nodes": [
            {"node_id": node_id, **dict(data)}
            for node_id, data in graph.graph.nodes(data=True)
        ],
        "edges": [
            {"source": source, "target": target, "edge_key": key, **dict(data)}
            for source, target, key, data in graph.graph.edges(keys=True, data=True)
        ],
    }


def embedding_retrieval_artifact(candidate_sets: dict[str, Any]) -> dict[str, Any]:
    """Export topic-local graph contexts, including persisted embedding rank/score."""
    return {
        "topics": [
            {
                "analysis_id": analysis_id,
                "retrieval_trace": candidate_set.retrieval_trace,
                "contexts": [
                    {
                        "context_id": group.group_id,
                        "study_id": group.study_id,
                        "evidence_ids": group.evidence_ids,
                        "selection_context": group.selection_context,
                        "retrieval_reasons": group.retrieval_reasons,
                    }
                    for group in candidate_set.groups
                ],
            }
            for analysis_id, candidate_set in candidate_sets.items()
        ]
    }
