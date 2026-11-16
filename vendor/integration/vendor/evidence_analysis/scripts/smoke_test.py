"""Run an API-free synthetic end-to-end smoke test of the frozen analysis core."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_engine.method_selector import DeterministicMethodSelector
from analysis_engine.planner import DeterministicAnalysisPlanner
from analysis_engine.reporting_bias_selector import DeterministicReportingBiasSelector
from analysis_engine.retrieval import HashEmbeddingBackend
from analysis_engine.runtime_facets import RuntimeFacetResult
from analysis_engine.topic_reviewer import DeterministicTopicReviewer
from analysis_engine.graph_agent import GraphAnalysisAgent
from contract_adapter import load_analysis_contract


class EmptySourceFacetAnnotator:
    """Offline test double: it neither classifies nor excludes Evidence."""

    mode = "offline_empty_source_facets"
    calls_made = 0

    def annotate(self, plan, records):
        return RuntimeFacetResult(mode=self.mode, note="Smoke-test fixture; no source facets asserted.")


def main() -> None:
    protocol, task, profile, payloads, studies = load_analysis_contract(ROOT / "examples")
    with tempfile.TemporaryDirectory(prefix="analysis_analysis_smoke_") as temporary:
        output_dir = Path(temporary)
        agent = GraphAnalysisAgent(
            embeddings=HashEmbeddingBackend(),
            planner=DeterministicAnalysisPlanner(),
            reviewer=DeterministicTopicReviewer(),
            method_selector=DeterministicMethodSelector(),
            reporting_bias_selector=DeterministicReportingBiasSelector(),
            runtime_facet_annotator=EmptySourceFacetAnnotator(),
            pre_review_filter_mode="semantic_semantic",
            audit_dir=output_dir,
        )
        result = agent.run(
            protocol=protocol,
            task=task,
            analysis_profile=profile,
            extractor_payloads=payloads,
            study_by_article=studies,
            reporting_bias_artifact_dir=output_dir / "reporting_bias",
            forest_plot_artifact_dir=output_dir / "forest_plots",
        )
        topic = result.analyses[0]
        assert result.status == "completed", result.status
        assert topic.statistics is not None
        assert topic.forest_plot is not None and topic.forest_plot.status == "completed"
        assert (output_dir / "graph.json").exists()
        assert (output_dir / "retrieval_expansion_audit.json").exists()
    print("smoke_test=ok; remote_calls=0; gold_accessed=false")


if __name__ == "__main__":
    main()
