"""Smoke tests for the installable package's Hybrid-first public workflow."""

from __future__ import annotations

import json
import gc
import tempfile
import unittest
from pathlib import Path

from miners.workflow import (
    HybridReviewConfig,
    HybridReviewWorkflow,
    CorpusPaths,
    build_bm25_index,
    prepare_document_store,
)


class HybridReviewWorkflowTest(unittest.TestCase):
    def test_bm25_workflow_starts_with_retrieval_and_writes_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.json"
            papers = root / "papers.json"
            test_ids = root / "test_ids.json"
            store = root / "cache" / "documents.sqlite"
            bm25 = root / "cache" / "bm25"

            records = [
                {
                    "ID": 101,
                    "title": "Zinc supplementation in children",
                    "abstract": "A randomized trial of zinc supplementation for childhood diarrhea.",
                    "year": "2020",
                    "sections": {"methods": "Children were randomly assigned to zinc or placebo."},
                },
                {
                    "ID": 102,
                    "title": "Unrelated imaging study",
                    "abstract": "Magnetic resonance imaging methods in adults.",
                    "year": "2020",
                    "sections": {},
                },
            ]
            corpus.write_text(
                "[\n" + ",\n".join(json.dumps(record) for record in records) + "\n]\n",
                encoding="utf-8",
            )
            papers.write_text(
                json.dumps(
                    [
                        {
                            "ID": 1,
                            "Title": "Zinc and childhood diarrhea",
                            "Research_Question": "Does zinc supplementation reduce childhood diarrhea?",
                            "Population": "Children with diarrhea",
                            "Intervention": "Zinc supplementation",
                            "Comparison": "Placebo",
                            "Outcome": "Diarrhea",
                            "search_end_date": "2024-12-31",
                            "matched_corpus_ids": [101],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            test_ids.write_text("[1]", encoding="utf-8")

            prepare_document_store(corpus, store)
            build_bm25_index(store, bm25, min_df=1)
            workflow = HybridReviewWorkflow(
                HybridReviewConfig(
                    paths=CorpusPaths(
                        papers=papers,
                        test_ids=test_ids,
                        corpus=corpus,
                        store=store,
                        bm25_index=bm25,
                        dense_index=root / "unused-dense-index",
                    ),
                    output_dir=root / "runs",
                    mode="bm25",
                    per_query_k=2,
                    fusion_k=2,
                    final_k=2,
                )
            )

            result = workflow.run_one(1)

            self.assertTrue(result.results_path.is_file())
            self.assertTrue(result.report_path.is_file())
            self.assertTrue(result.manual_review_path.is_file())
            self.assertTrue(result.metrics_path.is_file())
            self.assertEqual(result.metrics["retrieved_count"], 1)
            self.assertEqual(result.metrics["retrieval_recall@5"], 1.0)
            payload = json.loads(result.results_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_config"]["pipeline"], "miners_hybrid_retrieval_to_pico_screening")
            self.assertEqual([item["corpus_id"] for item in payload["retrieved_papers"]], [101])

            # NumPy keeps Windows memory-map file handles open until the index is released.
            del workflow
            gc.collect()


if __name__ == "__main__":
    unittest.main()
