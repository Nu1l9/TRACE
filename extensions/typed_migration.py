"""Non-destructive Baseline-to-Typed request migration for benchmark case studies.

The frozen contract deliberately keeps the original early benchmark case IDs
on Baseline.  A migration therefore creates a new, traceable Typed case ID and leaves
the original request untouched.  It changes no PICO, target, candidate, or
analysis field; the only operational change is the extractor runtime.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .common import ROOT, digest, read, require, write
from trace_runtime.contracts import normalize_request


SCHEMA = "trace_legacy/typed_migration/1.0"
TYPED_TEMPLATE_FAMILIES = {
    "adjusted_ratio_effect",
    "reported_ratio_effect",
    "time_to_event",
    "reported_standardized_mean_difference",
    "binary_arm_outcome",
    "binary_arm_counts",
    "continuous_arm_summary",
    "continuous_change_score_summary",
    "event_counts_with_person_time",
    "percentage_only",
    "generic_text_evidence",
    "complete_genotype_case_control_counts",
    "allele_frequency_summary_only",
    "independent_continuous_arms",
    "single_arm_continuous_summary",
    "diagnostic_four_cell_counts",
    "single_arm_event_counts",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _migration_case_id(case_id: str) -> str:
    return f"{case_id}_typed_migration"


def compatibility(raw: dict[str, Any]) -> dict[str, Any]:
    """State whether every declared extraction target has a Typed template."""
    targets = raw.get("protocol", {}).get("targets", [])
    unsupported = {
        target.get("target_id", "unknown"): sorted(
            set(target.get("accepted_data_families", [])) - TYPED_TEMPLATE_FAMILIES
        )
        for target in targets
    }
    unsupported = {target_id: families for target_id, families in unsupported.items() if families}
    return {
        "status": "ready" if not unsupported else "blocked",
        "unsupported_data_families": unsupported,
        "reason": (
            None if not unsupported else
            "The frozen Typed template runtime has no source-data template for the listed family. "
            "Do not replace the Baseline benchmark until an explicit compatible template is added."
        ),
    }


def migrate_request(source: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate a source Baseline request and build one separate Typed request if safe."""
    source = Path(source).resolve()
    raw = read(source)
    normalized = normalize_request(raw, source.parent)
    require(normalized["options"]["extraction_runtime"] == "baseline", "Typed migration accepts a Baseline source request only")
    assessment = compatibility(normalized)
    record = {
        "schema_version": SCHEMA,
        "source_request": str(source),
        "source_request_sha256": _sha256(source),
        "source_case_id": normalized["case_id"],
        "target_families": {
            target["target_id"]: list(target["accepted_data_families"])
            for target in normalized["protocol"]["targets"]
        },
        **assessment,
    }
    if assessment["status"] != "ready":
        return record, None

    migrated = deepcopy(raw)
    migrated_case_id = _migration_case_id(normalized["case_id"])
    migrated["case_id"] = migrated_case_id
    migrated["protocol"]["case_id"] = migrated_case_id
    migrated["protocol"]["protocol_id"] = f"{normalized['protocol']['protocol_id']}.typedmigration"
    migrated.setdefault("options", {})["extraction_runtime"] = "typed"
    # Provider/model choice remains project-level in Graph. The source JSON was
    # intentionally neutral, so a migration must stay neutral as well.
    migrated["options"].pop("provider", None)
    migrated["options"].pop("model", None)
    checked = normalize_request(migrated, source.parent)
    record.update({
        "migrated_case_id": checked["case_id"],
        "migrated_protocol_id": checked["protocol"]["protocol_id"],
        "changed_fields": [
            "case_id", "protocol.case_id", "protocol.protocol_id", "options.extraction_runtime",
        ],
        "source_pico_fingerprint": digest(normalized["protocol"]["pico"]),
        "migrated_pico_fingerprint": digest(checked["protocol"]["pico"]),
        "candidate_article_count": len(checked["documents"]),
        "validation": "normalize_request passed with local source paths; no model calls made",
    })
    require(record["source_pico_fingerprint"] == record["migrated_pico_fingerprint"], "migration changed PICO")
    require(len(normalized["documents"]) == len(checked["documents"]), "migration changed candidates")
    checked["options"].pop("provider", None)
    checked["options"].pop("model", None)
    return record, checked


def write_migration(source: Path, output: Path) -> dict[str, Any]:
    """Write a migration record and, when eligible, a validated Typed request."""
    record, migrated = migrate_request(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "migration_record.json", record)
    if migrated is not None:
        write(output / "request.typed.json", migrated)
    return record


def benchmark_requests() -> list[Path]:
    import os
    root = Path(os.environ.get("META_BENCHMARK_ROOT", str(ROOT.parent / "EviSynth-Bench"))) / "cases"
    require(root.is_dir(), "Benchmark cases directory missing; set META_BENCHMARK_ROOT or use --request")
    return sorted(root.glob("ma_????/request.json"))


def migrate_all_benchmark(output: Path) -> dict[str, Any]:
    """Generate one immutable migration folder per frozen Baseline benchmark case."""
    output = Path(output)
    require(not output.exists(), "migration output already exists; choose a new output directory")
    rows = []
    for source in benchmark_requests():
        raw = read(source)
        if raw.get("options", {}).get("extraction_runtime") != "baseline":
            continue
        case_output = output / raw["case_id"]
        rows.append(write_migration(source, case_output))
    manifest = {
        "schema_version": SCHEMA,
        "mode": "all_frozen_baseline_benchmark_cases",
        "ready_case_ids": [row["source_case_id"] for row in rows if row["status"] == "ready"],
        "blocked_case_ids": [row["source_case_id"] for row in rows if row["status"] != "ready"],
        "records": [
            {
                "source_case_id": row["source_case_id"],
                "migrated_case_id": row.get("migrated_case_id"),
                "status": row["status"],
                "unsupported_data_families": row["unsupported_data_families"],
            }
            for row in rows
        ],
        "policy": "Original Baseline requests are untouched. Run only request.typed.json in a new Graph output directory and compare it with its source Baseline run.",
    }
    write(output / "migration_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate separate Typed migration requests for Baseline benchmark cases")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="One Baseline request JSON")
    source.add_argument("--all-benchmark", action="store_true", help="Migrate every Baseline benchmark request that has a Typed template mapping")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.all_benchmark:
        result = migrate_all_benchmark(args.output)
    else:
        require(not args.output.exists(), "migration output already exists; choose a new output directory")
        result = write_migration(args.request, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
