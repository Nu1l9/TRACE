"""Offline structural checks for the frozen portable analysis component."""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract_adapter import load_analysis_contract


def main() -> None:
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        py_compile.compile(str(path), doraise=True)
    protocol, task, profile, payloads, _ = load_analysis_contract(ROOT / "examples")
    assert protocol["protocol_id"] == "demo_protocol"
    assert task["task_id"] == "demo_analysis_task"
    assert profile["profile_version"].startswith("2")
    assert sum(len(item["evidence"]) for item in payloads) == 2
    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and any(token in path.name.casefold() for token in ("gold", "evaluator"))
    ]
    if forbidden:
        raise AssertionError(f"Release must not contain private benchmark artifacts: {forbidden}")
    print(json.dumps({"status": "ok", "python_files": len(python_files), "gold_accessed": False}))


if __name__ == "__main__":
    main()
