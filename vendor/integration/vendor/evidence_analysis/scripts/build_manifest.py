"""Write a deterministic inventory and SHA-256 hashes for the frozen release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".env", "RELEASE_MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    payload = {
        "release": "evidence_analysis",
        "architecture": "semantic_source_semantic_graphrag_with_expandable_review",
        "input_contract": "INTERFACE_CONTRACT.md",
        "contains_gold": False,
        "contains_api_keys": False,
        "files": files,
    }
    (ROOT / "RELEASE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
