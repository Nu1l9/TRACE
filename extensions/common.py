"""Resolve runtime paths and contracts from this repository checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'vendor' / 'integration'
ANALYSIS = BASELINE / 'vendor' / 'evidence_analysis'


def activate():
    for directory in (BASELINE, ANALYSIS):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


activate()
from metapipeline.contracts import ContractError, digest, read, require, write
