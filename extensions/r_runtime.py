"""Reliable, auditable execution for the Analysis R statistical helpers."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Sequence


def r_timeout_seconds() -> float:
    """Use a separate timeout for local statistics, independent of LLM calls."""
    return float(os.getenv("META_PIPELINE_R_TIMEOUT_SECONDS", "300"))


def run_rscript(command: Sequence[str], *, purpose: str, attempts: int = 2) -> subprocess.CompletedProcess[str]:
    """Run one R helper in a clean session and retry a genuinely stuck process.

    ``Rscript --vanilla`` prevents user/workspace startup profiles from being
    loaded.  A timeout kills that process before a fresh attempt; statistical
    inputs and scripts remain identical across attempts.
    """
    argv = list(command)
    if len(argv) < 2:
        raise ValueError("R command is incomplete")
    if "--vanilla" not in argv[1:]:
        argv.insert(1, "--vanilla")
    timeout = r_timeout_seconds()
    last_timeout: subprocess.TimeoutExpired | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
                start_new_session=True,
            )
            elapsed = time.monotonic() - started
            print(
                f"[r-runtime] {purpose}: completed attempt={attempt}/{attempts} "
                f"elapsed={elapsed:.1f}s returncode={completed.returncode}",
                flush=True,
            )
            return completed
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            print(
                f"[r-runtime] {purpose}: timeout after {timeout:g}s "
                f"attempt={attempt}/{attempts}; restarting a clean R process",
                flush=True,
            )
    raise RuntimeError(
        f"{purpose}: R process timed out {attempts} times after {timeout:g}s each"
    ) from last_timeout
