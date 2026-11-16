from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceLocations(BaseModel):
    """High-recall source locations. This is deliberately not an extraction plan."""

    model_config = ConfigDict(extra="forbid")

    source_ids: list[str] = Field(default_factory=list)
    unresolved_location_terms: list[str] = Field(default_factory=list)
