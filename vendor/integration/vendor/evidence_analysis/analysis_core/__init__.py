"""Analysis Agent structured: primary, secondary, subgroup and sensitivity planning.

The package consumes only benchmark task/protocol/manifest inputs and extractor
outputs. Gold files are deliberately outside its runtime interface.
"""

from .base_agent import BaseAnalysisAgent

__all__ = ["BaseAnalysisAgent"]
