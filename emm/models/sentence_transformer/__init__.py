"""Sentence transformer functionality for entity matching.

Note: This module assumes sentence-transformers is installed.
Import checks are handled at the package level in emm.models.__init__.py
"""

from __future__ import annotations

from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
from emm.models.sentence_transformer.tuning import TuningConfig, SentenceTransformerTuner

__all__ = [
    "BaseSentenceTransformerComponent",
    "TuningConfig",
    "SentenceTransformerTuner"
] 