"""Sentence transformer functionality for entity matching.

This module requires the sentence-transformers package.
Install with: pip install emm[transformers]
"""

from __future__ import annotations

from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent

__all__ = ["BaseSentenceTransformerComponent"]

# Import tuning if available
try:
    from emm.models.sentence_transformer.tuning import TuningConfig, SentenceTransformerTuner
    __all__ += ["TuningConfig", "SentenceTransformerTuner"]
except ImportError:
    pass 