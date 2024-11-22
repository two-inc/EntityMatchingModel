"""Sentence transformer functionality for entity matching.

This module requires the sentence-transformers package.
Install with: pip install emm[transformers]
"""

from __future__ import annotations

__all__ = []

try:
    from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
    __all__.append("BaseSentenceTransformerComponent")
except ImportError:
    pass

try:
    from emm.models.sentence_transformer.tuning import TuningConfig, SentenceTransformerTuner
    __all__.extend(["TuningConfig", "SentenceTransformerTuner"])
except ImportError:
    pass