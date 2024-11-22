"""Optional model extensions for entity matching.

This module provides optional model-based features that can be enabled by installing
additional dependencies:

1. Sentence Transformers: Install with `pip install emm[transformers]`
2. Model Tuning: Install with `pip install emm[tuning]`
"""

from __future__ import annotations

__all__ = []

# Let Python's import system handle dependency errors naturally
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