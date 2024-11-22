"""Optional model extensions for entity matching.

This module provides optional model-based features that can be enabled by installing
additional dependencies:

1. Sentence Transformers: Install with `pip install emm[transformers]`
2. Model Tuning: Install with `pip install emm[tuning]`
"""

from __future__ import annotations

__all__ = []

# Core functionality is empty - models are all optional features
# This makes it explicit that this module only provides optional extensions

def _import_sentence_transformers():
    """Helper to import sentence transformer components"""
    try:
        import sentence_transformers
        from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
        __all__.append("BaseSentenceTransformerComponent")
        return True
    except ImportError:
        return False

def _import_tuning():
    """Helper to import tuning components"""
    try:
        import lightning
        import wandb
        from emm.models.sentence_transformer.tuning import SentenceTransformerTuner, TuningConfig
        __all__.extend(["SentenceTransformerTuner", "TuningConfig"])
        return True
    except ImportError:
        return False

# Try to import optional features
HAS_TRANSFORMERS = _import_sentence_transformers()
HAS_TUNING = HAS_TRANSFORMERS and _import_tuning()  # Tuning requires transformers 