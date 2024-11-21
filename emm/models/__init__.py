from __future__ import annotations

__all__ = []

# Core functionality is empty - models are all optional features
# This makes it explicit that this module only provides optional extensions

# Primary optional feature: Sentence Transformers
try:
    import sentence_transformers
    from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
    __all__.append("BaseSentenceTransformerComponent")

    # Secondary optional feature: Model Tuning
    # Only available if sentence transformers is installed
    try:
        import lightning
        import wandb
        from emm.models.sentence_transformer.tuning import SentenceTransformerTuner, TuningConfig
        __all__.extend(["SentenceTransformerTuner", "TuningConfig"])
    except ImportError:
        pass  # Tuning features unavailable

except ImportError:
    pass  # Transformer features unavailable 