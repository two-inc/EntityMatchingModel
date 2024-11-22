"""Model tuning functionality for sentence transformers.

This module requires additional dependencies:
- sentence-transformers
- lightning 
- wandb

Install with: pip install emm[tuning]
"""

from __future__ import annotations

from emm.models.sentence_transformer.tuning.config import TuningConfig
from emm.models.sentence_transformer.tuning.tuner import SentenceTransformerTuner

__all__ = ["TuningConfig", "SentenceTransformerTuner"]
 