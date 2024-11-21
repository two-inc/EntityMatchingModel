try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

if SENTENCE_TRANSFORMERS_AVAILABLE:
    from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
    __all__ = ["BaseSentenceTransformerComponent"]
else:
    __all__ = [] 